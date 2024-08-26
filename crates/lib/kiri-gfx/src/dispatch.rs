// Copyright (C) 2024 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use arrayvec::ArrayVec;
use ash::vk::{self};
use kiri_backend::{Image, ImageAttachment, RenderPass, MAX_ATTACHMENTS, MAX_COLOR_ATTACHMENTS};

use crate::{
    BufferHandle, BufferPool, DescriptorHandle, DescriptorPool, DrawStream, DynamicGpuMemory,
    Error, ImageHandle, ImagePool, PipelineHandle, PipelinePool, RenderPassHandle, RenderPassPool,
};

#[derive(Debug)]
pub struct RenderResourceResolver<'a> {
    pub(super) buffers: &'a BufferPool,
    pub(super) images: &'a ImagePool,
    pub(super) descriptors: &'a DescriptorPool,
    pub(super) render_passes: &'a RenderPassPool,
    pub(super) pipelines: &'a PipelinePool,
    pub empty_descriptor_set: vk::DescriptorSet,
}

impl<'a> RenderResourceResolver<'a> {
    pub fn resolve_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, Error> {
        self.buffers
            .get(handle)
            .copied()
            .ok_or(Error::InvalidBufferHandle(handle))
    }

    pub fn resolve_image(&self, handle: ImageHandle) -> Result<&Image, Error> {
        self.images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))
    }

    pub fn resolve_descriptor_set(
        &self,
        handle: DescriptorHandle,
    ) -> Result<vk::DescriptorSet, Error> {
        self.descriptors
            .get(handle)
            .copied()
            .ok_or(Error::InvalidDescriptorHandle(handle))
    }

    pub fn resolve_render_pass(&self, handle: RenderPassHandle) -> Result<&RenderPass, Error> {
        self.render_passes
            .get(handle)
            .ok_or(Error::InvalidRenderPassHandle(handle))
    }

    pub fn resolve_pipeline(
        &self,
        handle: PipelineHandle,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
        self.pipelines
            .get(handle)
            .copied()
            .ok_or(Error::InvalidPipelineHandle(handle))
    }
}

pub trait PassDispatcher {
    fn name(&self) -> &str;
    fn dispatch(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        resolver: &RenderResourceResolver,
    ) -> Result<(), Error>;
}

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub image: ImageHandle,
    pub clear: Option<vk::ClearValue>,
}

impl RenderTarget {
    pub fn new(image: ImageHandle) -> Self {
        Self { image, clear: None }
    }

    pub fn clear_color(mut self, value: [f32; 4]) -> Self {
        self.clear = Some(vk::ClearValue {
            color: vk::ClearColorValue { float32: value },
        });
        self
    }

    pub fn clear_depth_stencil(mut self, depth: f32, stencil: u32) -> Self {
        self.clear = Some(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
        });
        self
    }
}

pub struct RasterizerPassDispatcher {
    color_targets: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth_target: Option<RenderTarget>,
    render_pass: RenderPassHandle,
    streams: Vec<DrawStream>,
    area: Option<vk::Rect2D>,
    name: String,
}

impl RasterizerPassDispatcher {
    pub fn new(
        name: &str,
        render_pass: RenderPassHandle,
        color_targets: &[RenderTarget],
        depth_target: Option<RenderTarget>,
        streams: Vec<DrawStream>,
        area: Option<vk::Rect2D>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            render_pass,
            color_targets: color_targets.iter().copied().collect(),
            depth_target,
            streams,
            area,
        }
    }
}

impl PassDispatcher for RasterizerPassDispatcher {
    fn name(&self) -> &str {
        &self.name
    }

    fn dispatch(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        resolver: &RenderResourceResolver,
    ) -> Result<(), Error> {
        let mut attachments = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        for color in &self.color_targets {
            let image = resolver.resolve_image(color.image)?;
            attachments.push(ImageAttachment {
                image,
                aspect: vk::ImageAspectFlags::COLOR,
            });
        }
        if let Some(depth) = &self.depth_target {
            let image = resolver.resolve_image(depth.image)?;
            attachments.push(ImageAttachment {
                image,
                aspect: vk::ImageAspectFlags::DEPTH,
            });
        }

        let render_pass = resolver.resolve_render_pass(self.render_pass)?;
        let fbo = render_pass.fbo(&attachments)?;
        let render_area = self.area.unwrap_or(fbo.area());
        let clear_values = self
            .color_targets
            .iter()
            .map(|x| x.clear.unwrap_or(vk::ClearValue::default()))
            .chain(
                self.depth_target
                    .iter()
                    .map(|x| x.clear.unwrap_or(vk::ClearValue::default())),
            )
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
        let begine_info = vk::RenderPassBeginInfo::default()
            .framebuffer(fbo.raw)
            .render_area(render_area)
            .clear_values(&clear_values)
            .render_pass(render_pass.raw);
        unsafe {
            device.cmd_begin_render_pass(command_buffer, &begine_info, vk::SubpassContents::INLINE)
        };
        let mut curent_subpass = 0;
        for stream in &self.streams {
            if stream.subpass != curent_subpass {
                assert!(
                    stream.subpass == curent_subpass + 1,
                    "Skipping over subpasses aren't allowed"
                );
                curent_subpass = stream.subpass;
                unsafe { device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE) };
            }
            stream.execute(device, command_buffer, render_area, resolver)?;
        }
        unsafe { device.cmd_end_render_pass(command_buffer) };
        Ok(())
    }
}
