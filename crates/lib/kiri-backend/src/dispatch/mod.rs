// Copyright (C) 2025 gigablaster

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

mod copy_to_backbuffer;
mod render;
use copy_to_backbuffer::CopyToBackbufferPassDispatcher;
use parking_lot::{Mutex, MutexGuard, RwLockReadGuard, RwLockWriteGuard};
use render::*;
use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::vk;

use crate::{
    vulkan::{
        DescriptorHandle, DescriptorPool, DescriptorSetCreateDesc, Frame, GraphicsDevice,
        ImageHandle, ImageViewDesc, RenderResourceResolver, SwapchainImage, MAX_ATTACHMENTS,
    },
    DrawStream, Error,
};

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub image: ImageHandle,
    pub initial_layout: Option<vk::ImageLayout>,
    pub final_layout: Option<vk::ImageLayout>,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub clear: Option<vk::ClearValue>,
}

impl RenderTarget {
    pub fn new(image: ImageHandle) -> Self {
        Self {
            image,
            initial_layout: None,
            final_layout: None,
            clear: None,
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::STORE,
        }
    }

    pub fn initial_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.initial_layout = Some(layout);
        self
    }

    pub fn final_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.final_layout = Some(layout);
        self
    }

    pub fn clear_color(mut self, value: [f32; 4]) -> Self {
        self.load = vk::AttachmentLoadOp::CLEAR;
        self.clear = Some(vk::ClearValue {
            color: vk::ClearColorValue { float32: value },
        });
        self
    }

    pub fn clear_depth_stencil(mut self, depth: f32, stencil: u32) -> Self {
        self.load = vk::AttachmentLoadOp::CLEAR;
        self.clear = Some(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
        });
        self
    }

    pub fn discard(mut self) -> Self {
        self.store = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    pub fn load(mut self) -> Self {
        self.load = vk::AttachmentLoadOp::LOAD;
        self
    }

    fn build(
        &self,
        resolver: &RenderResourceResolver,
        aspect: vk::ImageAspectFlags,
        layout: vk::ImageLayout,
    ) -> Result<vk::RenderingAttachmentInfo, Error> {
        let view = resolver.resolve_image_view(self.image, ImageViewDesc::new(aspect))?;
        let info = vk::RenderingAttachmentInfo::default()
            .image_layout(layout)
            .image_view(view)
            .load_op(self.load)
            .store_op(self.store)
            .clear_value(self.clear.unwrap_or_default());
        Ok(info)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageDependency {
    pub aspect: vk::ImageAspectFlags,
    pub image: ImageHandle,
    pub initial_layout: vk::ImageLayout,
    pub desired_layout: Option<vk::ImageLayout>,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
}

impl ImageDependency {
    pub fn color(image: ImageHandle) -> Self {
        Self {
            aspect: vk::ImageAspectFlags::COLOR,
            image,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            desired_layout: None,
            src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_access: vk::AccessFlags2::SHADER_READ,
            src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        }
    }

    pub fn depth(image: ImageHandle) -> Self {
        Self {
            aspect: vk::ImageAspectFlags::DEPTH,
            image,
            initial_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            desired_layout: None,
            src_access: vk::AccessFlags2::SHADER_READ,
            dst_access: vk::AccessFlags2::SHADER_READ,
            src_stage: vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        }
    }

    pub fn desired_layout(mut self, value: vk::ImageLayout) -> Self {
        self.desired_layout = Some(value);
        self
    }

    fn build<'a>(
        self,
        resolver: &RenderResourceResolver,
        desired_layout: vk::ImageLayout,
    ) -> Result<vk::ImageMemoryBarrier2<'a>, Error> {
        Ok(vk::ImageMemoryBarrier2::default()
            .old_layout(self.initial_layout)
            .new_layout(self.desired_layout.unwrap_or(desired_layout))
            .src_access_mask(self.src_access)
            .dst_access_mask(self.dst_access)
            .src_stage_mask(self.src_stage)
            .dst_stage_mask(self.dst_stage)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
            .image(resolver.resolve_image(self.image)?))
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

pub struct FrameDispatcher<'a> {
    device: &'a GraphicsDevice,
    frame: Arc<Frame>,
    passes: Vec<Box<dyn PassDispatcher>>,
    target: SwapchainImage<'a>,
    temp_descriptors: Mutex<Vec<DescriptorHandle>>,
}

pub enum RenderArea {
    AllTarget,
    Area(vk::Rect2D),
}

impl From<RenderArea> for Option<vk::Rect2D> {
    fn from(value: RenderArea) -> Self {
        match value {
            RenderArea::AllTarget => None,
            RenderArea::Area(area) => Some(area),
        }
    }
}

pub enum RenderFrame<'a> {
    NeedRecreateSwapchain,
    Dispatch(FrameDispatcher<'a>),
}

impl<'a> FrameDispatcher<'a> {
    pub(crate) fn new(
        device: &'a GraphicsDevice,
        frame: Arc<Frame>,
        target: SwapchainImage<'a>,
    ) -> Self {
        Self {
            device,
            frame,
            passes: Default::default(),
            target,
            temp_descriptors: Default::default(),
        }
    }

    pub fn render_pass(
        &mut self,
        name: impl AsRef<str>,
        area: RenderArea,
        color_targets: impl IntoIterator<Item = RenderTarget>,
        depth_target: Option<RenderTarget>,
        reads: impl IntoIterator<Item = ImageDependency>,
        streams: impl IntoIterator<Item = DrawStream>,
    ) {
        self.passes.push(Box::new(RenderPassDispatcher::new(
            name,
            area,
            color_targets,
            depth_target,
            reads,
            streams,
        )))
    }

    pub fn temp_descriptors(
        &self,
        builder: DescriptorSetCreateDesc,
    ) -> Result<DescriptorHandle, Error> {
        let handle = self.device.descriptors().create_descriptor(builder)?;
        self.temp_descriptors.lock().push(handle);
        Ok(handle)
    }

    pub fn present(self, image: ImageHandle) -> Result<(), Error> {
        self.device.execute(
            self.frame,
            self.target,
            self.passes.into_iter().chain([
                Box::new(CopyToBackbufferPassDispatcher::new(image)) as Box<dyn PassDispatcher>
            ]),
            self.temp_descriptors.into_inner(),
        )
    }
}
