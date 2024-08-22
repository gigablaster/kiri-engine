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
use ash::vk;
use kiri_backend::{AttachmentClearValue, Image, MAX_COLOR_ATTACHMENTS};
use parking_lot::{Mutex, RwLock};

use crate::{
    BufferHandle, DescriptorHandle, DescriptorPool, DescriptorSetBuilder, DrawStream,
    DynamicGpuMemory, Error, ImageHandle, Renderer, Resolution, TempImageGuard, TempImagePool,
};

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub image: ImageHandle,
    pub layout: vk::ImageLayout,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub clear: vk::ClearValue,
}

pub struct RenderPassRecorder<'a> {
    context: &'a RenderContext<'a>,
    color: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth: Option<RenderTarget>,
    streams: Vec<DrawStream>,
    temp_descriptors: Vec<DescriptorHandle>,
}

impl<'a> RenderPassRecorder<'a> {
    fn new(
        context: &'a RenderContext<'a>,
        color: &[RenderTarget],
        depth: Option<RenderTarget>,
    ) -> Self {
        Self {
            context,
            color: color.iter().copied().collect(),
            depth,
            streams: Default::default(),
            temp_descriptors: Default::default(),
        }
    }

    pub fn draw(&mut self, stream: DrawStream) {
        self.streams.push(stream);
    }

    pub fn push<T: Copy>(&self, data: &[T]) -> Result<usize, Error> {
        self.context.dynamic.push(data)
    }

    pub fn get_temprary_buffer(&self) -> BufferHandle {
        self.context.dynamic.get_buffer_handle()
    }

    pub fn submit(self) {
        self.context.passes.lock().push(RenderPass {
            color: self.color,
            depth: self.depth,
            streams: self.streams,
        });
        self.context
            .temp_descriptors
            .lock()
            .extend(self.temp_descriptors.iter());
    }

    pub fn get_image(
        &self,
        resolution: Resolution,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<TempImageGuard<'a>, Error> {
        self.context.image_pool.get(
            &self.context.renderer,
            resolution,
            format,
            usage,
            self.context.backbuffer,
        )
    }

    pub fn allocate_descriptor_set(
        &mut self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let handle = self.context.descriptors.write().push(
            vk::DescriptorSet::null(),
            builder.build(&self.context.renderer.device)?,
        );
        self.temp_descriptors.push(handle);
        Ok(handle)
    }
}

struct RenderPass {
    color: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth: Option<RenderTarget>,
    streams: Vec<DrawStream>,
}

pub struct RenderContext<'a> {
    renderer: &'a Renderer,
    image_pool: &'a TempImagePool,
    backbuffer: &'a Image,
    dynamic: &'a DynamicGpuMemory,
    passes: Mutex<Vec<RenderPass>>,
    descriptors: &'a RwLock<DescriptorPool>,
    pub(super) temp_descriptors: Mutex<Vec<DescriptorHandle>>,
}

impl RenderTarget {
    fn new(image: ImageHandle, layout: vk::ImageLayout) -> Self {
        Self {
            image,
            layout,
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::STORE,
            clear: Default::default(),
        }
    }

    pub fn color(image: ImageHandle) -> Self {
        Self::new(image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    }

    pub fn depth(image: ImageHandle) -> Self {
        Self::new(image, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    }

    pub fn load(mut self) -> Self {
        self.load = vk::AttachmentLoadOp::LOAD;
        self
    }

    pub fn discard(mut self) -> Self {
        self.store = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    pub fn clear(mut self, value: AttachmentClearValue) -> Self {
        self.load = vk::AttachmentLoadOp::CLEAR;
        self.clear = value.into();
        self
    }
}

impl<'a> RenderContext<'a> {
    pub(crate) fn new(
        renderer: &'a Renderer,
        image_pool: &'a TempImagePool,
        dynamic: &'a DynamicGpuMemory,
        descriptors: &'a RwLock<DescriptorPool>,
        backbuffer: &'a Image,
    ) -> Self {
        Self {
            renderer,
            image_pool,
            dynamic,
            passes: Default::default(),
            descriptors,
            temp_descriptors: Default::default(),
            backbuffer,
        }
    }

    pub fn create_pass(
        &'a self,
        color: &[RenderTarget],
        depth: Option<RenderTarget>,
    ) -> RenderPassRecorder {
        RenderPassRecorder {
            context: self,
            color: color
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>(),
            depth,
            streams: Default::default(),
            temp_descriptors: Default::default(),
        }
    }
}
