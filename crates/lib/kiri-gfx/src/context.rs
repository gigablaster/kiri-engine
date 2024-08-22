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
use kiri_backend::{AttachmentClearValue, Image, ImageViewDesc, MAX_COLOR_ATTACHMENTS};
use parking_lot::{Mutex, RwLock};

use crate::{
    BufferHandle, DescriptorHandle, DescriptorPool, DescriptorSetBuilder, DrawStream,
    DynamicGpuMemory, Error, ImageBarrier, ImageBarrierType, ImageHandle, ImagePool, Renderer,
    Resolution, TempImageGuard, TempImagePool,
};

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub image: ImageHandle,
    pub layout: vk::ImageLayout,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub clear: vk::ClearValue,
}

pub struct RenderPassBuilder<'a> {
    context: &'a RenderContext<'a>,
    color: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth: Option<RenderTarget>,
    streams: Vec<DrawStream>,
    descriptor_sets: Vec<DescriptorHandle>,
    image_barriers: Vec<ImageBarrier>,
}

impl<'a> RenderPassBuilder<'a> {
    pub fn draw(&mut self, stream: DrawStream) {
        self.streams.push(stream);
    }

    pub fn push<T: Copy>(&self, data: &[T]) -> Result<u32, Error> {
        self.context.dynamic.push(data)
    }

    pub fn get_temprary_buffer(&self) -> BufferHandle {
        self.context.dynamic.get_buffer_handle()
    }

    pub fn allocate_descriptor_set(
        &mut self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let handle = self.context.descriptors.write().push(
            vk::DescriptorSet::null(),
            builder.build(&self.context.renderer.device)?,
        );
        self.descriptor_sets.push(handle);
        Ok(handle)
    }

    pub fn image_barrier(
        &mut self,
        image: ImageHandle,
        ty: ImageBarrierType,
        aspect: vk::ImageAspectFlags,
    ) {
        self.image_barriers.push(ImageBarrier(image, ty, aspect));
    }

    pub fn build(self) -> RenderPass {
        RenderPass {
            color: self.color,
            depth: self.depth,
            streams: self.streams,
            descriptor_sets: self.descriptor_sets,
            image_barriers: self.image_barriers,
        }
    }
}

pub struct RenderPass {
    pub(super) color: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    pub(super) depth: Option<RenderTarget>,
    pub(super) streams: Vec<DrawStream>,
    pub(super) descriptor_sets: Vec<DescriptorHandle>,
    pub(super) image_barriers: Vec<ImageBarrier>,
}

pub struct RenderContext<'a> {
    renderer: &'a Renderer,
    image_pool: &'a TempImagePool,
    backbuffer: &'a Image,
    pub target: ImageHandle,
    dynamic: &'a DynamicGpuMemory,
    pub(super) passes: Mutex<Vec<RenderPass>>,
    descriptors: &'a RwLock<DescriptorPool>,
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

    pub(super) fn build(
        &self,
        images: &ImagePool,
        aspect: vk::ImageAspectFlags,
    ) -> Result<vk::RenderingAttachmentInfo, Error> {
        let image = images
            .get(self.image)
            .ok_or(Error::InvalidImageHandle(self.image))?;
        let view = image.view(ImageViewDesc::new(aspect))?;
        Ok(vk::RenderingAttachmentInfo::default()
            .clear_value(self.clear)
            .image_layout(self.layout)
            .image_view(view)
            .load_op(self.load)
            .store_op(self.store))
    }
}

impl<'a> RenderContext<'a> {
    pub(crate) fn new(
        renderer: &'a Renderer,
        image_pool: &'a TempImagePool,
        dynamic: &'a DynamicGpuMemory,
        descriptors: &'a RwLock<DescriptorPool>,
        backbuffer: &'a Image,
        target: ImageHandle,
    ) -> Self {
        Self {
            renderer,
            image_pool,
            dynamic,
            passes: Default::default(),
            descriptors,
            backbuffer,
            target,
        }
    }

    pub fn create_render_pass(
        &'a self,
        color: &[RenderTarget],
        depth: Option<RenderTarget>,
    ) -> RenderPassBuilder {
        RenderPassBuilder {
            context: self,
            color: color
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>(),
            depth,
            streams: Default::default(),
            descriptor_sets: Default::default(),
            image_barriers: Default::default(),
        }
    }

    pub fn get_image(
        &mut self,
        resolution: Resolution,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<TempImageGuard<'a>, Error> {
        let image =
            self.image_pool
                .get(&self.renderer, resolution, format, usage, self.backbuffer)?;
        Ok(image)
    }

    pub fn submit(&self, pass: RenderPass) {
        self.passes.lock().push(pass);
    }
}
