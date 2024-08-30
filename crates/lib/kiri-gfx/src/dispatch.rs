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

use ash::vk::{self};
use kiri_backend::Image;

use crate::{
    BufferHandle, BufferPool, DescriptorHandle, DescriptorPool, Error, ImageHandle, ImagePool,
    PipelineHandle, PipelinePool,
};

#[derive(Debug)]
pub struct RenderResourceResolver<'a> {
    pub(super) buffers: &'a BufferPool,
    pub(super) images: &'a ImagePool,
    pub(super) descriptors: &'a DescriptorPool,
    pub(super) pipelines: &'a PipelinePool,
    pub(super) empty_descriptor_set: vk::DescriptorSet,
    pub backbuffer: &'a Image,
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
