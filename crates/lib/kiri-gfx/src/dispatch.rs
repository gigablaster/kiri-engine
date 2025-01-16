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

use std::sync::Arc;

use kiri_backend::{
    ash::{self, vk},
    Image,
};

use crate::{
    BufferHandle, BufferPool, DescriptorHandle, DescriptorPool, Error, ImageHandle, ImagePool,
};

#[derive(Debug)]
pub struct RenderResourceResolver<'a> {
    buffers: &'a BufferPool,
    images: &'a ImagePool,
    descriptors: &'a DescriptorPool,
    pub empty_descriptor_set: vk::DescriptorSet,
    pub backbuffer: &'a Image,
}

impl<'a> RenderResourceResolver<'a> {
    pub(super) fn new(
        backbuffer: &'a Image,
        buffers: &'a BufferPool,
        images: &'a ImagePool,
        descriptors: &'a DescriptorPool,
        empty_descriptor_set: vk::DescriptorSet,
    ) -> Self {
        Self {
            buffers,
            images,
            descriptors,
            empty_descriptor_set,
            backbuffer,
        }
    }

    pub fn resolve_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, Error> {
        self.buffers
            .get(handle)
            .copied()
            .ok_or(Error::InvalidBufferHandle(handle))
    }

    pub fn resolve_image_view(&self, handle: ImageHandle) -> Result<vk::ImageView, Error> {
        self.images
            .get(handle)
            .copied()
            .ok_or(Error::InvalidImageHandle(handle))
    }

    pub fn resolve_image(&self, handle: ImageHandle) -> Result<&'a Image, Error> {
        Ok(self
            .images
            .get_cold(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .0
            .as_ref())
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
