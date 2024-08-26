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

use std::{mem, sync::Arc};

use kiri_backend::{BufferCreateDesc, GpuAllocator};
use kiri_common::BumpAllocator;
use kiri_gfx::{BufferHandle, BufferPointer, BufferSlice, Renderer};

use crate::Error;

/// Static uniform allocator
///
/// Allocate uniforms of single type.
#[derive(Debug)]
pub struct ConstUniformBuffer {
    renderer: Arc<Renderer>,
    pub buffer: BufferHandle,
    allocator: BumpAllocator,
}

impl Drop for ConstUniformBuffer {
    fn drop(&mut self) {
        self.renderer.destroy_buffer(self.buffer);
    }
}

impl ConstUniformBuffer {
    pub fn new(renderer: &Arc<Renderer>, size: u64) -> Result<Self, Error> {
        let buffer = renderer.create_buffer(
            BufferCreateDesc::gpu(size)
                .uniform_buffer()
                .transfer_destination(),
        )?;
        Ok(Self {
            renderer: renderer.clone(),
            buffer,
            allocator: BumpAllocator::new(size),
        })
    }

    pub fn push<T: Copy>(&self, data: T) -> Result<BufferSlice, Error> {
        let aligment = self
            .renderer
            .device
            .physical_device
            .properties
            .limits
            .min_uniform_buffer_offset_alignment;
        let size = mem::size_of::<T>() as u64;
        let offset = self
            .allocator
            .allocate(size, aligment)
            .ok_or(Error::TooManyUniforms)?;
        self.renderer
            .upload_buffer(BufferPointer::new(self.buffer, offset), &[data])?;
        Ok(BufferSlice::new(self.buffer, offset, size))
    }
}
