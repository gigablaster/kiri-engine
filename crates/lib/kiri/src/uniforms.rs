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

use std::{any::type_name, marker::PhantomData, mem, sync::Arc};

use kiri_backend::{BufferCreateDesc, GpuAllocator};
use kiri_common::BlockAllocator;
use kiri_gfx::{BufferHandle, BufferPointer, BufferSlice, Renderer};
use parking_lot::Mutex;

use crate::Error;

/// Static uniform allocator
///
/// Allocate uniforms of single type.
#[derive(Debug)]
pub struct ConstUniforms<T: Copy> {
    renderer: Arc<Renderer>,
    pub buffer: BufferHandle,
    allocator: Mutex<BlockAllocator>,
    _phantom: PhantomData<T>,
}

impl<T: Copy> Drop for ConstUniforms<T> {
    fn drop(&mut self) {
        self.renderer.destroy_buffer(self.buffer);
    }
}

impl<T: Copy> ConstUniforms<T> {
    pub fn new(
        renderer: &Arc<Renderer>,
        allocator: &GpuAllocator,
        count: u64,
    ) -> Result<Self, Error> {
        let block_size = mem::size_of::<T>().max(
            renderer
                .device
                .physical_device
                .properties
                .limits
                .min_uniform_buffer_offset_alignment as _,
        ) as u64;
        let buffer = renderer.create_buffer(
            BufferCreateDesc::gpu(block_size * count)
                .uniform_buffer()
                .transfer_destination()
                .allocator(allocator)
                .name(&format!("{:?} uniforms", type_name::<T>())),
        )?;
        Ok(Self {
            renderer: renderer.clone(),
            buffer,
            allocator: Mutex::new(BlockAllocator::new(block_size as _, count as _)),
            _phantom: PhantomData,
        })
    }

    pub fn push(&self, data: T) -> Result<BufferSlice, Error> {
        let offset = self
            .allocator
            .lock()
            .allocate()
            .ok_or(Error::TooManyUniforms)?;
        self.renderer
            .upload_buffer(BufferPointer::new(self.buffer, offset), &[data])?;
        Ok(BufferSlice::new(
            self.buffer,
            offset,
            mem::size_of::<T>() as _,
        ))
    }

    pub fn free(&self, buffer: BufferSlice) {
        self.allocator.lock().dealloc(buffer.offset);
    }
}
