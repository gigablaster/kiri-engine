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
use kiri_common::{Align, BumpAllocator};
use kiri_gfx::{BufferHandle, BufferPointer, Renderer};

use crate::Error;

/// Push data into GPU storag buffer
///
/// Data can't be deallocated and must be addressed by index in shader.
#[derive(Debug)]
pub struct ConstStorageBuffer<T: Copy> {
    renderer: Arc<Renderer>,
    pub buffer: BufferHandle,
    allocator: BumpAllocator,
    element_size: u64,
    _phantom: PhantomData<T>,
}

impl<T: Copy> Drop for ConstStorageBuffer<T> {
    fn drop(&mut self) {
        self.renderer.destroy_buffer(self.buffer);
    }
}

impl<T: Copy> ConstStorageBuffer<T> {
    pub fn new(
        renderer: &Arc<Renderer>,
        allocator: &GpuAllocator,
        count: u64,
    ) -> Result<Self, Error> {
        let element_size = mem::size_of::<T>().align(
            renderer
                .device
                .physical_device
                .properties
                .limits
                .min_storage_buffer_offset_alignment
                .max(16) as _,
        ) as u64;
        let buffer = renderer.create_buffer(
            BufferCreateDesc::gpu(element_size * count)
                .storage_buffer()
                .transfer_destination()
                .allocator(allocator)
                .name(&format!("{:?}", type_name::<T>())),
        )?;
        Ok(Self {
            renderer: renderer.clone(),
            buffer,
            element_size,
            allocator: BumpAllocator::new(element_size * count),
            _phantom: PhantomData,
        })
    }

    pub fn push_and_get_index(&self, data: T) -> Result<u32, Error> {
        let offset = self
            .allocator
            .allocate(self.element_size, self.element_size)
            .ok_or(Error::NotEnoughGpuConstMemory)?;
        self.renderer
            .upload_buffer(BufferPointer::new(self.buffer, offset), &[data])?;

        Ok((offset / self.element_size) as u32)
    }

    pub fn get_buffer(&self) -> BufferPointer {
        BufferPointer::new(self.buffer, 0)
    }
}
