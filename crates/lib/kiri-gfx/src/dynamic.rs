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

use std::{
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    sync::Arc,
};

use kiri_backend::{BufferCreateDesc, PhysicalDevice};
use kiri_common::BumpAllocator;

use crate::{BufferHandle, Error, Renderer};

#[derive(Debug)]
pub struct DynamicGpuMemory {
    buffer: BufferHandle,
    mapping: NonNull<u8>,
    allocator: BumpAllocator,
}

unsafe impl Send for DynamicGpuMemory {}
unsafe impl Sync for DynamicGpuMemory {}

impl DynamicGpuMemory {
    pub fn new(renderer: &Renderer, size: usize) -> Result<Self, Error> {
        let buffer = renderer.create_buffer(
            BufferCreateDesc::shared(size)
                .name("Dynamic data")
                .storage_buffer()
                .indirect_draw()
                .uniform_buffer()
                .veretex_buffer()
                .index_buffer(),
        )?;
        let mapping = renderer.get_buffer_mapping(buffer)?.unwrap();
        Ok(Self {
            buffer,
            mapping,
            allocator: BumpAllocator::new(size),
        })
    }

    pub fn push<T: Copy>(&self, pdevice: &PhysicalDevice, data: &[T]) -> Result<u32, Error> {
        let size = mem::size_of_val(data);
        if let Some(offset) = self.allocator.allocate(
            size,
            pdevice
                .properties
                .limits
                .min_uniform_buffer_offset_alignment as _,
        ) {
            unsafe {
                copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    self.mapping.as_ptr().add(offset),
                    size,
                )
            }
            Ok(offset as u32)
        } else {
            Err(Error::OutOfDynamicMemory)
        }
    }

    pub fn get_buffer_handle(&self) -> BufferHandle {
        self.buffer
    }

    pub fn recycle(&self) {
        self.allocator.reset();
    }
}

#[derive(Debug, Default)]
pub struct DynamicGpuMemoryPool {
    pool: Vec<Arc<DynamicGpuMemory>>,
    used: Vec<Arc<DynamicGpuMemory>>,
    recycle: Vec<Arc<DynamicGpuMemory>>,
}

const DYNAMIC_PAGE_SIZE: usize = 16 * 1024 * 1024;

impl DynamicGpuMemoryPool {
    pub fn get(&mut self, renderer: &Renderer) -> Result<Arc<DynamicGpuMemory>, Error> {
        if let Some(page) = self.pool.pop() {
            self.used.push(page.clone());
            Ok(page)
        } else {
            let page = Arc::new(DynamicGpuMemory::new(renderer, DYNAMIC_PAGE_SIZE)?);
            self.used.push(page.clone());
            Ok(page)
        }
    }

    pub fn recycle(&mut self) {
        self.pool.append(&mut self.recycle);
        self.recycle.append(&mut self.used);
        self.pool.iter().for_each(|x| x.recycle());
    }
}
