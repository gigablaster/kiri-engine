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

use std::{
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    sync::Arc,
};

use kiri_common::BumpAllocator;

use crate::Error;

use super::{BufferCreateDesc, BufferHandle, BufferSlice, GraphicsDevice};

const PAGE_SIZE: usize = 1024 * 1024;

#[derive(Debug)]
pub struct DynamicMemoryPage {
    aligment: usize,
    buffer: BufferHandle,
    mapping: NonNull<u8>,
    allocator: BumpAllocator,
}

impl DynamicMemoryPage {
    pub fn new(device: &GraphicsDevice, size: usize) -> Result<Self, Error> {
        let buffer = device.create_buffer(
            BufferCreateDesc::shared(size)
                .index_buffer()
                .veretex_buffer()
                .storage_buffer()
                .uniform_buffer(),
        )?;
        Ok(Self {
            aligment: device
                .physical_device
                .properties
                .limits
                .min_uniform_buffer_offset_alignment
                .max(
                    device
                        .physical_device
                        .properties
                        .limits
                        .min_storage_buffer_offset_alignment,
                ) as usize,
            buffer,
            mapping: device.get_buffer_mapping(buffer)?,
            allocator: BumpAllocator::new(size),
        })
    }

    pub fn try_push<T: Sized + Copy>(&self, data: &[T]) -> Option<BufferSlice> {
        let size = mem::size_of_val(data);
        if let Some(offset) = self.allocator.allocate(size, self.aligment) {
            unsafe {
                copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    self.mapping.as_ptr().add(offset as _),
                    size,
                )
            };
            Some(BufferSlice::new(self.buffer, offset, size))
        } else {
            None
        }
    }

    pub fn recycle(&self) {
        self.allocator.reset();
    }

    fn free(&self, device: &GraphicsDevice) {
        device.destroy_buffer(self.buffer);
    }
}

unsafe impl Sync for DynamicMemoryPage {}
unsafe impl Send for DynamicMemoryPage {}

#[derive(Debug, Default)]
pub struct RingBuffer {
    free: Vec<Arc<DynamicMemoryPage>>,
    all: Vec<Arc<DynamicMemoryPage>>,
}

impl RingBuffer {
    pub fn allocate(&mut self, device: &GraphicsDevice) -> Result<Arc<DynamicMemoryPage>, Error> {
        if let Some(free) = self.free.pop() {
            Ok(free)
        } else {
            let page = Arc::new(DynamicMemoryPage::new(device, PAGE_SIZE)?);
            self.all.push(page.clone());
            Ok(page)
        }
    }

    pub fn recycle(&mut self, page: Arc<DynamicMemoryPage>) {
        page.recycle();
        self.free.push(page);
    }

    pub fn free(&mut self, device: &GraphicsDevice) {
        self.free.clear();
        self.all.drain(..).for_each(|page| page.free(device));
    }
}
