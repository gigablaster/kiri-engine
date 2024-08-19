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

use ash::vk::{self};
use kiri_backend::{Buffer, BufferCreateDesc, RenderDevice};
use kiri_common::BumpAllocator;

use crate::Error;

#[derive(Debug)]
pub struct DynamicGpuMemory {
    _device: Arc<RenderDevice>,
    _buffer: Buffer,
    address: vk::DeviceAddress,
    mapping: NonNull<u8>,
    allocator: BumpAllocator,
}

impl DynamicGpuMemory {
    pub fn new(device: &Arc<RenderDevice>, size: usize) -> Result<Self, Error> {
        let mut buffer = Buffer::new(
            device,
            BufferCreateDesc::shared(size)
                .name("Dynamic")
                .storage_buffer()
                .indirect_draw()
                .dedicated(true),
        )?;
        let mapping = buffer.map()?;
        let address = buffer.device_address();
        Ok(Self {
            _device: device.clone(),
            _buffer: buffer,
            address,
            mapping,
            allocator: BumpAllocator::new(
                size,
                device
                    .physical_device
                    .properties
                    .limits
                    .min_uniform_buffer_offset_alignment as _,
            ),
        })
    }

    pub fn push<T: Copy>(&self, data: &[T]) -> Result<vk::DeviceAddress, Error> {
        let size = mem::size_of_val(data);
        if let Some(offset) = self.allocator.allocate(size) {
            unsafe {
                copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    self.mapping.as_ptr().add(offset),
                    size,
                )
            }
            Ok(self.address + size as u64)
        } else {
            Err(Error::OutOfDynamicMemory)
        }
    }

    pub fn recycle(&self) {
        self.allocator.reset();
    }
}

#[derive(Debug)]
pub struct DynamicGpuMemoryPool {
    device: Arc<RenderDevice>,
    pool: Vec<Arc<DynamicGpuMemory>>,
    used: Vec<Arc<DynamicGpuMemory>>,
    recycle: Vec<Arc<DynamicGpuMemory>>,
}

const DYNAMIC_PAGE_SIZE: usize = 16 * 1024 * 1024;

impl DynamicGpuMemoryPool {
    pub fn new(device: &Arc<RenderDevice>) -> Result<Self, Error> {
        Ok(Self {
            device: device.clone(),
            pool: Default::default(),
            used: Default::default(),
            recycle: Default::default(),
        })
    }

    pub fn get(&mut self) -> Result<Arc<DynamicGpuMemory>, Error> {
        if let Some(page) = self.pool.pop() {
            self.used.push(page.clone());
            Ok(page)
        } else {
            let page = Arc::new(DynamicGpuMemory::new(&self.device, DYNAMIC_PAGE_SIZE)?);
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
