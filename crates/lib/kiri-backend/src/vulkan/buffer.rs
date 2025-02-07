// Copyright (C) 2023-2024 gigablaster

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

use std::ptr::NonNull;

use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use kiri_common::HotColdPool;

use crate::Error;

use super::{drop_list::DropList, BufferCreateDesc, BufferHandle, GpuMemoryBlock, RenderDevice};

pub type BufferPool = HotColdPool<vk::Buffer, BufferData>;

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
}

#[derive(Debug)]
pub struct BufferData {
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    pub mapping: Option<NonNull<u8>>,
    pub memory: Option<GpuMemoryBlock>,
}

unsafe impl Send for BufferData {}
unsafe impl Sync for BufferData {}

impl BufferData {
    pub fn free(mut self, drop_list: &mut DropList) {
        if let Some(memory) = self.memory.take() {
            drop_list.drop_buffer(self.raw);
            drop_list.drop_memory(memory);
        }
    }
}

impl RenderDevice {
    pub fn create_buffer(&self, desc: BufferCreateDesc) -> Result<BufferHandle, Error> {
        let buffer = unsafe { self.raw.create_buffer(&desc.build(), None) }?;
        let requirements = unsafe { self.raw.get_buffer_memory_requirements(buffer) };
        if let Some(name) = desc.name {
            self.set_object_name(buffer, name);
        }

        let mut memory = self.allocate(requirements, desc.memory_usage, desc.dedicated)?;
        unsafe {
            self.raw
                .bind_buffer_memory(buffer, *memory.memory(), memory.offset())
        }?;

        let mapping = if desc
            .memory_usage
            .contains(gpu_alloc::UsageFlags::HOST_ACCESS)
            | desc.memory_usage.contains(gpu_alloc::UsageFlags::UPLOAD)
            | desc.memory_usage.contains(gpu_alloc::UsageFlags::DOWNLOAD)
        {
            Some(unsafe { memory.map(AshMemoryDevice::wrap(&self.raw), 0, desc.size as _) }?)
        } else {
            None
        };
        let buffer = BufferData {
            raw: buffer,
            desc: BufferDesc {
                size: desc.size,
                usage: desc.usage,
            },
            mapping,
            memory: Some(memory),
        };
        Ok(self.buffers.write().push(buffer.raw, buffer))
    }

    pub fn upload_buffer<T: Copy>(
        &self,
        handle: BufferHandle,
        offset: usize,
        data: &[T],
    ) -> Result<(), Error> {
        let buffer = self
            .buffers
            .read()
            .get(handle)
            .copied()
            .ok_or(Error::InvalidBufferHandle(handle))?;
        self.staging
            .lock()
            .upload_buffer(&self.raw, buffer, offset, data)
    }

    pub fn get_buffer_mapping(&self, handle: BufferHandle) -> Result<NonNull<u8>, Error> {
        self.buffers
            .read()
            .get_cold(handle)
            .ok_or(Error::InvalidBufferHandle(handle))?
            .mapping
            .ok_or(Error::BufferIsntMapped(handle))
    }

    pub fn destroy_buffer(&self, handle: BufferHandle) {
        self.buffers_to_destroy.lock().push(handle);
    }
}
