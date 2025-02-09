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

use super::{drop_list::DropList, BufferHandle, BufferPointer, GpuMemoryBlock, GraphicsDevice};

pub(crate) type BufferPool = HotColdPool<vk::Buffer, Buffer>;

#[derive(Debug, Clone, Copy)]
pub struct BufferCreateDesc<'a> {
    size: usize,
    usage: vk::BufferUsageFlags,
    memory_usage: gpu_alloc::UsageFlags,
    name: Option<&'a str>,
    dedicated: bool,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn host(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::HOST_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn upload(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::HOST_ACCESS | gpu_alloc::UsageFlags::UPLOAD,
            name: None,
            dedicated: false,
        }
    }

    pub fn shared(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                | gpu_alloc::UsageFlags::HOST_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn index_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        self
    }

    pub fn veretex_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        self
    }

    pub fn storage_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        self
    }

    pub fn uniform_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        self
    }

    pub fn transfer_destination(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        self
    }

    pub fn transfer_source(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        self
    }

    pub fn indirect_draw(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        self
    }

    pub fn name(mut self, value: &'a str) -> Self {
        self.name = Some(value);
        self
    }

    pub fn dedicated(mut self) -> Self {
        self.dedicated = true;
        self
    }

    fn build(&self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo::default()
            .usage(self.usage)
            .size(self.size as _)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
}

#[derive(Debug)]
pub struct Buffer {
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    pub mapping: Option<NonNull<u8>>,
    pub memory: Option<GpuMemoryBlock>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn free(mut self, drop_list: &mut DropList) {
        if let Some(memory) = self.memory.take() {
            drop_list.drop_buffer(self.raw);
            drop_list.drop_memory(memory);
        }
    }
}

impl GraphicsDevice {
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
        let buffer = Buffer {
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

    pub fn upload_buffer<T: Copy>(&self, target: BufferPointer, data: &[T]) -> Result<(), Error> {
        let buffer = self
            .buffers
            .read()
            .get(target.handle)
            .copied()
            .ok_or(Error::InvalidBufferHandle(target.handle))?;
        self.staging
            .lock()
            .upload_buffer(&self.raw, buffer, target.offset as _, data)
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
