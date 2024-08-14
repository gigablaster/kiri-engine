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

use crate::{BufferHandle, BufferUsage, Error, RenderDevice};

use super::{DropList, GpuMemory};

impl From<BufferUsage> for vk::BufferUsageFlags {
    fn from(value: BufferUsage) -> Self {
        let mut result = vk::BufferUsageFlags::empty();
        if value.contains(BufferUsage::Vertex) {
            result |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if value.contains(BufferUsage::Index) {
            result |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if value.contains(BufferUsage::Storage) {
            result |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if value.contains(BufferUsage::Uniform) {
            result |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if value.contains(BufferUsage::Destination) {
            result |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if value.contains(BufferUsage::Source) {
            result |= vk::BufferUsageFlags::TRANSFER_SRC;
        }

        result
    }
}

#[derive(Debug)]
pub(crate) struct Buffer {
    pub raw: vk::Buffer,
    pub size: u32,
    pub memory: Option<GpuMemory>,
}

impl Buffer {
    pub(crate) fn free(mut self, drop_list: &mut DropList) {
        if let Some(memory) = self.memory.take() {
            drop_list.drop_buffer(self.raw);
            drop_list.drop_memory(memory);
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferCreateDesc<'a> {
    pub size: u32,
    pub usage: BufferUsage,
    pub alignment: Option<u64>,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    memory_location: gpu_alloc::UsageFlags,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: u32) -> Self {
        Self {
            size,
            usage: BufferUsage::empty(),
            memory_location: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn host(size: u32) -> Self {
        Self {
            size,
            usage: BufferUsage::empty(),
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn upload(size: u32) -> Self {
        Self {
            size,
            usage: BufferUsage::empty(),
            memory_location: gpu_alloc::UsageFlags::UPLOAD,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn shared(size: u32) -> Self {
        Self {
            size,
            usage: BufferUsage::empty(),
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS
                | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: true,
            name: None,
        }
    }

    pub fn usage(mut self, usage: BufferUsage) -> Self {
        self.usage = usage;
        self
    }

    pub fn aligment(mut self, aligment: u64) -> Self {
        self.alignment = Some(aligment);
        self
    }

    pub fn dedicated(mut self, value: bool) -> Self {
        self.dedicated = value;
        self
    }

    pub fn name(mut self, value: &'a str) -> Self {
        self.name = Some(value);
        self
    }

    fn build(&self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo::default()
            .usage(self.usage.into())
            .size(self.size as _)
    }
}

impl<'game> RenderDevice<'game> {
    pub fn create_buffer(
        &self,
        desc: BufferCreateDesc,
        data: Option<&[u8]>,
    ) -> Result<BufferHandle, Error> {
        let buffer = unsafe { self.device.create_buffer(&desc.build(), None) }?;
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory = self.allocate(requirements, desc.memory_location, desc.dedicated)?;
        unsafe {
            self.device
                .bind_buffer_memory(buffer, *memory.memory(), memory.offset())
        }?;
        if let Some(name) = desc.name {
            self.set_object_name(buffer, name);
        }
        if let Some(data) = data {
            self.staging.lock().upload_buffer(self, buffer, 0, data)?;
        }
        let handle = self.buffers.write().push(
            buffer,
            Buffer {
                raw: buffer,
                size: desc.size,
                memory: Some(memory),
            },
        );
        Ok(handle)
    }

    pub fn update_buffer(
        &self,
        handle: BufferHandle,
        offset: u32,
        data: &[u8],
    ) -> Result<(), Error> {
        let buffer = self
            .buffers
            .read()
            .get_cold(handle)
            .ok_or(Error::InvalidBufferHandle(handle))?
            .raw;
        self.staging
            .lock()
            .upload_buffer(self, buffer, offset, data)?;
        Ok(())
    }

    pub fn destroy_buffer(&self, handle: BufferHandle) {
        // Can't manually delete temporarty buffer
        debug_assert!(handle != self.temp_buffer_handle);
        if let Some((_, buffer)) = self.buffers.write().remove(handle) {
            self.with_drop_list(|drop_list| {
                buffer.free(drop_list);
            })
        }
    }

    pub fn map_buffer(&self, handle: BufferHandle) -> Result<NonNull<u8>, Error> {
        let mut buffers = self.buffers.write();
        let buffer = buffers
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBufferHandle(handle))?;
        if let Some(memory) = &mut buffer.memory {
            Ok(unsafe { memory.map(AshMemoryDevice::wrap(&self.device), 0, buffer.size as _) }?)
        } else {
            Err(Error::MemoryNotAllocated)
        }
    }

    pub fn unmap_buffer(&self, handle: BufferHandle) {
        let mut buffers = self.buffers.write();
        if let Some(buffer) = buffers.get_cold_mut(handle) {
            if let Some(memory) = &mut buffer.memory {
                unsafe {
                    memory.unmap(AshMemoryDevice::wrap(&self.device));
                }
            }
        }
    }
}
