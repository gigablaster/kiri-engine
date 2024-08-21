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

use std::{ptr::NonNull, sync::Arc};

use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;

use crate::{Error, RenderDevice};

use super::GpuMemory;

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
}

/// Wraps a vulkan buffer
///
/// Tracks it's own resources.
#[derive(Debug)]
pub struct Buffer {
    device: Arc<RenderDevice>,
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    memory: Option<GpuMemory>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferCreateDesc<'a> {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub alignment: Option<u64>,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    pub memory_location: gpu_alloc::UsageFlags,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn host(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn upload(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::UPLOAD,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn shared(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS
                | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: true,
            name: None,
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

    pub fn device_address(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        self
    }

    pub fn usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn aligment(mut self, aligment: u64) -> Self {
        self.alignment = Some(aligment);
        self
    }

    pub fn dedicated(mut self) -> Self {
        self.dedicated = true;
        self
    }

    pub fn name(mut self, value: &'a str) -> Self {
        self.name = Some(value);
        self
    }

    fn build(&self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo::default()
            .usage(self.usage)
            .size(self.size as _)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            self.device.with_drop_list(|drop_list| {
                drop_list.drop_buffer(self.raw);
                drop_list.drop_memory(memory);
            })
        }
    }
}

impl Buffer {
    pub fn new(device: &Arc<RenderDevice>, desc: BufferCreateDesc) -> Result<Self, Error> {
        let mut location = desc.memory_location;
        if desc
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        {
            location |= gpu_alloc::UsageFlags::DEVICE_ADDRESS;
        }
        let buffer = unsafe { device.raw.create_buffer(&desc.build(), None) }?;
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };

        let memory = device.allocate_memory(requirements, location, desc.dedicated)?;
        unsafe {
            device
                .raw
                .bind_buffer_memory(buffer, *memory.memory(), memory.offset())
        }?;
        if let Some(name) = desc.name {
            device.set_object_name(buffer, name);
        }
        Ok(Self {
            device: device.clone(),
            raw: buffer,
            desc: BufferDesc {
                size: desc.size,
                usage: desc.usage,
            },
            memory: Some(memory),
        })
    }

    pub fn map(&mut self) -> Result<NonNull<u8>, Error> {
        Ok(unsafe {
            self.memory.as_mut().unwrap().map(
                AshMemoryDevice::wrap(&self.device.raw),
                0,
                self.desc.size as _,
            )
        }?)
    }

    pub fn unmap(&mut self) {
        unsafe {
            self.memory
                .as_mut()
                .unwrap()
                .unmap(AshMemoryDevice::wrap(&self.device.raw))
        };
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.device
                .raw
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(self.raw))
        }
    }
}
