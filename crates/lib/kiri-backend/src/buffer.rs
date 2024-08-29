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

use crate::{Error, GpuMemoryBlock, RenderDevice};

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub size: u64,
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
    pub mapping: Option<NonNull<u8>>,
    pub device_address: Option<vk::DeviceAddress>,
    memory: Option<GpuMemoryBlock>,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferCreateDesc<'a> {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_usage: gpu_alloc::UsageFlags,
    pub name: Option<&'a str>,
    pub dedicated: bool,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn host(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::HOST_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn upload(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::HOST_ACCESS | gpu_alloc::UsageFlags::UPLOAD,
            name: None,
            dedicated: false,
        }
    }

    pub fn shared(size: u64) -> Self {
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

    pub fn device_address(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR;
        self
    }

    pub fn usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage = usage;
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

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.with_drop_list(|drop_list| {
            if let Some(memory) = self.memory.take() {
                drop_list.drop_buffer(self.raw);
                drop_list.drop_memory(memory);
            }
        });
    }
}

impl Buffer {
    pub fn new(device: &Arc<RenderDevice>, desc: BufferCreateDesc) -> Result<Self, Error> {
        let buffer = unsafe { device.raw.create_buffer(&desc.build(), None) }?;
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        if let Some(name) = desc.name {
            device.set_object_name(buffer, name);
        }

        let mut memory = device.allocate(requirements, desc.memory_usage, desc.dedicated)?;
        unsafe {
            device
                .raw
                .bind_buffer_memory(buffer, *memory.memory(), memory.offset())
        }?;

        let mapping = if desc
            .memory_usage
            .contains(gpu_alloc::UsageFlags::HOST_ACCESS)
            | desc.memory_usage.contains(gpu_alloc::UsageFlags::UPLOAD)
            | desc.memory_usage.contains(gpu_alloc::UsageFlags::DOWNLOAD)
        {
            Some(unsafe { memory.map(AshMemoryDevice::wrap(&device.raw), 0, desc.size as _) }?)
        } else {
            None
        };
        let device_address = if desc
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR)
        {
            Some(unsafe {
                device.raw.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            })
        } else {
            None
        };
        Ok(Self {
            device: device.clone(),
            raw: buffer,
            desc: BufferDesc {
                size: desc.size,
                usage: desc.usage,
            },
            mapping,
            device_address,
            memory: Some(memory),
        })
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.device
                .raw
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(self.raw))
        }
    }
}
