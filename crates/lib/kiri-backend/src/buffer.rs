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

use crate::{Error, GpuAllocator, GpuMemoryPage, RenderDevice};

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
    dedicated: Option<GpuMemoryPage>,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferCreateDesc<'a> {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: vk::MemoryPropertyFlags,
    pub name: Option<&'a str>,
    pub allocator: Option<&'a GpuAllocator>,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            name: None,
            allocator: None,
        }
    }

    pub fn host(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: vk::MemoryPropertyFlags::HOST_VISIBLE,
            name: None,
            allocator: None,
        }
    }

    pub fn shared(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            name: None,
            allocator: None,
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

    pub fn allocator(mut self, value: &'a GpuAllocator) -> Self {
        self.allocator = Some(value);
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
        self.device.with_drop_list(|drop_list| {
            drop_list.drop_buffer(self.raw);
        });
        if let Some(memory) = self.dedicated.take() {
            memory.free(&self.device.raw);
        }
    }
}

impl Buffer {
    pub fn new(device: &Arc<RenderDevice>, desc: BufferCreateDesc) -> Result<Self, Error> {
        let buffer = unsafe { device.raw.create_buffer(&desc.build(), None) }?;
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        if let Some(name) = desc.name {
            device.set_object_name(buffer, name);
        }

        let (memory, offset, page) = device.use_allocator_or_dedicated(
            desc.allocator,
            requirements,
            desc.memory_location,
        )?;
        unsafe { device.raw.bind_buffer_memory(buffer, memory, offset) }?;

        let mapping = if desc
            .memory_location
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && page.is_some()
        {
            NonNull::new(unsafe {
                device
                    .raw
                    .map_memory(memory, 0, desc.size as _, vk::MemoryMapFlags::empty())
            }? as *mut u8)
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
            dedicated: page,
            mapping,
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
