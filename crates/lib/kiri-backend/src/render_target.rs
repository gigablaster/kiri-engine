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

use ash::vk;
use kiri_common::BumpAllocator;

use crate::{Error, PhysicalDevice};

/// Special alllocator for render targets
///
/// Allocate big block of memory and then bump-allocate it when we need new render target.
/// It's on client to take care of images that point there. We're supposed to recreate all
/// of our render targets when swapchain changed.
///
/// Swapchain creatin resets allocator.
#[derive(Debug)]
pub struct RenderTargetAllocator {
    pub memory: vk::DeviceMemory,
    allocator: BumpAllocator,
}

const ALIGMENT: u64 = 65536;

impl RenderTargetAllocator {
    pub fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        pdevice: &PhysicalDevice,
        size: u64,
    ) -> Result<Self, Error> {
        let index = unsafe { instance.get_physical_device_memory_properties(pdevice.raw) }
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .find_map(|(index, data)| {
                if data
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    Some(index)
                } else {
                    None
                }
            })
            .ok_or(Error::NoRenderTargetMemory)?;
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(index as _),
                None,
            )
        }?;
        Ok(Self {
            memory,
            allocator: BumpAllocator::new(size as _, ALIGMENT as _), // 4096 because reasons, should work
        })
    }

    pub fn allocate(&self, requirements: vk::MemoryRequirements) -> Result<u64, Error> {
        assert!(requirements.alignment <= ALIGMENT);
        Ok(self
            .allocator
            .allocate(requirements.size as _)
            .ok_or(Error::OutOfDeviceMemory)? as u64)
    }

    pub fn reset(&self) {
        self.allocator.reset();
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.free_memory(self.memory, None) };
    }
}
