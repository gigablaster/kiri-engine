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

use std::sync::Arc;

use ash::vk::{self, MemoryRequirements};
use kiri_common::BumpAllocator;
use log::debug;
use parking_lot::Mutex;

use crate::{Error, RenderDevice};

/// Linear gpu memory allocator
///
/// All of our GPU resources are allocated in big logical groups and deallocated at same time.
/// There's no way or reason to unload just one resource. So we're using chain of big memory blocks
/// with linear allocators. Each resource group is using one of those chains. Blocks are never really
/// deallocated, just returned to global pool.
#[derive(Debug)]
pub struct GpuAllocator {
    device: Arc<RenderDevice>,
    pages: Mutex<Vec<GpuMemoryPage>>,
}

impl GpuAllocator {
    pub fn new(device: &Arc<RenderDevice>) -> Self {
        Self {
            device: device.clone(),
            pages: Default::default(),
        }
    }

    pub(super) fn allocate(
        &self,
        requirements: MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize), Error> {
        let index = self
            .device
            .get_suitable_memory_index(requirements.memory_type_bits, flags)
            .ok_or(Error::NoSuitableMemoryType)?;
        let mut pages = self.pages.lock();
        if let Some(mem) = pages.iter().filter(|x| x.index == index).find_map(|page| {
            page.allocate(requirements)
                .map(|offset| (page.memory, offset))
        }) {
            Ok(mem)
        } else {
            let page = self.device.get_memory_page(index)?;
            let mem = (
                page.memory,
                page.allocate(requirements)
                    .ok_or(Error::OutOfDeviceMemory)?,
            );
            pages.push(page);
            Ok(mem)
        }
    }

    /// Returns all pages back to main pool
    pub fn recycle(&self) {
        self.pages
            .lock()
            .drain(..)
            .for_each(|page| self.device.release_memory_page(page));
    }
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        self.recycle();
    }
}

#[derive(Debug)]
pub(super) struct GpuMemoryPage {
    pub index: u32,
    pub memory: vk::DeviceMemory,
    allocator: BumpAllocator,
}

impl GpuMemoryPage {
    pub fn new(device: &ash::Device, index: u32, size: u64) -> Result<Self, Error> {
        debug!("Allocate memory page of type {} size {}", index, size);
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(index),
                None,
            )
        }?;
        Ok(Self {
            index,
            memory,
            allocator: BumpAllocator::new(size as _),
        })
    }

    /// Allocates some memory from this page
    ///
    /// Assume that heap is correct, should be checked by client.
    fn allocate(&self, requirements: vk::MemoryRequirements) -> Option<u64> {
        self.allocator
            .allocate(requirements.size as _, requirements.alignment as _)
    }

    /// Reset memory allocator
    ///
    /// So we can allocate memory again. Called when we recycle this page.
    pub fn reset(&self) {
        self.allocator.reset();
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.free_memory(self.memory, None) }
    }
}
