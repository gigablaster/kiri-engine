// Copyright (C) 2023-2025 gigablaster

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
    collections::HashMap,
    thread::{self, ThreadId},
};

use ash::vk::{self};
use parking_lot::Mutex;

use crate::Error;

use super::{GpuAllocator, GpuDescriptorAllocator};

use super::DropList;

#[derive(Debug, Default)]
struct CommandBufferPool {
    pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

/// Contains per-frame data
#[derive(Debug)]
pub(crate) struct Frame {
    drop_list: DropList,
    per_thread_pools: Mutex<HashMap<ThreadId, CommandBufferPool>>,
    /// Submit this fence when submit rendering
    pub render_fence: vk::Fence,
    /// Signal this semaphore when finsihed rendering
    pub render_finished: vk::Semaphore,
    pub upload_semaphore: vk::Semaphore,
}

impl CommandBufferPool {
    pub fn new(device: &ash::Device) -> Result<Self, Error> {
        let create_info =
            vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.create_command_pool(&create_info, None) }?;
        Ok(Self {
            pool,
            command_buffers: Vec::default(),
        })
    }

    pub fn get(
        &mut self,
        device: &ash::Device,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, Error> {
        let create_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .command_buffer_count(1)
            .level(level);
        let cb = unsafe { device.allocate_command_buffers(&create_info)? }.remove(0);
        self.command_buffers.push(cb);
        Ok(cb)
    }

    pub fn recycle(&mut self, device: &ash::Device) -> Result<(), Error> {
        unsafe {
            device.free_command_buffers(self.pool, &self.command_buffers);
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?
        };
        self.command_buffers.clear();
        Ok(())
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_command_pool(self.pool, None) }
    }
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Frame {
    pub(super) fn new(device: &ash::Device) -> Result<Self, Error> {
        unsafe {
            let render_fence = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
            let render_finished =
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let drop_list = DropList::default();
            Ok(Self {
                render_fence,
                render_finished,
                drop_list,
                per_thread_pools: Default::default(),
                upload_semaphore: vk::Semaphore::null(),
            })
        }
    }

    pub(super) fn reset(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
        descriptor_allocator: &mut GpuDescriptorAllocator,
    ) -> Result<(), Error> {
        self.drop_list
            .purge(device, memory_allocator, descriptor_allocator);
        unsafe {
            device.reset_fences(&[self.render_fence])?;
        }
        self.per_thread_pools
            .lock()
            .iter_mut()
            .try_for_each(|(_, x)| x.recycle(device))?;

        Ok(())
    }

    pub(super) fn free(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
        descriptor_allocator: &mut GpuDescriptorAllocator,
    ) {
        unsafe {
            device.destroy_fence(self.render_fence, None);
            device.destroy_semaphore(self.render_finished, None);
        }
        self.drop_list
            .purge(device, memory_allocator, descriptor_allocator);
        self.per_thread_pools
            .lock()
            .drain()
            .for_each(|(_, x)| x.free(device));
    }

    pub(super) fn assign_drop_list(&mut self, drop_list: DropList) {
        self.drop_list = drop_list;
    }

    pub fn get_command_buffer(
        &self,
        device: &ash::Device,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, Error> {
        let therad_id = thread::current().id();
        let mut pools = self.per_thread_pools.lock();
        if let Some(pool) = pools.get_mut(&therad_id) {
            Ok(pool.get(device, level)?)
        } else {
            let mut pool = CommandBufferPool::new(device)?;
            let cb = pool.get(device, level)?;
            pools.insert(therad_id, pool);
            Ok(cb)
        }
    }
}
