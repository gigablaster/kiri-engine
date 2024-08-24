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

use core::slice;
use std::{
    cell::Cell,
    collections::HashMap,
    sync::Arc,
    thread::{self, ThreadId},
};

use ash::vk::{self};
use parking_lot::Mutex;

use crate::Error;

use super::DropList;

const DESCRIPTORS_PER_PAGE: u32 = 64;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorSetCount {
    pub sampled_images: u32,
    pub storage_buffers: u32,
    pub unifroms_buffers: u32,
    pub dynamic_storage_buffers: u32,
    pub dynamic_uniform_buffers: u32,
    pub combined_image_samplers: u32,
    pub storage_images: u32,
}

impl DescriptorSetCount {
    pub fn to_pool_size(&self, sets: u32) -> Vec<vk::DescriptorPoolSize> {
        let mut sizes = Vec::with_capacity(7);
        if self.sampled_images > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: self.sampled_images * sets,
            })
        }
        if self.storage_buffers > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: self.storage_buffers * sets,
            })
        }
        if self.unifroms_buffers > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: self.unifroms_buffers * sets,
            })
        }
        if self.dynamic_storage_buffers > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: self.dynamic_storage_buffers * sets,
            })
        }
        if self.dynamic_uniform_buffers > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: self.dynamic_uniform_buffers * sets,
            })
        }
        if self.combined_image_samplers > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: self.combined_image_samplers * sets,
            })
        }
        if self.storage_images > 0 {
            sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: self.storage_images * sets,
            })
        }
        sizes
    }
}

#[derive(Debug, Clone)]
struct DescriptorAllocator {
    pool: vk::DescriptorPool,
    free: Cell<u32>,
    total: u32,
}

unsafe impl Send for DescriptorAllocator {}
unsafe impl Sync for DescriptorAllocator {}

#[derive(Debug, Default)]
struct DescriptorAllocatorPool {
    pools: HashMap<DescriptorSetCount, Vec<Arc<DescriptorAllocator>>>,
}

#[derive(Debug, Default)]
struct CommandBufferPool {
    pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

/// Contains per-frame data
#[derive(Debug)]
pub struct Frame {
    drop_list: DropList,
    per_thread_pools: Mutex<HashMap<ThreadId, CommandBufferPool>>,
    descriptor_allocators: Mutex<DescriptorAllocatorPool>,
    pub(super) present_fence: vk::Fence,
    /// Submit this fence when submit rendering
    pub render_fence: vk::Fence,
    /// Signal this semaphore when finsihed rendering
    pub render_finished: vk::Semaphore,
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
            let present_fence = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
            let render_fence = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
            let render_finished =
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let drop_list = DropList::default();
            Ok(Self {
                present_fence,
                render_fence,
                render_finished,
                drop_list,
                per_thread_pools: Default::default(),
                descriptor_allocators: Default::default(),
            })
        }
    }

    pub(super) fn reset(&mut self, device: &ash::Device) -> Result<(), Error> {
        self.drop_list.purge(device);
        unsafe {
            device.reset_fences(&[self.present_fence, self.render_fence])?;
        }
        self.per_thread_pools
            .lock()
            .iter_mut()
            .try_for_each(|(_, x)| x.recycle(device))?;
        self.descriptor_allocators.lock().reset(device);

        Ok(())
    }

    pub(super) fn free(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_fence(self.present_fence, None);
            device.destroy_fence(self.render_fence, None);
            device.destroy_semaphore(self.render_finished, None);
        }
        self.drop_list.purge(device);
        self.per_thread_pools
            .lock()
            .drain()
            .for_each(|(_, x)| x.free(device));
        self.descriptor_allocators.lock().free(device);
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

    pub fn with_descriptor_allocator<
        E: std::error::Error,
        CB: FnOnce(&mut DescriptorAllocatorContext) -> Result<(), E>,
    >(
        &self,
        device: &ash::Device,
        cb: CB,
    ) -> Result<(), E> {
        cb(&mut DescriptorAllocatorContext {
            device,
            pool: &mut self.descriptor_allocators.lock(),
        })
    }

    pub fn get_descriptor(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        count: DescriptorSetCount,
    ) -> Result<vk::DescriptorSet, Error> {
        let mut context = DescriptorAllocatorContext {
            device,
            pool: &mut self.descriptor_allocators.lock(),
        };
        context.allocate(layout, count)
    }
}

impl DescriptorAllocator {
    pub fn new(device: &ash::Device, count: DescriptorSetCount, sets: u32) -> Result<Self, Error> {
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(sets)
                    .pool_sizes(&count.to_pool_size(sets)),
                None,
            )
        }?;
        Ok(Self {
            pool,
            free: Cell::new(sets),
            total: sets,
        })
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) }
    }

    pub fn reset(&self, device: &ash::Device) -> Result<(), Error> {
        self.free.set(self.total);
        Ok(unsafe {
            device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::default())
        }?)
    }

    pub fn is_empty(&self) -> bool {
        self.free.get() == 0
    }

    pub fn allocate(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
    ) -> Result<Option<vk::DescriptorSet>, Error> {
        if self.free.get() == 0 {
            return Ok(None);
        }
        let mut allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(slice::from_ref(&layout));
        allocate_info.descriptor_set_count = 1;
        self.free.set(self.free.get() - 1);
        Ok(Some(
            unsafe { device.allocate_descriptor_sets(&allocate_info) }?[0],
        ))
    }
}

impl DescriptorAllocatorPool {
    pub fn get_pool(
        &mut self,
        device: &ash::Device,
        count: DescriptorSetCount,
    ) -> Result<Arc<DescriptorAllocator>, Error> {
        let pool = self.pools.entry(count).or_default();
        let allocator = if let Some(allocator) = pool.iter().find(|x| !x.is_empty()) {
            allocator
        } else {
            let allocator = DescriptorAllocator::new(device, count, DESCRIPTORS_PER_PAGE)?;
            let index = pool.len();
            pool.push(allocator.into());
            &mut pool[index]
        };
        Ok(allocator.clone())
    }

    pub fn reset(&self, device: &ash::Device) {
        self.pools.iter().for_each(|(_, pool)| {
            pool.iter().for_each(|x| x.reset(device).unwrap());
        });
    }

    pub fn free(&mut self, device: &ash::Device) {
        self.pools
            .drain()
            .for_each(|(_, mut pool)| pool.drain(..).for_each(|x| x.free(device)));
    }
}

pub struct DescriptorAllocatorContext<'a> {
    device: &'a ash::Device,
    pool: &'a mut DescriptorAllocatorPool,
}

impl<'a> DescriptorAllocatorContext<'a> {
    pub fn allocate(
        &mut self,
        layout: vk::DescriptorSetLayout,
        count: DescriptorSetCount,
    ) -> Result<vk::DescriptorSet, Error> {
        loop {
            if let Some(descriptor_set) = self
                .pool
                .get_pool(self.device, count)?
                .allocate(self.device, layout)?
            {
                return Ok(descriptor_set);
            }
        }
    }
}
