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
    collections::HashMap,
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    thread,
};

use ash::vk::{self};
use gpu_alloc::UsageFlags;
use gpu_alloc_ash::AshMemoryDevice;
use kiri_common::BumpAllocator;
use parking_lot::Mutex;

use crate::Error;

use super::{DropList, GpuAllocator, GpuMemory, PhysicalDevice, RenderContext};

const PREALLOCATED_COMMAND_BUFFERS: usize = 8;

#[derive(Debug)]
struct SecondaryCommandBufferPool {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    free: Vec<vk::CommandBuffer>,
}

pub(crate) struct SecondaryCommandBuffer<'a> {
    device: &'a ash::Device,
    cb: vk::CommandBuffer,
}

impl SecondaryCommandBufferPool {
    pub fn new(device: &ash::Device, queue_family_index: u32) -> Result<Self, Error> {
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
        }?;

        Ok(Self {
            pool,
            free: Vec::new(),
            buffers: Vec::new(),
        })
    }

    pub fn get_or_create(&mut self, device: &ash::Device) -> Result<vk::CommandBuffer, Error> {
        if let Some(cb) = self.free.pop() {
            Ok(cb)
        } else {
            let mut buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .command_buffer_count(PREALLOCATED_COMMAND_BUFFERS as _)
                        .level(vk::CommandBufferLevel::SECONDARY),
                )
            }?;
            self.buffers.append(&mut buffers.clone());
            self.free.append(&mut buffers);
            let cb = self.free.pop().unwrap();

            Ok(cb)
        }
    }

    pub fn reset(&mut self, device: &ash::Device) {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }
            .unwrap();
        self.free.clear();
        for it in &self.buffers {
            self.free.push(*it);
        }
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_command_pool(self.pool, None) };
    }
}

#[derive(Debug)]
struct TempBuffer {
    buffer: vk::Buffer,
    memory: Option<GpuMemory>,
    allocator: BumpAllocator,
    address: vk::DeviceAddress,
    aligment: u64,
    map: NonNull<u8>,
}

impl TempBuffer {
    pub fn new(
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        size: u64,
        allocator: &mut GpuAllocator,
    ) -> Result<Self, Error> {
        let create_info = vk::BufferCreateInfo::default().size(size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let buffer = unsafe { device.create_buffer(&create_info, None) }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut memory = RenderContext::allocate_impl(
            device,
            allocator,
            requirements,
            UsageFlags::FAST_DEVICE_ACCESS | UsageFlags::DEVICE_ADDRESS | UsageFlags::HOST_ACCESS,
            true,
        )?;
        unsafe { device.bind_buffer_memory(buffer, *memory.memory(), memory.offset()) }?;
        let map = unsafe { memory.map(AshMemoryDevice::wrap(device), 0, size as _) }?;
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok(Self {
            buffer,
            memory: Some(memory),
            allocator: BumpAllocator::new(size as _),
            aligment: pdevice
                .properties
                .limits
                .min_storage_buffer_offset_alignment,
            address,
            map,
        })
    }

    pub fn push(&self, data: &[u8]) -> Result<vk::DeviceAddress, Error> {
        let offset = self
            .allocator
            .allocate(data.len(), self.aligment as _)
            .ok_or(Error::OutOfTempMemory)? as u64;
        unsafe {
            copy_nonoverlapping(
                data.as_ptr(),
                self.map.byte_add(offset as _).as_ptr(),
                data.len(),
            )
        }
        Ok(offset + self.address)
    }

    pub fn free(&mut self, device: &ash::Device, allocator: &mut GpuAllocator) {
        if let Some(memory) = self.memory.take() {
            unsafe {
                allocator.dealloc(AshMemoryDevice::wrap(device), memory);
                device.destroy_buffer(self.buffer, None)
            }
        }
    }

    pub fn reset(&self) {
        self.allocator.reset();
    }
}

#[derive(Debug)]
pub(crate) struct Frame {
    pool: vk::CommandPool,
    pub cb: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub finished: vk::Semaphore,
    drop_list: DropList,
    per_thread_buffers: Mutex<HashMap<thread::ThreadId, SecondaryCommandBufferPool>>,
    temp: TempBuffer,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

const TEMP_BUFFER_SIZE: u64 = 16 * 1024 * 1024;

impl Frame {
    pub fn new(
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        allocator: &mut GpuAllocator,
        queue_family_index: u32,
    ) -> Result<Self, Error> {
        unsafe {
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?;
            let cb = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0];
            let fence = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
            let finished = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let drop_list = DropList::default();
            Ok(Self {
                pool,
                cb,
                fence,
                finished,
                drop_list,
                per_thread_buffers: Mutex::default(),
                temp: TempBuffer::new(device, pdevice, TEMP_BUFFER_SIZE, allocator)?,
            })
        }
    }

    pub fn reset(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
    ) -> Result<(), Error> {
        self.drop_list.purge(device, memory_allocator);
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;
        self.per_thread_buffers
            .lock()
            .iter_mut()
            .for_each(|(_, x)| x.reset(device));
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            device.reset_fences(&[self.fence])?;
        }
        self.temp.reset();

        Ok(())
    }

    pub fn free(&mut self, device: &ash::Device, memory_allocator: &mut GpuAllocator) {
        self.temp.free(device, memory_allocator);
        unsafe {
            device.destroy_command_pool(self.pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.finished, None);
        }
        self.drop_list.purge(device, memory_allocator);
        self.per_thread_buffers
            .lock()
            .drain()
            .for_each(|(_, x)| x.free(device));
    }

    pub fn assign_drop_list(&mut self, drop_list: DropList) {
        self.drop_list = drop_list;
    }

    pub fn secondary_buffer(
        &self,
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<vk::CommandBuffer, Error> {
        let thread_id = thread::current().id();
        let mut pools = self.per_thread_buffers.lock();
        if let Some(pool) = pools.get_mut(&thread_id) {
            pool.get_or_create(device)
        } else {
            let mut pool = SecondaryCommandBufferPool::new(device, queue_family_index).unwrap();
            let cb = pool.get_or_create(device);
            pools.insert(thread_id, pool);

            cb
        }
    }

    pub fn push_temp<T: Copy + Sized>(&self, data: &[T]) -> Result<vk::DeviceAddress, Error> {
        let data =
            unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, mem::size_of_val(data)) };
        self.temp.push(data)
    }
}
