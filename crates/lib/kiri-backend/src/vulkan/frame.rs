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
    mem,
    ptr::{copy_nonoverlapping, NonNull},
};

use ash::vk::{self};
use kiri_common::BumpAllocator;

use crate::Error;

use super::{DropList, GpuAllocator, GpuDescriptorAllocator, PhysicalDevice, Uniforms};

#[derive(Debug)]
struct TempBuffer {
    offset: u32,
    allocator: BumpAllocator,
    aligment: u64,
    memory: NonNull<u8>,
}

impl TempBuffer {
    pub fn new(size: u32, offset: u32, pdevice: &PhysicalDevice, memory: NonNull<u8>) -> Self {
        Self {
            offset,
            allocator: BumpAllocator::new(size as _),
            aligment: pdevice
                .properties
                .limits
                .min_storage_buffer_offset_alignment,
            memory,
        }
    }

    pub fn push(&self, data: &[u8]) -> Result<u32, Error> {
        let offset = self
            .allocator
            .allocate(data.len(), self.aligment as _)
            .ok_or(Error::OutOfTempMemory)? as u32;
        unsafe {
            copy_nonoverlapping(
                data.as_ptr(),
                self.memory.byte_add(offset as _).as_ptr(),
                data.len(),
            )
        }
        Ok(self.offset + offset)
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
    temp: TempBuffer,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

pub(crate) const TEMP_BUFFER_SIZE: u32 = 16 * 1024 * 1024;

impl Frame {
    pub fn new(
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        queue_family_index: u32,
        temp_memory: NonNull<u8>,
        temp_memory_offset: u32,
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
                temp: TempBuffer::new(
                    TEMP_BUFFER_SIZE,
                    temp_memory_offset,
                    pdevice,
                    temp_memory.add(temp_memory_offset as _),
                ),
            })
        }
    }

    pub fn reset(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
        descriptor_allocator: &mut GpuDescriptorAllocator,
        uniforms: &mut Uniforms,
    ) -> Result<(), Error> {
        self.drop_list
            .purge(device, memory_allocator, descriptor_allocator, uniforms);
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            device.reset_fences(&[self.fence])?;
        }
        self.temp.reset();

        Ok(())
    }

    pub fn free(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
        descriptor_allocator: &mut GpuDescriptorAllocator,
        uniforms: &mut Uniforms,
    ) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.finished, None);
        }
        self.drop_list
            .purge(device, memory_allocator, descriptor_allocator, uniforms);
    }

    pub fn assign_drop_list(&mut self, drop_list: DropList) {
        self.drop_list = drop_list;
    }

    pub fn push_temp<T: Copy + Sized>(&self, data: &[T]) -> Result<u32, Error> {
        let data =
            unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, mem::size_of_val(data)) };
        self.temp.push(data)
    }
}
