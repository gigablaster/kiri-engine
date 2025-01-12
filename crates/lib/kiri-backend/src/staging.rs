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
    mem::{self},
    ptr::{copy_nonoverlapping, NonNull},
    sync::Arc,
};

use ash::vk;
use gpu_alloc::{Request, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;
use kiri_common::BumpAllocator;

use crate::{
    Error, GpuAllocator, GpuMemoryBlock, ImageDesc, ImageUploadData, PhysicalDevice, Queue,
};

#[derive(Debug, Clone, Copy)]
struct ImageUploadRequest(vk::BufferImageCopy, vk::ImageSubresourceRange);

#[derive(Debug)]
pub struct Staging {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    transfer_queue: Arc<Queue>,
    allocator: BumpAllocator,
    upload_buffers: HashMap<vk::Buffer, Vec<vk::BufferCopy>>,
    upload_images: HashMap<vk::Image, Vec<ImageUploadRequest>>,
    staging: vk::Buffer,
    memory: Option<GpuMemoryBlock>,
    mapping: NonNull<u8>,
    semaphore: vk::Semaphore,
    aligment: u64,
}

unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

const STAGING_SIZE: u64 = 128 * 1024 * 1024;

impl Staging {
    pub fn new(
        device: &ash::Device,
        physical_device: &PhysicalDevice,
        allocator: &mut GpuAllocator,
        transfer_queue: Arc<Queue>,
    ) -> Result<Self, Error> {
        let pool_info =
            vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }?;

        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(command_pool),
            )
        }?[0];
        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }?;
        let semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        let staging = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(STAGING_SIZE)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }?;
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(staging) };
        let mut memory = unsafe {
            allocator.alloc(
                AshMemoryDevice::wrap(device),
                Request {
                    size: memory_requirements.size,
                    align_mask: memory_requirements.alignment,
                    usage: UsageFlags::UPLOAD,
                    memory_types: memory_requirements.memory_type_bits,
                },
            )
        }?;
        unsafe { device.bind_buffer_memory(staging, *memory.memory(), memory.offset()) }?;
        let mapping = unsafe { memory.map(AshMemoryDevice::wrap(device), 0, STAGING_SIZE as _) }?;

        Ok(Self {
            fence,
            command_pool,
            command_buffer,
            allocator: BumpAllocator::new(STAGING_SIZE as _),
            upload_buffers: Default::default(),
            upload_images: Default::default(),
            mapping,
            transfer_queue,
            memory: Some(memory),
            staging,
            semaphore,
            aligment: physical_device
                .properties
                .limits
                .buffer_image_granularity
                .max(64),
        })
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        device: &ash::Device,
        target: vk::Buffer,
        offset: u64,
        data: &[T],
    ) -> Result<(), Error> {
        let mut current_offset = 0;
        loop {
            let data_len = mem::size_of_val(data) as u64;
            let pushed = self.try_push_buffer(
                target,
                offset + current_offset,
                data_len - current_offset,
                unsafe { (data.as_ptr() as *const u8).add(current_offset as _) },
            )?;
            current_offset += pushed;
            if current_offset == data_len {
                return Ok(());
            } else {
                self.upload_impl(device, false)?;
            }
        }
    }

    pub fn upload_image(
        &mut self,
        device: &ash::Device,
        target: vk::Image,
        desc: ImageDesc,
        data: &[ImageUploadData],
    ) -> Result<(), Error> {
        // If we have operations for same target then we just cancel them
        self.upload_images.remove(&target);
        for (mip, data) in data.iter().enumerate() {
            while !self.try_push_mip(target, desc, mip as _, data)? {
                self.upload_impl(device, false)?;
            }
        }
        Ok(())
    }

    fn try_push_mip(
        &mut self,
        target: vk::Image,
        desc: ImageDesc,
        mip: u32,
        data: &ImageUploadData,
    ) -> Result<bool, Error> {
        let size = data.data.len() as u64;
        if size > STAGING_SIZE {
            return Err(Error::ImageTooBig);
        }
        if let Some(offset) = self.allocator.allocate(size, self.aligment) {
            unsafe {
                copy_nonoverlapping(
                    data.data.as_ptr(),
                    self.mapping.as_ptr().add(offset as _),
                    size as _,
                )
            };
            let dims = desc.dims;
            let op = vk::BufferImageCopy::default()
                .image_extent(vk::Extent3D {
                    width: dims[0] >> mip,
                    height: dims[1] >> mip,
                    depth: 1,
                })
                .buffer_offset(offset as _)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            self.upload_images
                .entry(target)
                .or_default()
                .push(ImageUploadRequest(op, range));

            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn try_push_buffer(
        &mut self,
        target: vk::Buffer,
        offset: u64,
        bytes: u64,
        data: *const u8,
    ) -> Result<u64, Error> {
        let can_send = self.allocator.validate(bytes, self.aligment);
        let dst_offset = self
            .allocator
            .allocate(can_send as _, self.aligment)
            .unwrap(); // Already checked that allocator can allocate enough space
        unsafe {
            copy_nonoverlapping(
                data,
                self.mapping.as_ptr().add(dst_offset as _),
                can_send as _,
            )
        };
        let op = vk::BufferCopy::default()
            .src_offset(dst_offset as _)
            .dst_offset(offset)
            .size(can_send);
        self.upload_buffers.entry(target).or_default().push(op);

        Ok(can_send)
    }

    pub fn upload(&mut self, device: &ash::Device) -> Result<vk::Semaphore, Error> {
        let semaphore = self.upload_impl(device, true)?;
        Ok(semaphore.unwrap())
    }

    fn upload_impl(
        &mut self,
        device: &ash::Device,
        need_semaphore: bool,
    ) -> Result<Option<vk::Semaphore>, Error> {
        unsafe {
            device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            device.reset_fences(&[self.fence])?;
            device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.barrier_before(device);
            self.copy_buffers(device);
            self.copy_images(device);
            self.barriers_after(device);
            device.end_command_buffer(self.command_buffer)?;
        }
        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();
        if need_semaphore {
            self.transfer_queue.submit(
                device,
                &[self.command_buffer],
                self.fence,
                &[],
                &[self.semaphore],
            )?;
            Ok(Some(self.semaphore))
        } else {
            self.transfer_queue
                .submit(device, &[self.command_buffer], self.fence, &[], &[])?;
            Ok(None)
        }
    }

    fn barriers_after(&mut self, device: &ash::Device) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier::default()
                    .buffer(*x.0)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::INDEX_READ)
                    .offset(op.dst_offset)
                    .size(op.size);
                buffer_barriers.push(barrier);
            })
        });
        let size = self.upload_images.iter().map(|x| x.1.len()).sum::<usize>();
        let mut image_barriers = Vec::with_capacity(size);
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(*x.0)
                    .subresource_range(op.1);
                image_barriers.push(barrier);
            })
        });
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &buffer_barriers,
                &image_barriers,
            );
        };
    }

    fn barrier_before(&self, device: &ash::Device) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier::default()
                    .buffer(*x.0)
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .offset(op.dst_offset)
                    .size(op.size);
                buffer_barriers.push(barrier);
            })
        });
        let size = self.upload_images.iter().map(|x| x.1.len()).sum::<usize>();
        let mut image_barriers = Vec::with_capacity(size);
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(*x.0)
                    .subresource_range(op.1);
                image_barriers.push(barrier);
            })
        });
        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::VERTEX_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &buffer_barriers,
                &image_barriers,
            );
        };
    }

    fn copy_buffers(&self, device: &ash::Device) {
        self.upload_buffers.iter().for_each(|x| unsafe {
            device.cmd_copy_buffer(self.command_buffer, self.staging, *x.0, x.1)
        })
    }

    fn copy_images(&self, device: &ash::Device) {
        self.upload_images.iter().for_each(|x| {
            let regions = x.1.iter().map(|x| x.0).collect::<Vec<_>>();
            unsafe {
                device.cmd_copy_buffer_to_image(
                    self.command_buffer,
                    self.staging,
                    *x.0,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            }
        })
    }

    pub fn free(&mut self, device: &ash::Device, allocator: &mut GpuAllocator) {
        if let Some(memory) = self.memory.take() {
            self.upload_impl(device, false).unwrap();
            unsafe {
                device.device_wait_idle().unwrap();
                device.destroy_command_pool(self.command_pool, None);
                device.destroy_semaphore(self.semaphore, None);
                device.destroy_fence(self.fence, None);
                device.destroy_buffer(self.staging, None);
                allocator.dealloc(AshMemoryDevice::wrap(device), memory);
            }
        }
    }
}
