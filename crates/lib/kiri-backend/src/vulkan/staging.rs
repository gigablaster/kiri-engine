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

use std::{
    collections::HashMap,
    mem::{self},
    ptr::{copy_nonoverlapping, NonNull},
};

use arrayvec::ArrayVec;
use ash::vk::{self, ImageSubresourceRange};
use gpu_alloc::{Dedicated, Request};
use gpu_alloc_ash::AshMemoryDevice;
use kiri_common::BumpAllocator;
use parking_lot::Mutex;

use crate::{Error, RenderDevice};

use super::{GpuAllocator, GpuMemory, Image, ImageSubresourceData, PhysicalDevice};

#[derive(Debug, Clone, Copy)]
struct ImageUploadRequest(vk::BufferImageCopy, vk::ImageSubresourceRange);

#[derive(Debug)]
pub(crate) struct Staging {
    command_pool: vk::CommandPool,
    command_buffers: Vec<(vk::CommandBuffer, vk::Fence)>,
    allocator: BumpAllocator,
    upload_buffers: HashMap<vk::Buffer, Vec<vk::BufferCopy>>,
    upload_images: HashMap<vk::Image, Vec<ImageUploadRequest>>,
    buffer: vk::Buffer,
    mapping: NonNull<u8>,
    semaphores: Vec<vk::Semaphore>,
    render_semaphores: Vec<vk::Semaphore>,
    last: Option<usize>,
    current: usize,
    pending_buffer_barriers: Mutex<Vec<(vk::Buffer, u64, u64)>>,
    pending_image_barriers: Mutex<Vec<(vk::Image, ImageSubresourceRange)>>,
    memory: Option<GpuMemory>,
}

unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

const PAGE_SIZE: usize = 32 * 1024 * 1024;
const PAGE_COUNT: usize = 4;

fn create_semaphore(device: &ash::Device) -> Result<vk::Semaphore, Error> {
    let semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
    Ok(semaphore)
}

fn create_transfer_command_buffer(
    device: &ash::Device,
    pool: vk::CommandPool,
) -> Result<(vk::CommandBuffer, vk::Fence), Error> {
    let cb = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(pool),
        )
    }?[0];
    let fence = unsafe {
        device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )
    }?;
    Ok((cb, fence))
}

impl Staging {
    pub fn new(
        device: &ash::Device,
        transfer_queue: u32,
        main_queue: u32,
        pdevice: &PhysicalDevice,
        allocator: &mut GpuAllocator,
    ) -> Result<Self, Error> {
        let size = (PAGE_SIZE * PAGE_COUNT) as u64;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(transfer_queue)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }?;
        let transfer_cbs = Vec::from_iter(
            (0..PAGE_COUNT).map(|_| create_transfer_command_buffer(device, command_pool).unwrap()),
        );
        let semaphores = Vec::from_iter((0..PAGE_COUNT).map(|_| create_semaphore(device).unwrap()));
        let render_semaphores =
            Vec::from_iter((0..PAGE_COUNT).map(|_| create_semaphore(device).unwrap()));

        let queues = [main_queue, transfer_queue];
        let buffer_info = vk::BufferCreateInfo::default()
            .queue_family_indices(&queues)
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let request = Request {
            size: requirements.size,
            align_mask: requirements.alignment,
            usage: gpu_alloc::UsageFlags::HOST_ACCESS,
            memory_types: requirements.memory_type_bits,
        };

        let mut memory = unsafe {
            allocator.alloc_with_dedicated(
                AshMemoryDevice::wrap(device),
                request,
                Dedicated::Required,
            )
        }?;
        unsafe { device.bind_buffer_memory(buffer, *memory.memory(), memory.offset()) }?;
        let mapping = unsafe {
            memory.map(
                AshMemoryDevice::wrap(device),
                0,
                (PAGE_SIZE * PAGE_COUNT) as _,
            )
        }?;

        Ok(Self {
            command_pool,
            command_buffers: transfer_cbs,
            allocator: BumpAllocator::new(
                PAGE_SIZE as _,
                pdevice.properties.limits.buffer_image_granularity as _,
            ),
            upload_buffers: HashMap::with_capacity(64),
            upload_images: HashMap::with_capacity(64),
            mapping,
            buffer,
            semaphores,
            render_semaphores,
            last: None,
            current: 0,
            pending_buffer_barriers: Mutex::default(),
            pending_image_barriers: Mutex::default(),
            memory: Some(memory),
        })
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        device: &RenderDevice,
        target: vk::Buffer,
        offset: usize,
        data: &[T],
    ) -> Result<(), Error> {
        let mut current_offset = 0;
        loop {
            let data_len = mem::size_of_val(data);
            let pushed = self.try_push_buffer(
                target,
                offset + current_offset,
                data_len - current_offset,
                unsafe { (data.as_ptr() as *const u8).add(current_offset) },
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
        device: &RenderDevice,
        target: &Image,
        data: &[ImageSubresourceData],
    ) -> Result<(), Error> {
        for (mip, data) in data.iter().enumerate() {
            while !self.try_push_mip(target, mip as _, data)? {
                self.upload_impl(device, false)?;
            }
        }
        Ok(())
    }

    fn try_push_mip(
        &mut self,
        target: &Image,
        mip: u32,
        data: &ImageSubresourceData,
    ) -> Result<bool, Error> {
        let size = data.data.len();
        if size > PAGE_SIZE {
            return Err(Error::ImageTooBig);
        }
        if let Some(allocated_offset) = self.allocator.allocate(size) {
            let buffer_offset = PAGE_SIZE * self.current + allocated_offset;
            unsafe {
                copy_nonoverlapping(
                    data.data.as_ptr(),
                    self.mapping.as_ptr().add(buffer_offset),
                    size,
                )
            };
            let op = vk::BufferImageCopy::default()
                .image_extent(vk::Extent3D {
                    width: target.desc.dims[0] >> mip,
                    height: target.desc.dims[1] >> mip,
                    depth: 1,
                })
                .buffer_offset(buffer_offset as _)
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
                .entry(target.raw)
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
        offset: usize,
        bytes: usize,
        data: *const u8,
    ) -> Result<usize, Error> {
        let can_send = self.allocator.validate(bytes as _);
        let allocated = self.allocator.allocate(bytes as _).unwrap(); // Already checked that allocator can allocate enough space
        let src_offset = PAGE_SIZE * self.current + allocated;
        unsafe { copy_nonoverlapping(data, self.mapping.as_ptr().add(src_offset), can_send) };
        let op = vk::BufferCopy::default()
            .src_offset(src_offset as _)
            .dst_offset(offset as _)
            .size(can_send as _);
        self.upload_buffers.entry(target).or_default().push(op);

        Ok(can_send)
    }

    pub fn upload(
        &mut self,
        device: &RenderDevice,
    ) -> Result<(vk::Semaphore, vk::PipelineStageFlags), Error> {
        self.upload_impl(device, true)
    }

    fn upload_impl(
        &mut self,
        device: &RenderDevice,
        client_will_wait: bool,
    ) -> Result<(vk::Semaphore, vk::PipelineStageFlags), Error> {
        puffin::profile_function!();
        let cb = &self.command_buffers[self.current];

        unsafe {
            device.device.wait_for_fences(&[cb.1], true, u64::MAX)?;
            device.device.reset_fences(&[cb.1])?;
            device
                .device
                .reset_command_buffer(cb.0, vk::CommandBufferResetFlags::empty())?;
        }
        {
            unsafe {
                device
                    .device
                    .begin_command_buffer(cb.0, &vk::CommandBufferBeginInfo::default())
            }?;
            self.barrier_before(&device.device, cb.0);
            self.copy_buffers(&device.device, cb.0);
            self.copy_images(&device.device, cb.0);
            self.barrier_after(device, cb.0);
            unsafe { device.device.end_command_buffer(cb.0) }?;
        }

        let semaphore = self.semaphores[self.current];
        let render_semaphore = self.render_semaphores[self.current];
        let mut triggers = ArrayVec::<_, 2>::new();
        triggers.push(semaphore);
        if client_will_wait {
            // fixme?
            triggers.push(render_semaphore);
        }

        if let Some(last) = self.last {
            device.submit_transfer(
                self.command_buffers[self.current],
                &[(self.semaphores[last], vk::PipelineStageFlags::TRANSFER)],
                &triggers,
            )?;
        } else {
            device.submit_transfer(self.command_buffers[self.current], &[], &triggers)?;
        }

        self.last = Some(self.current);
        self.current += 1;
        self.current %= PAGE_COUNT;
        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();

        Ok((render_semaphore, vk::PipelineStageFlags::TRANSFER))
    }

    pub fn execute_pending_barriers(&self, device: &RenderDevice, cb: vk::CommandBuffer) {
        let mut pending_buffer_barriers = self.pending_buffer_barriers.lock();
        let mut pending_image_barriers = self.pending_image_barriers.lock();

        let image_barriers = pending_image_barriers
            .drain(..)
            .map(|x| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(device.transfer_queue_index)
                    .dst_queue_family_index(device.universal_queue_index)
                    .image(x.0)
                    .subresource_range(x.1)
            })
            .collect::<Vec<_>>();
        let buffer_barriers = pending_buffer_barriers
            .drain(..)
            .map(|x| {
                vk::BufferMemoryBarrier::default()
                    .buffer(x.0)
                    .src_queue_family_index(device.transfer_queue_index)
                    .dst_queue_family_index(device.universal_queue_index)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .offset(x.1)
                    .size(x.2)
            })
            .collect::<Vec<_>>();
        unsafe {
            device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &image_barriers,
            );
            device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &buffer_barriers,
                &[],
            );
        };
    }

    fn barrier_before(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier::default()
                    .buffer(*x.0)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .src_access_mask(vk::AccessFlags::MEMORY_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    // .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
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
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    // .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(*x.0)
                    .subresource_range(op.1);
                image_barriers.push(barrier);
            })
        });
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &buffer_barriers,
                &image_barriers,
            )
        }
    }

    fn barrier_after(&self, device: &RenderDevice, cb: vk::CommandBuffer) {
        let mut pending_buffer_barriers = self.pending_buffer_barriers.lock();
        let mut pending_image_barriers = self.pending_image_barriers.lock();

        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                pending_buffer_barriers.push((*x.0, op.dst_offset, op.size));
            })
        });
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                pending_image_barriers.push((*x.0, op.1));
            })
        });

        if device.transfer_queue_index != device.universal_queue_index {
            let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
            let mut buffer_barriers = Vec::with_capacity(size);
            self.upload_buffers.iter().for_each(|x| {
                x.1.iter().for_each(|op| {
                    let barrier = vk::BufferMemoryBarrier::default()
                        .buffer(*x.0)
                        .src_queue_family_index(device.transfer_queue_index)
                        .dst_queue_family_index(device.universal_queue_index)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
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
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(device.transfer_queue_index)
                        .dst_queue_family_index(device.universal_queue_index)
                        .image(*x.0)
                        .subresource_range(op.1);
                    image_barriers.push(barrier);
                })
            });
            unsafe {
                device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &[],
                    &image_barriers,
                );
                device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &buffer_barriers,
                    &[],
                );
            }
        }
    }

    fn copy_buffers(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        self.upload_buffers
            .iter()
            .for_each(|x| unsafe { device.cmd_copy_buffer(cb, self.buffer, *x.0, x.1) })
    }

    fn copy_images(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        self.upload_images.iter().for_each(|x| {
            let regions = x.1.iter().map(|x| x.0).collect::<Vec<_>>();
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cb,
                    self.buffer,
                    *x.0,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            }
        })
    }

    pub(crate) fn free(&mut self, device: &RenderDevice) {
        self.upload_impl(device, false).unwrap();
        unsafe {
            device.device.device_wait_idle().unwrap();
            device.device.destroy_command_pool(self.command_pool, None);
        }

        self.command_buffers
            .iter()
            .for_each(|(_, fence)| unsafe { device.device.destroy_fence(*fence, None) });

        if let Some(memory) = self.memory.take() {
            device.with_drop_list(|drop_list| {
                drop_list.drop_buffer(self.buffer);
                drop_list.drop_memory(memory);
            })
        }

        for index in 0..PAGE_COUNT {
            unsafe {
                device
                    .device
                    .destroy_semaphore(self.semaphores[index], None);
                device
                    .device
                    .destroy_semaphore(self.render_semaphores[index], None);
            }
        }
    }
}
