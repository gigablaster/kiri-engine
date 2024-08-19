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
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk::{self, ImageSubresourceRange};
use kiri_backend::{AsVulkan, Buffer, BufferCreateDesc, Image, RenderDevice};
use kiri_common::BumpAllocator;

use crate::Error;

#[derive(Debug, Clone, Copy)]
struct ImageUploadRequest(vk::BufferImageCopy, vk::ImageSubresourceRange);

#[derive(Debug, Clone, Copy)]
pub struct ImageSubresourceData<'a> {
    pub data: &'a [u8],
}

#[derive(Debug)]
pub struct Staging {
    device: Arc<RenderDevice>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<(vk::CommandBuffer, vk::Fence)>,
    allocator: BumpAllocator,
    upload_buffers: HashMap<vk::Buffer, Vec<vk::BufferCopy>>,
    upload_images: HashMap<vk::Image, Vec<ImageUploadRequest>>,
    buffer: Buffer,
    mapping: NonNull<u8>,
    semaphores: Vec<vk::Semaphore>,
    render_semaphores: Vec<vk::Semaphore>,
    last: Option<usize>,
    current: usize,
    pending_buffer_barriers: Vec<(vk::Buffer, u64, u64)>,
    pending_image_barriers: Vec<(vk::Image, ImageSubresourceRange)>,
}

unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

const PAGE_SIZE: usize = 32 * 1024 * 1024;
const PAGE_COUNT: usize = 4;

fn create_semaphore(device: &ash::Device) -> Result<vk::Semaphore, Error> {
    let semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
    Ok(semaphore)
}

fn create_command_buffer(
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
    pub fn new(device: &Arc<RenderDevice>) -> Result<Self, Error> {
        let size = (PAGE_SIZE * PAGE_COUNT) as u64;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.get().create_command_pool(&pool_info, None) }?;
        let transfer_cbs = Vec::from_iter(
            (0..PAGE_COUNT).map(|_| create_command_buffer(device.get(), command_pool).unwrap()),
        );
        let semaphores =
            Vec::from_iter((0..PAGE_COUNT).map(|_| create_semaphore(device.get()).unwrap()));
        let render_semaphores =
            Vec::from_iter((0..PAGE_COUNT).map(|_| create_semaphore(device.get()).unwrap()));

        let mut buffer = Buffer::new(
            device,
            BufferCreateDesc::shared(PAGE_SIZE * PAGE_COUNT)
                .transfer_source()
                .dedicated(true),
        )?;
        let mapping = buffer.map()?;

        Ok(Self {
            device: device.clone(),
            command_pool,
            command_buffers: transfer_cbs,
            allocator: BumpAllocator::new(
                PAGE_SIZE as _,
                device
                    .physical_device()
                    .properties
                    .limits
                    .buffer_image_granularity as _,
            ),
            upload_buffers: Default::default(),
            upload_images: Default::default(),
            mapping,
            buffer,
            semaphores,
            render_semaphores,
            last: None,
            current: 0,
            pending_buffer_barriers: Default::default(),
            pending_image_barriers: Default::default(),
        })
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        target: &Buffer,
        offset: usize,
        data: &[T],
    ) -> Result<(), Error> {
        let mut current_offset = 0;
        loop {
            let data_len = mem::size_of_val(data);
            let pushed = self.try_push_buffer(
                target.as_vulkan(),
                offset + current_offset,
                data_len - current_offset,
                unsafe { (data.as_ptr() as *const u8).add(current_offset) },
            )?;
            current_offset += pushed;
            if current_offset == data_len {
                return Ok(());
            } else {
                self.upload_impl(false)?;
            }
        }
    }

    pub fn upload_image(
        &mut self,
        target: &Image,
        data: &[ImageSubresourceData],
    ) -> Result<(), Error> {
        for (mip, data) in data.iter().enumerate() {
            while !self.try_push_mip(target, mip as _, data)? {
                self.upload_impl(false)?;
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
            let dims = target.desc().dims;
            let op = vk::BufferImageCopy::default()
                .image_extent(vk::Extent3D {
                    width: dims[0] >> mip,
                    height: dims[1] >> mip,
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
                .entry(target.as_vulkan())
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

    pub fn upload(&mut self) -> Result<(vk::Semaphore, vk::PipelineStageFlags), Error> {
        self.upload_impl(true)
    }

    fn upload_impl(
        &mut self,
        client_will_wait: bool,
    ) -> Result<(vk::Semaphore, vk::PipelineStageFlags), Error> {
        puffin::profile_function!();
        let (cb, fence) = self.command_buffers[self.current];

        unsafe {
            self.device
                .get()
                .wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.get().reset_fences(&[fence])?;
            self.device
                .get()
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            self.device
                .get()
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
        }
        self.barrier_before(cb);
        self.copy_buffers(cb);
        self.copy_images(cb);
        self.defer_barrier_after();
        unsafe { self.device.get().end_command_buffer(cb) }?;

        let semaphore = self.semaphores[self.current];
        let render_semaphore = self.render_semaphores[self.current];
        let mut triggers = ArrayVec::<_, 2>::new();
        triggers.push(semaphore);
        if client_will_wait {
            // fixme?
            triggers.push(render_semaphore);
        }

        if let Some(last) = self.last {
            self.device.submit(
                self.command_buffers[self.current],
                &[(self.semaphores[last], vk::PipelineStageFlags::TRANSFER)],
                &triggers,
            )?;
        } else {
            self.device
                .submit(self.command_buffers[self.current], &[], &triggers)?;
        }

        self.last = Some(self.current);
        self.current += 1;
        self.current %= PAGE_COUNT;
        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();

        Ok((render_semaphore, vk::PipelineStageFlags::TRANSFER))
    }

    pub fn execute_pending_barriers(&mut self, device: &RenderDevice, cb: vk::CommandBuffer) {
        let image_barriers = self
            .pending_image_barriers
            .drain(..)
            .map(|x| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(x.0)
                    .subresource_range(x.1)
            })
            .collect::<Vec<_>>();
        let buffer_barriers = self
            .pending_buffer_barriers
            .drain(..)
            .map(|x| {
                vk::BufferMemoryBarrier::default()
                    .buffer(x.0)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::INDEX_READ)
                    .offset(x.1)
                    .size(x.2)
            })
            .collect::<Vec<_>>();
        unsafe {
            self.device.get().cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &image_barriers,
            );
            self.device.get().cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::BY_REGION,
                &[],
                &buffer_barriers,
                &[],
            );
        };
    }

    fn barrier_before(&self, cb: vk::CommandBuffer) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier::default()
                    .buffer(*x.0)
                    .src_access_mask(vk::AccessFlags::UNIFORM_READ)
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
            self.device.get().cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &buffer_barriers,
                &[],
            );
            self.device.get().cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &image_barriers,
            )
        }
    }

    fn defer_barrier_after(&mut self) {
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                self.pending_buffer_barriers
                    .push((*x.0, op.dst_offset, op.size));
            })
        });
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                self.pending_image_barriers.push((*x.0, op.1));
            })
        });
    }

    fn copy_buffers(&self, cb: vk::CommandBuffer) {
        self.upload_buffers.iter().for_each(|x| unsafe {
            self.device
                .get()
                .cmd_copy_buffer(cb, self.buffer.as_vulkan(), *x.0, x.1)
        })
    }

    fn copy_images(&self, cb: vk::CommandBuffer) {
        self.upload_images.iter().for_each(|x| {
            let regions = x.1.iter().map(|x| x.0).collect::<Vec<_>>();
            unsafe {
                self.device.get().cmd_copy_buffer_to_image(
                    cb,
                    self.buffer.as_vulkan(),
                    *x.0,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            }
        })
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        self.upload_impl(false).unwrap();
        unsafe {
            self.device.get().device_wait_idle().unwrap();
            self.device
                .get()
                .destroy_command_pool(self.command_pool, None);
        }

        self.command_buffers
            .iter()
            .for_each(|(_, fence)| unsafe { self.device.get().destroy_fence(*fence, None) });

        for index in 0..PAGE_COUNT {
            unsafe {
                self.device
                    .get()
                    .destroy_semaphore(self.semaphores[index], None);
                self.device
                    .get()
                    .destroy_semaphore(self.render_semaphores[index], None);
            }
        }
    }
}
