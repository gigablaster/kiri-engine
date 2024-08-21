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

use ash::vk::{self};
use kiri_backend::{Buffer, BufferCreateDesc, Image, RenderDevice};
use kiri_common::BumpAllocator;

use crate::{Error, ImageUploadData};

#[derive(Debug, Clone, Copy)]
struct ImageUploadRequest(vk::BufferImageCopy, vk::ImageSubresourceRange);

#[derive(Debug)]
pub struct Staging {
    device: Arc<RenderDevice>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    allocator: BumpAllocator,
    upload_buffers: HashMap<vk::Buffer, Vec<vk::BufferCopy>>,
    upload_images: HashMap<vk::Image, Vec<ImageUploadRequest>>,
    staging: Buffer,
    mapping: NonNull<u8>,
    semaphore: vk::Semaphore,
}

unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

const STAGING_SIZE: usize = 64 * 1024 * 1024;

impl Staging {
    pub fn new(device: &Arc<RenderDevice>) -> Result<Self, Error> {
        let pool_info =
            vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe { device.raw.create_command_pool(&pool_info, None) }?;

        let command_buffer = unsafe {
            device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(command_pool),
            )
        }?[0];
        let fence = unsafe {
            device.raw.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }?;
        let semaphore = unsafe {
            device
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }?;
        let mut staging = Buffer::new(
            device,
            BufferCreateDesc::shared(STAGING_SIZE)
                .transfer_source()
                .dedicated(),
        )?;
        let mapping = staging.map()?;

        Ok(Self {
            device: device.clone(),
            fence,
            command_pool,
            command_buffer,
            allocator: BumpAllocator::new(
                STAGING_SIZE as _,
                device
                    .physical_device
                    .properties
                    .limits
                    .buffer_image_granularity as _,
            ),
            upload_buffers: Default::default(),
            upload_images: Default::default(),
            mapping,
            staging,
            semaphore,
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
                target.raw,
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

    pub fn upload_image(&mut self, target: &Image, data: &[ImageUploadData]) -> Result<(), Error> {
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
        data: &ImageUploadData,
    ) -> Result<bool, Error> {
        let size = data.data.len();
        if size > STAGING_SIZE {
            return Err(Error::ImageTooBig);
        }
        if let Some(offset) = self.allocator.allocate(size) {
            unsafe {
                copy_nonoverlapping(data.data.as_ptr(), self.mapping.as_ptr().add(offset), size)
            };
            let dims = target.desc.dims;
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
        let dst_offset = self.allocator.allocate(can_send).unwrap(); // Already checked that allocator can allocate enough space
        unsafe { copy_nonoverlapping(data, self.mapping.as_ptr().add(dst_offset), can_send) };
        let op = vk::BufferCopy::default()
            .src_offset(dst_offset as _)
            .dst_offset(offset as _)
            .size(can_send as _);
        self.upload_buffers.entry(target).or_default().push(op);

        Ok(can_send)
    }

    pub fn upload(&mut self) -> Result<Option<vk::Semaphore>, Error> {
        self.upload_impl(true)
    }

    fn upload_impl(&mut self, need_semaphore: bool) -> Result<Option<vk::Semaphore>, Error> {
        if self.upload_images.is_empty() && self.upload_buffers.is_empty() {
            return Ok(None);
        }
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.raw.reset_fences(&[self.fence])?;
            self.device
                .raw
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            self.device.raw.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.barrier_before();
            self.copy_buffers();
            self.copy_images();
            self.barriers_after();
            self.device.raw.end_command_buffer(self.command_buffer)?;
        }
        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();
        if need_semaphore {
            self.device.submit(
                &[self.command_buffer],
                self.fence,
                &[],
                &[(self.semaphore, vk::PipelineStageFlags2::DRAW_INDIRECT)],
            )?;
            Ok(Some(self.semaphore))
        } else {
            self.device
                .submit(&[self.command_buffer], self.fence, &[], &[])?;
            Ok(None)
        }
    }

    fn barriers_after(&mut self) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier2::default()
                    .buffer(*x.0)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .offset(op.dst_offset)
                    .size(op.size);
                buffer_barriers.push(barrier);
            })
        });
        let size = self.upload_images.iter().map(|x| x.1.len()).sum::<usize>();
        let mut image_barriers = Vec::with_capacity(size);
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(*x.0)
                    .subresource_range(op.1);
                image_barriers.push(barrier);
            })
        });
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default()
                    .buffer_memory_barriers(&buffer_barriers)
                    .image_memory_barriers(&image_barriers)
                    .dependency_flags(vk::DependencyFlags::BY_REGION),
            );
        };
    }

    fn barrier_before(&self) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier2::default()
                    .buffer(*x.0)
                    .src_access_mask(vk::AccessFlags2::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .offset(op.dst_offset)
                    .size(op.size);
                buffer_barriers.push(barrier);
            })
        });
        let size = self.upload_images.iter().map(|x| x.1.len()).sum::<usize>();
        let mut image_barriers = Vec::with_capacity(size);
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_access_mask(vk::AccessFlags2::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(*x.0)
                    .subresource_range(op.1);
                image_barriers.push(barrier);
            })
        });
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default()
                    .buffer_memory_barriers(&buffer_barriers)
                    .image_memory_barriers(&image_barriers)
                    .dependency_flags(vk::DependencyFlags::BY_REGION),
            );
        }
    }

    fn copy_buffers(&self) {
        self.upload_buffers.iter().for_each(|x| unsafe {
            self.device
                .raw
                .cmd_copy_buffer(self.command_buffer, self.staging.raw, *x.0, x.1)
        })
    }

    fn copy_images(&self) {
        self.upload_images.iter().for_each(|x| {
            let regions = x.1.iter().map(|x| x.0).collect::<Vec<_>>();
            unsafe {
                self.device.raw.cmd_copy_buffer_to_image(
                    self.command_buffer,
                    self.staging.raw,
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
            self.device.raw.device_wait_idle().unwrap();
            self.device
                .raw
                .destroy_command_pool(self.command_pool, None);
            self.device.raw.destroy_semaphore(self.semaphore, None);
            self.device.raw.destroy_fence(self.fence, None);
        }
    }
}
