// Copyright (C) 2024-2025 gigablaster

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

use std::{ptr::NonNull, sync::Arc};

use futures::executor::block_on;
use kiri_backend::{
    ash::{
        self,
        vk::{self},
    },
    AcquiredSurface, Buffer, BufferCreateDesc, DescriptorTotalCount, Image, RenderDevice,
    Swapchain, EMPTY_DESCRIPTOR_SET,
};
use kiri_common::{GameAppConfig, Handle, HotColdPool};
use parking_lot::{Mutex, RwLock};

use crate::{
    DescriptorHandle, DescriptorManager, DescriptorResolver, DescriptorSetBuilder,
    DescriptorUpdateContext, DynamicGpuMemory, DynamicGpuMemoryPool, DynamicWriter, Error,
    PipelineCache, PipelineResolver, RasterPipelineHandle,
};

pub type ImageHandle = Handle<vk::ImageView>;
pub type BufferHandle = Handle<vk::Buffer>;

type BufferPool = HotColdPool<vk::Buffer, Arc<Buffer>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameState {
    Rendered,
    NeedRecreateSwapchain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlice {
    pub handle: BufferHandle,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferPointer {
    pub handle: BufferHandle,
    pub offset: u64,
}

impl Default for BufferSlice {
    fn default() -> Self {
        Self {
            handle: Handle::default(),
            offset: u64::MAX,
            size: u64::MAX,
        }
    }
}

impl BufferSlice {
    pub fn new(handle: BufferHandle, offset: u64, size: u64) -> BufferSlice {
        Self {
            handle,
            offset,
            size,
        }
    }
}

impl BufferPointer {
    pub fn new(handle: BufferHandle, offset: u64) -> BufferPointer {
        Self { handle, offset }
    }
}

impl Default for BufferPointer {
    fn default() -> Self {
        Self {
            handle: Handle::default(),
            offset: u64::MAX,
        }
    }
}

impl From<BufferSlice> for BufferPointer {
    fn from(value: BufferSlice) -> Self {
        Self::new(value.handle, value.offset)
    }
}

pub struct RenderContext<'a> {
    renderer: &'a Renderer,
    dynamic: &'a DynamicGpuMemory,
    passes: Vec<Box<dyn PassDispatcher>>,
    update_descriptors: DescriptorUpdateContext<'a>,
    pub backbuffer: &'a Image,
}

unsafe impl Sync for RenderContext<'_> {}
unsafe impl Send for RenderContext<'_> {}

#[derive(Debug)]
pub struct RenderResourceResolver<'a> {
    buffers: &'a BufferPool,
    pipeline_resolver: PipelineResolver<'a>,
    descriptor_resolver: DescriptorResolver<'a>,
    pub empty_descriptor_set: vk::DescriptorSet,
    pub backbuffer: &'a Image,
}

impl<'a> RenderResourceResolver<'a> {
    pub(super) fn new(
        backbuffer: &'a Image,
        buffers: &'a BufferPool,
        pipeline_resolver: PipelineResolver<'a>,
        descriptor_resolver: DescriptorResolver<'a>,
        empty_descriptor_set: vk::DescriptorSet,
    ) -> Self {
        Self {
            buffers,
            pipeline_resolver,
            descriptor_resolver,
            empty_descriptor_set,
            backbuffer,
        }
    }

    pub fn resolve_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, Error> {
        self.buffers
            .get(handle)
            .copied()
            .ok_or(Error::InvalidBufferHandle(handle))
    }

    pub fn resolve_raster_pipeline(
        &self,
        handle: RasterPipelineHandle,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
        self.pipeline_resolver.resolve_raster_pipeline(handle)
    }

    pub fn resolve_descriptor_set(
        &self,
        handle: DescriptorHandle,
    ) -> Result<vk::DescriptorSet, Error> {
        self.descriptor_resolver.resolve(handle)
    }
}

pub trait PassDispatcher {
    fn name(&self) -> &str;
    fn dispatch(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        resolver: &RenderResourceResolver,
    ) -> Result<(), Error>;
}

impl<'a> RenderContext<'a> {
    pub(crate) fn new(
        renderer: &'a Renderer,
        dynamic: &'a DynamicGpuMemory,
        update_descriptors: DescriptorUpdateContext<'a>,
        backbuffer: &'a Image,
    ) -> Self {
        Self {
            renderer,
            dynamic,
            passes: Default::default(),
            update_descriptors,
            backbuffer,
        }
    }

    pub fn push_dynamic_data<T: Copy>(&self, data: &[T]) -> Result<BufferSlice, Error> {
        self.dynamic
            .push(&self.renderer.device.physical_device, data)
    }

    pub fn write_dynamic_data<T: Copy>(&self, count: usize) -> Result<DynamicWriter<T>, Error> {
        self.dynamic
            .write(&self.renderer.device.physical_device, count)
    }

    pub fn get_temprary_buffer(&self) -> BufferHandle {
        self.dynamic.get_buffer_handle()
    }

    pub fn update_descriptor(
        &mut self,
        handle: DescriptorHandle,
        builder: DescriptorSetBuilder,
    ) -> Result<(), Error> {
        self.update_descriptors.update_descriptor(handle, builder)
    }

    pub fn submit(&mut self, pass: Box<dyn PassDispatcher>) {
        self.passes.push(pass);
    }

    fn consume(self) -> Vec<Box<dyn PassDispatcher>> {
        self.passes
    }
}

/// Low-level renderer
#[derive(Debug)]
pub struct Renderer {
    pub device: Arc<RenderDevice>,
    buffers: RwLock<BufferPool>,
    dynamic_memory: Mutex<DynamicGpuMemoryPool>,
    pipeline_cache: PipelineCache,
    descriptor_manager: DescriptorManager,
}

unsafe impl Sync for Renderer {}
unsafe impl Send for Renderer {}

const MAX_RESOURCE_COUNT: usize = 64536;

impl Renderer {
    pub fn new(device: Arc<RenderDevice>, config: &GameAppConfig) -> Result<Arc<Self>, Error> {
        Ok(Arc::new(Self {
            buffers: RwLock::new(BufferPool::new(MAX_RESOURCE_COUNT)),
            dynamic_memory: Mutex::new(DynamicGpuMemoryPool::new(device.clone())),
            pipeline_cache: PipelineCache::new(device.clone(), config.cache()),
            descriptor_manager: DescriptorManager::new(device.clone()),
            device,
        }))
    }

    pub fn register_buffer(&self, buffer: Arc<Buffer>) -> BufferHandle {
        self.buffers.write().push(buffer.raw, buffer)
    }

    pub fn create_buffer(&self, desc: BufferCreateDesc) -> Result<BufferHandle, Error> {
        let buffer = Buffer::new(&self.device, desc)?;
        Ok(self.register_buffer(Arc::new(buffer)))
    }

    pub fn remove_buffer(&self, handle: BufferHandle) {
        self.buffers.write().remove(handle);
    }

    pub fn get_buffer_mapping(&self, handle: BufferHandle) -> Result<Option<NonNull<u8>>, Error> {
        Ok(self
            .buffers
            .write()
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBufferHandle(handle))?
            .mapping)
    }

    pub fn upload_buffer_data<T: Copy>(
        &self,
        target: BufferPointer,
        data: &[T],
    ) -> Result<(), Error> {
        self.buffers
            .read()
            .get_cold(target.handle)
            .ok_or(Error::InvalidBufferHandle(target.handle))?
            .upload(target.offset, data)?;
        Ok(())
    }

    pub fn render<RenderCB: FnOnce(&mut RenderContext) -> Result<(), Error>>(
        &self,
        swapchain: &Swapchain,
        render: RenderCB,
    ) -> Result<FrameState, Error> {
        puffin::profile_function!();
        // Preparations
        let target = match swapchain.acquire_next_image()? {
            AcquiredSurface::NeedRecreate => return Ok(FrameState::NeedRecreateSwapchain),
            AcquiredSurface::Image(target) => target,
        };
        let (frame, staging_semaphore) = self.device.begin_frame()?;
        let mut dynamic_memory = self.dynamic_memory.lock();
        dynamic_memory.recycle();
        let dynamic = dynamic_memory.take(self)?;

        // Generate render streams
        let mut context = RenderContext::new(
            self,
            &dynamic,
            self.descriptor_manager.update(),
            target.image,
        );
        render(&mut context)?;
        let passes = context.consume();
        self.descriptor_manager.update_descriptors()?;

        // Prepare
        let buffers = self.buffers.write();

        block_on(self.pipeline_cache.compile_pending_pipelines())?;

        // Actual rendering
        let command_buffer =
            frame.get_command_buffer(&self.device.raw, vk::CommandBufferLevel::PRIMARY)?;
        unsafe {
            self.device.raw.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let empty_descriptor_set = self
            .device
            .allocate_descriptor_sets(
                self.device.get_or_create_layout(
                    vk::ShaderStageFlags::ALL_GRAPHICS,
                    EMPTY_DESCRIPTOR_SET,
                )?,
                &DescriptorTotalCount::default(),
                1,
            )?
            .remove(0);
        let resolver = RenderResourceResolver::new(
            target.image,
            &buffers,
            self.pipeline_cache.resolve(),
            self.descriptor_manager.resolve(),
            *empty_descriptor_set.raw(),
        );
        for pass in passes {
            self.device.begin_label(command_buffer, pass.name());
            pass.dispatch(&self.device.raw, command_buffer, &resolver)?;
            self.device.end_label(command_buffer);
        }
        unsafe {
            self.device.raw.end_command_buffer(command_buffer)?;
        }
        drop(resolver);
        self.device.drop_descriptors([empty_descriptor_set]);
        // Submit
        self.device.submit(
            &[command_buffer],
            frame.render_fence,
            &[
                (
                    staging_semaphore,
                    vk::PipelineStageFlags::VERTEX_INPUT
                        | vk::PipelineStageFlags::FRAGMENT_SHADER
                        | vk::PipelineStageFlags::DRAW_INDIRECT,
                ),
                (
                    target.acquire_semaphore,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ),
            ],
            &[frame.render_finished],
        )?;
        // Present
        self.device.present(target, &frame)?;
        self.device.end_frame(frame);
        Ok(FrameState::Rendered)
    }
}
