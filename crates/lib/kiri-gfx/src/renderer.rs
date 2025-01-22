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

use std::{ptr::NonNull, slice, sync::Arc};

use futures::executor::block_on;
use kiri_backend::{
    ash::{
        self,
        vk::{self},
    },
    AcquiredSurface, Buffer, BufferCreateDesc, DescriptorSetLayoutDesc, DescriptorTotalCount,
    GpuDescriptor, Image, ImageCreateDesc, ImageDesc, ImageUploadData, ImageViewDesc, RenderDevice,
    Swapchain, EMPTY_DESCRIPTOR_SET,
};
use kiri_common::{GameAppConfig, Handle, HotColdPool, Pool, TempList};
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    DynamicGpuMemory, DynamicGpuMemoryPool, DynamicWriter, Error, PipelineCache, PipelineResolver,
    RasterPipelineHandle,
};

pub type ImageHandle = Handle<Image>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type DescriptorHandle = Handle<vk::DescriptorSet>;

type BufferPool = HotColdPool<vk::Buffer, Buffer>;
type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;
type ImagePool = Pool<Image>;

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
    passes: Option<Vec<Box<dyn PassDispatcher>>>,
    update_descriptors: DescriptorUpdateContext<'a>,
    temp_descriptors: Vec<DescriptorHandle>,
    pub backbuffer: &'a Image,
}

unsafe impl Sync for RenderContext<'_> {}
unsafe impl Send for RenderContext<'_> {}

#[derive(Debug)]
pub struct RenderResourceResolver<'a> {
    buffers: &'a BufferPool,
    images: &'a ImagePool,
    pipeline_resolver: PipelineResolver<'a>,
    descriptor_resolver: DescriptorResolver<'a>,
    pub empty_descriptor_set: vk::DescriptorSet,
    pub backbuffer: &'a Image,
}

impl<'a> RenderResourceResolver<'a> {
    fn new(
        backbuffer: &'a Image,
        buffers: &'a BufferPool,
        images: &'a ImagePool,
        pipeline_resolver: PipelineResolver<'a>,
        descriptor_resolver: DescriptorResolver<'a>,
        empty_descriptor_set: vk::DescriptorSet,
    ) -> Self {
        Self {
            buffers,
            images,
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

    pub fn resolve_image_view(
        &self,
        handle: ImageHandle,
        desc: ImageViewDesc,
    ) -> Result<vk::ImageView, Error> {
        Ok(self
            .images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .view(desc)?)
    }

    pub fn resolve_image(&self, handle: ImageHandle) -> Result<vk::Image, Error> {
        Ok(self
            .images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .raw)
    }

    pub fn resolve_image_desc(&self, handle: ImageHandle) -> Result<&ImageDesc, Error> {
        Ok(&self
            .images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .desc)
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
            temp_descriptors: Default::default(),
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

    pub fn submit(&mut self, pass: Box<dyn PassDispatcher>) {
        self.passes.get_or_insert_default().push(pass);
    }

    pub fn create_temp_descriptor(
        &mut self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let handle = self.update_descriptors.create_descriptor(builder)?;
        self.temp_descriptors.push(handle);
        Ok(handle)
    }

    fn consume(mut self) -> Vec<Box<dyn PassDispatcher>> {
        self.passes.take().unwrap_or_default()
    }
}

impl Drop for RenderContext<'_> {
    fn drop(&mut self) {
        self.temp_descriptors
            .drain(..)
            .for_each(|handle| self.update_descriptors.destroy_descriptor(handle));
    }
}

#[derive(Debug, Clone, Copy)]
struct Binding<T: Copy> {
    pub slot: u32,
    pub element: u32,
    pub ty: vk::DescriptorType,
    pub data: T,
}

#[derive(Debug, Clone, Copy)]
struct ImageBindingData {
    image: ImageHandle,
    desc: ImageViewDesc,
}

#[derive(Debug, Clone, Copy)]
struct StaticBufferBindingData {
    pub buffer: BufferHandle,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, Copy)]
struct DynamicBufferBindingData {
    pub buffer: BufferHandle,
    pub size: u32,
}

#[derive(Debug)]
struct DescriptorSetData {
    descriptor: Option<GpuDescriptor>,
    count: DescriptorTotalCount,
    layout: vk::DescriptorSetLayout,
    images: Vec<Binding<ImageBindingData>>,
    unifom_buffers: Vec<Binding<StaticBufferBindingData>>,
    storage_buffers: Vec<Binding<StaticBufferBindingData>>,
    dynamic_uniform_buffers: Vec<Binding<DynamicBufferBindingData>>,
    dynamic_storage_buffers: Vec<Binding<DynamicBufferBindingData>>,
    name: Option<String>,
}

#[derive(Debug)]
pub struct DescriptorSetBuilder<'a> {
    layout: DescriptorSetLayoutDesc<'static>,
    stages: vk::ShaderStageFlags,
    images: Vec<Binding<ImageBindingData>>,
    unifom_buffers: Vec<Binding<StaticBufferBindingData>>,
    storage_buffers: Vec<Binding<StaticBufferBindingData>>,
    dynamic_uniform_buffers: Vec<Binding<DynamicBufferBindingData>>,
    dynamic_storage_buffers: Vec<Binding<DynamicBufferBindingData>>,
    name: Option<&'a str>,
}

impl<'a> DescriptorSetBuilder<'a> {
    pub fn new(stages: vk::ShaderStageFlags, layout: DescriptorSetLayoutDesc<'static>) -> Self {
        let count = layout.get_descriptor_count();
        Self {
            layout,
            stages,
            images: Vec::with_capacity((count.sampled_image + count.combined_image_sampler) as _),
            unifom_buffers: Vec::with_capacity(count.uniform_buffer as _),
            storage_buffers: Vec::with_capacity(count.storage_buffer as _),
            dynamic_uniform_buffers: Vec::with_capacity(count.uniform_buffer_dynamic as _),
            dynamic_storage_buffers: Vec::with_capacity(count.storage_buffer_dynamic as _),
            name: None,
        }
    }

    pub fn bind_image(
        mut self,
        slot: &str,
        image: ImageHandle,
        desc: ImageViewDesc,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::TextureSlotNotFound(slot.to_owned()))?;
        self.images.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: ImageBindingData { image, desc },
        });
        Ok(self)
    }

    pub fn bind_uniform_buffer(
        mut self,
        slot: &str,
        buffer: BufferHandle,
        offset: usize,
        size: usize,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.unifom_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: StaticBufferBindingData {
                offset: offset as u32,
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn bind_storage_buffer(
        mut self,
        slot: &str,
        buffer: BufferHandle,
        offset: usize,
        size: u32,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.storage_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: StaticBufferBindingData {
                offset: offset as u32,
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn bind_dynamic_uniform_buffer(
        mut self,
        slot: &str,
        buffer: BufferHandle,
        size: usize,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.dynamic_uniform_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData {
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn bind_dynamic_storage_buffer(
        mut self,
        slot: &str,
        buffer: BufferHandle,
        size: usize,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.dynamic_storage_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData {
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    fn build(self, device: &RenderDevice) -> Result<DescriptorSetData, Error> {
        Ok(DescriptorSetData {
            descriptor: None,
            count: self.layout.get_descriptor_count(),
            layout: device.get_or_create_layout(self.stages, self.layout)?,
            images: self.images,
            unifom_buffers: self.unifom_buffers,
            storage_buffers: self.storage_buffers,
            dynamic_uniform_buffers: self.dynamic_uniform_buffers,
            dynamic_storage_buffers: self.dynamic_storage_buffers,
            name: self.name.map(|x| x.to_owned()),
        })
    }
}

const MAX_DESCRIPTORS: usize = 16384;

#[derive(Debug)]
struct DescriptorResolver<'a> {
    descriptors: RwLockReadGuard<'a, DescriptorPool>,
}

impl DescriptorResolver<'_> {
    pub fn resolve(&self, handle: DescriptorHandle) -> Result<vk::DescriptorSet, Error> {
        self.descriptors
            .get(handle)
            .copied()
            .ok_or(Error::InvalidDescriptorHandle(handle))
    }
}

pub struct DescriptorUpdateContext<'a> {
    device: &'a RenderDevice,
    descriptors: RwLockWriteGuard<'a, DescriptorPool>,
    dirty: MutexGuard<'a, Vec<DescriptorHandle>>,
    to_destroy: MutexGuard<'a, Vec<DescriptorHandle>>,
}

impl DescriptorUpdateContext<'_> {
    pub fn create_descriptor(
        &mut self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let data = builder.build(self.device)?;
        let handle = self.descriptors.push(vk::DescriptorSet::null(), data);
        self.dirty.push(handle);
        Ok(handle)
    }

    pub fn update_descriptor(
        &mut self,
        handle: DescriptorHandle,
        builder: DescriptorSetBuilder,
    ) -> Result<(), Error> {
        let data = builder.build(self.device)?;
        if self.descriptors.replace_cold(handle, data).is_some() {
            self.dirty.push(handle);
        }
        Ok(())
    }

    pub fn destroy_descriptor(&mut self, handle: DescriptorHandle) {
        self.to_destroy.push(handle);
    }
}

/// Low-level renderer
#[derive(Debug)]
pub struct Renderer {
    pub device: Arc<RenderDevice>,
    buffers: RwLock<BufferPool>,
    images: RwLock<ImagePool>,
    dynamic_memory: Mutex<DynamicGpuMemoryPool>,
    pipeline_cache: PipelineCache,
    descriptors: RwLock<DescriptorPool>,
    dirty_descriptors: Mutex<Vec<DescriptorHandle>>,
    descriptors_to_destroy: Mutex<Vec<DescriptorHandle>>,
    buffers_to_destroy: Mutex<Vec<BufferHandle>>,
    images_to_destroy: Mutex<Vec<ImageHandle>>,
}

unsafe impl Sync for Renderer {}
unsafe impl Send for Renderer {}

const MAX_RESOURCE_COUNT: usize = 64536;

impl Renderer {
    pub fn new(device: Arc<RenderDevice>, config: &GameAppConfig) -> Result<Arc<Self>, Error> {
        Ok(Arc::new(Self {
            buffers: RwLock::new(BufferPool::new(MAX_RESOURCE_COUNT)),
            images: RwLock::new(ImagePool::new(MAX_RESOURCE_COUNT)),
            dynamic_memory: Mutex::new(DynamicGpuMemoryPool::new(device.clone())),
            pipeline_cache: PipelineCache::new(device.clone(), config.cache()),
            descriptors: RwLock::new(HotColdPool::new(MAX_DESCRIPTORS)),
            dirty_descriptors: Default::default(),
            descriptors_to_destroy: Default::default(),
            buffers_to_destroy: Default::default(),
            images_to_destroy: Default::default(),
            device,
        }))
    }

    pub fn register_buffer(&self, buffer: Buffer) -> BufferHandle {
        self.buffers.write().push(buffer.raw, buffer)
    }

    pub fn create_buffer(&self, desc: BufferCreateDesc) -> Result<BufferHandle, Error> {
        Ok(self.register_buffer(Buffer::new(&self.device, desc)?))
    }

    pub fn destroy_buffer(&self, handle: BufferHandle) {
        self.buffers_to_destroy.lock().push(handle);
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

    pub fn register_image(&self, image: Image) -> ImageHandle {
        self.images.write().push(image)
    }

    pub fn create_image(
        &self,
        desc: ImageCreateDesc,
        data: Option<&[ImageUploadData]>,
    ) -> Result<ImageHandle, Error> {
        Ok(self.register_image(Image::new(&self.device, desc, data)?))
    }

    pub fn destroy_image(&self, handle: ImageHandle) {
        self.images_to_destroy.lock().push(handle);
    }

    pub fn with_descriptors(&self) -> DescriptorUpdateContext {
        let descriptors = self.descriptors.write();
        let dirty = self.dirty_descriptors.lock();
        let to_destroy = self.descriptors_to_destroy.lock();
        DescriptorUpdateContext {
            device: &self.device,
            descriptors: descriptors,
            dirty: dirty,
            to_destroy: to_destroy,
        }
    }

    // We want descriptors, buffers and images to be locked at this point. So we pass them from outside.
    fn update_descriptors(
        &self,
        descriptors: &mut DescriptorPool,
        buffers: &BufferPool,
        images: &ImagePool,
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let mut drop_list = Vec::new();
        let mut dirty = self.dirty_descriptors.lock();
        self.device
            .with_descriptor_allocator(|context| -> Result<(), Error> {
                let image_writes = TempList::new();
                let buffer_writes = TempList::new();
                let mut writes = Vec::with_capacity(MAX_DESCRIPTORS);
                // Process all dirty descriptors
                for handle in dirty.iter().copied() {
                    // Allocate and assing new descriptor set
                    let data = if let Some(data) = descriptors.get_cold_mut(handle) {
                        data
                    } else {
                        // Skip invalid descriptors
                        continue;
                    };
                    let descriptor_set = context.allocate(data.layout, &data.count, 1)?.remove(0);
                    let ds = *descriptor_set.raw();
                    // Remove old descriptor if any
                    if let Some(descriptor) = data.descriptor.replace(descriptor_set) {
                        drop_list.push(descriptor);
                    }
                    if let Some(name) = &data.name {
                        self.device.set_object_name(ds, name);
                    }

                    // Process images
                    for image in &data.images {
                        // Add to write list.
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .image_info(slice::from_ref(
                                    image_writes.add(
                                        vk::DescriptorImageInfo::default()
                                            .image_view(
                                                images
                                                    .get(image.data.image)
                                                    .ok_or(Error::InvalidImageHandle(
                                                        image.data.image,
                                                    ))?
                                                    .view(image.data.desc)?,
                                            )
                                            .image_layout(
                                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                            ),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(image.ty)
                                .dst_array_element(image.element)
                                .dst_binding(image.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process uniform buffers
                    for buffer in &data.unifom_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(*buffers.get(buffer.data.buffer).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.buffer),
                                            )?)
                                            .offset(buffer.data.offset as _)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process storage buffers
                    for buffer in &data.storage_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(*buffers.get(buffer.data.buffer).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.buffer),
                                            )?)
                                            .offset(buffer.data.offset as _)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process dynamic uniform buffers
                    for buffer in &data.dynamic_uniform_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(*buffers.get(buffer.data.buffer).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.buffer),
                                            )?)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process dynamic storage buffers
                    for buffer in &data.dynamic_storage_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(*buffers.get(buffer.data.buffer).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.buffer),
                                            )?)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                }
                unsafe { self.device.raw.update_descriptor_sets(&writes, &[]) };
                Ok(())
            })?;
        // Update actual vulkan objects for all dirty descriptors
        for handle in dirty.iter().copied() {
            let ds = *descriptors
                .get_cold(handle)
                .unwrap()
                .descriptor
                .as_ref()
                .unwrap()
                .raw();
            descriptors.replace(handle, ds);
        }
        dirty.clear();
        self.device.drop_descriptors(drop_list);
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
        drop(dynamic_memory);

        // Generate render streams
        let mut context = RenderContext::new(self, &dynamic, self.with_descriptors(), target.image);
        render(&mut context)?;
        let passes = context.consume();
        let mut descriptors = self.descriptors.write();
        let mut buffers = self.buffers.write();
        let mut images = self.images.write();

        self.update_descriptors(&mut descriptors, &buffers, &images)?;

        // Prepare
        block_on(self.pipeline_cache.compile_pending_pipelines())?;

        let mut buffers_to_destroy = self.buffers_to_destroy.lock();
        let mut images_to_destroy = self.images_to_destroy.lock();
        let mut descriptors_to_destroy = self.descriptors_to_destroy.lock();

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
            &images,
            self.pipeline_cache.resolve(),
            DescriptorResolver {
                descriptors: self.descriptors.read(),
            },
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
        let mut descriptors_to_drop = Vec::new();
        descriptors_to_drop.push(empty_descriptor_set);
        for handle in descriptors_to_destroy.drain(..) {
            if let Some((_, mut data)) = descriptors.remove(handle) {
                if let Some(descriptor) = data.descriptor.take() {
                    descriptors_to_drop.push(descriptor);
                }
            }
        }
        self.device.drop_descriptors(descriptors_to_drop);
        buffers_to_destroy.drain(..).for_each(|handle| {
            buffers.remove(handle);
        });
        images_to_destroy.drain(..).for_each(|handle| {
            images.remove(handle);
        });
        drop(descriptors);
        drop(buffers);
        drop(images);
        drop(buffers_to_destroy);
        drop(images_to_destroy);
        drop(descriptors_to_destroy);
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
