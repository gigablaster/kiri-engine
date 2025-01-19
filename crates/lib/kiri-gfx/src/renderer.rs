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

use core::slice;
use std::{ptr::NonNull, sync::Arc};

use futures::executor::block_on;
use kiri_backend::{
    ash::vk::{self},
    AcquiredSurface, Buffer, BufferCreateDesc, DescriptorSetLayoutDesc, DescriptorTotalCount,
    GpuDescriptor, Image, ImageViewDesc, RenderDevice, Swapchain,
};
use kiri_common::{GameAppConfig, Handle, HotColdPool, TempList};
use parking_lot::{Mutex, RwLock};

use crate::{
    DescriptorSetBuilder, DescriptorSetData, DynamicGpuMemoryPool, Error, PipelineCache,
    RenderContext, RenderResourceResolver,
};

pub type ImageHandle = Handle<vk::ImageView>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type DescriptorHandle = Handle<Option<GpuDescriptor>>;

pub(super) type ImagePool = HotColdPool<vk::ImageView, (Arc<Image>, ImageViewDesc)>;
pub(super) type BufferPool = HotColdPool<vk::Buffer, Arc<Buffer>>;
pub(super) type DescriptorPool = HotColdPool<Option<GpuDescriptor>, DescriptorSetData>;

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

/// Low-level renderer
#[derive(Debug)]
pub struct Renderer {
    pub device: Arc<RenderDevice>,
    images: RwLock<ImagePool>,
    buffers: RwLock<BufferPool>,
    descriptors: RwLock<DescriptorPool>,
    sampled_images_to_update: Mutex<Vec<ImageHandle>>,
    storage_images_to_update: Mutex<Vec<ImageHandle>>,
    storage_buffers_to_update: Mutex<Vec<BufferHandle>>,
    dynamic_memory: Mutex<DynamicGpuMemoryPool>,
    pipeline_cache: PipelineCache,
    dirty_descriptors: Mutex<Vec<DescriptorHandle>>,
    descriptors_to_destroy: Mutex<Vec<DescriptorHandle>>,
    empty_descriptor_set: GpuDescriptor,
}

unsafe impl Sync for Renderer {}
unsafe impl Send for Renderer {}

const MAX_RESOURCE_COUNT: usize = 64536;
const MAX_DESCRIPTORS: usize = 8192;

impl Renderer {
    pub fn new(device: &Arc<RenderDevice>, config: &GameAppConfig) -> Result<Arc<Self>, Error> {
        Ok(Arc::new(Self {
            device: device.clone(),
            images: RwLock::new(ImagePool::new(MAX_RESOURCE_COUNT)),
            buffers: RwLock::new(BufferPool::new(MAX_RESOURCE_COUNT)),
            descriptors: RwLock::new(DescriptorPool::new(MAX_DESCRIPTORS)),
            dynamic_memory: Default::default(),
            sampled_images_to_update: Default::default(),
            storage_images_to_update: Default::default(),
            storage_buffers_to_update: Default::default(),
            pipeline_cache: PipelineCache::new(device, config.cache()),
            dirty_descriptors: Default::default(),
            descriptors_to_destroy: Default::default(),
            empty_descriptor_set: device
                .allocate_descriptor_sets(
                    device.get_or_create_layout(
                        vk::ShaderStageFlags::ALL,
                        &DescriptorSetLayoutDesc::default(),
                    )?,
                    &DescriptorTotalCount::default(),
                    1,
                )?
                .remove(0),
        }))
    }

    pub fn register_image(
        &self,
        image: Arc<Image>,
        desc: ImageViewDesc,
    ) -> Result<ImageHandle, Error> {
        let view = image.view(desc)?;
        let usage = image.desc.usage;
        let handle = self.images.write().push(view, (image, desc));
        self.update_bindless_image(handle, usage);
        Ok(handle)
    }

    pub fn replace_image(&self, handle: ImageHandle, image: Arc<Image>) -> Result<(), Error> {
        let mut images = self.images.write();
        let desc = images
            .get_cold(handle)
            .map(|(_, desc)| desc)
            .copied()
            .ok_or(Error::InvalidImageHandle(handle))?;
        let view = image.view(desc)?;
        let usage = image.desc.usage;
        images.replace_hot_cold(handle, view, (image, desc));
        drop(images);
        self.invalidate_image_views(handle);
        self.update_bindless_image(handle, usage);
        Ok(())
    }

    pub fn remove_image(&self, handle: ImageHandle) {
        self.images.write().remove(handle);
    }

    fn update_bindless_image(&self, handle: ImageHandle, usage: vk::ImageUsageFlags) {
        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            self.sampled_images_to_update.lock().push(handle);
        }
        if usage.contains(vk::ImageUsageFlags::STORAGE) {
            self.storage_images_to_update.lock().push(handle);
        }
    }

    pub fn register_buffer(&self, buffer: Arc<Buffer>) -> BufferHandle {
        let need_update = buffer
            .desc
            .usage
            .contains(vk::BufferUsageFlags::STORAGE_BUFFER);
        let handle = self.buffers.write().push(buffer.raw, buffer);
        if need_update {
            self.storage_buffers_to_update.lock().push(handle);
        }
        handle
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

    pub fn create_descriptor_set(
        &self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let data = builder.build(&self.device)?;
        let handle = self.descriptors.write().push(None, data);
        self.dirty_descriptors.lock().push(handle);
        Ok(handle)
    }

    pub fn destroy_descriptor_set(&self, handle: DescriptorHandle) {
        self.descriptors_to_destroy.lock().push(handle);
    }

    pub fn render<RenderCB: FnOnce(&RenderContext) -> Result<(), Error>>(
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
        let dynamic = dynamic_memory.get(self)?;

        // Generate render streams
        let context = RenderContext::new(self, &dynamic, &self.descriptors, target.image);
        render(&context)?;

        // Prepare
        let images = self.images.write();
        let buffers = self.buffers.write();

        self.update_descriptors(&images, &buffers)?;

        block_on(self.pipeline_cache.compile_pending_pipelines())?;

        // Actual rendering
        let command_buffer =
            frame.get_command_buffer(&self.device.raw, vk::CommandBufferLevel::PRIMARY)?;
        let (mut trash_descriptors, passes) = context.consume();
        unsafe {
            self.device.raw.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let mut descriptors = self.descriptors.write();
        let resolver = RenderResourceResolver::new(
            target.image,
            &buffers,
            &images,
            &descriptors,
            self.pipeline_cache.resolve_pipelines(),
            *self.empty_descriptor_set.raw(),
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
        // Submit
        self.device.submit(
            &[command_buffer],
            frame.render_fence,
            &[
                (
                    staging_semaphore,
                    vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::FRAGMENT_SHADER,
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
        // Cleanup
        trash_descriptors.drain(..).for_each(|x| {
            descriptors.remove(x);
        });
        self.device.end_frame(frame);

        Ok(FrameState::Rendered)
    }

    fn update_descriptors(&self, images: &ImagePool, buffers: &BufferPool) -> Result<(), Error> {
        puffin::profile_function!();
        let mut descriptors = self.descriptors.write();
        let mut drop_list = Vec::new();
        for to_destroy in self.descriptors_to_destroy.lock().drain(..) {
            if let Some((mut descriptor, _)) = descriptors.remove(to_destroy) {
                if let Some(descriptor) = descriptor.take() {
                    drop_list.push(descriptor);
                }
            }
        }
        self.device.drop_descriptors(drop_list);
        self.device
            .with_descriptor_allocator(|context| -> Result<(), Error> {
                let dirty = self.dirty_descriptors.lock().drain(..).collect::<Vec<_>>();
                let image_writes = TempList::new();
                let buffer_writes = TempList::new();
                let mut writes = Vec::with_capacity(16384);
                for handle in dirty {
                    // Allocate and assing new descriptor set
                    let data = descriptors.get_cold(handle).unwrap();
                    let descriptor_set = context.allocate(data.layout, &data.count, 1)?.remove(0);
                    let ds = *descriptor_set.raw();
                    descriptors.replace(handle, Some(descriptor_set));
                    let data = descriptors.get_cold_mut(handle).unwrap();
                    if let Some(name) = &data.name {
                        self.device.set_object_name(ds, name);
                    }

                    // Process images
                    for image in &mut data.images {
                        // Update image view if needed
                        if image.data.view == vk::ImageView::null() {
                            image.data.view = images
                                .get(image.data.handle)
                                .copied()
                                .ok_or(Error::InvalidImageHandle(image.data.handle))?;
                        }
                        // Add to write list.
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .image_info(slice::from_ref(
                                    image_writes.add(
                                        vk::DescriptorImageInfo::default()
                                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                            .image_view(image.data.view),
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
                                            .buffer(*buffers.get(buffer.data.handle).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.handle),
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
                                            .buffer(*buffers.get(buffer.data.handle).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.handle),
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
                                            .buffer(*buffers.get(buffer.data.handle).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.handle),
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
                                            .buffer(*buffers.get(buffer.data.handle).ok_or(
                                                Error::InvalidBufferHandle(buffer.data.handle),
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
        Ok(())
    }

    fn invalidate_image_views(&self, image: ImageHandle) {
        self.descriptors.write().for_each_mut(|_, data| {
            data.images
                .iter_mut()
                .filter(|x| x.data.handle == image)
                .for_each(|x| x.data.view = vk::ImageView::null());
        })
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe { self.device.raw.device_wait_idle() }.unwrap();
    }
}
