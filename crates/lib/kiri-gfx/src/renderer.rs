// Copyright (C) 2024 gigablaster

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
    mem::{self},
    ptr::NonNull,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk::{self};
use bevy_tasks::{block_on, ComputeTaskPool};
use kiri_backend::{
    compile_raster_pipeline, AcquiredSurface, Buffer, BufferCreateDesc, DescriptorSetCount,
    DescriptorSetDesc, DescriptorSetLayoutDesc, DescriptorSetType, Frame, GpuAllocator, Image,
    ImageCreateDesc, ImageViewDesc, InputVertexStreamLayout, Program, RasterPipelineCreateDesc,
    RenderAttachmentLayoutDesc, RenderDevice, Swapchain, MAX_BINDLESS_RESOURCES,
    MAX_COLOR_ATTACHMENTS,
};
use kiri_common::{Handle, HotColdPool, Pool, SentinelPoolStrategy, TempList};
use lazy_static::lazy_static;
use parking_lot::{Mutex, RwLock};

use crate::{
    record_barriers, DescriptorSetBuilder, DescriptorSetData, DrawStreamExecuteContext,
    DynamicGpuMemoryPool, Error, ImageUploadData, RenderContext, Staging,
};

pub type ImageHandle = Handle<Image>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type PipelineHandle = Handle<(vk::Pipeline, vk::PipelineLayout)>;
pub type DescriptorHandle = Handle<vk::DescriptorSet>;
pub type BindlessHandle = Handle<vk::ImageView>;

pub(super) type ImagePool = Pool<Image>;
pub(super) type BufferPool = HotColdPool<vk::Buffer, Buffer>;
pub(super) type PipelinePool = Pool<
    (vk::Pipeline, vk::PipelineLayout),
    SentinelPoolStrategy<(vk::Pipeline, vk::PipelineLayout)>,
>;
pub(super) type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;

pub(super) type BindlessPool = HotColdPool<vk::ImageView, (ImageHandle, ImageViewDesc)>;

lazy_static! {
    static ref BINDLESS_DESCRIPTOR_DESC: DescriptorSetLayoutDesc =
        DescriptorSetLayoutDesc::default()
            .slot(0, "images", vk::DescriptorType::SAMPLED_IMAGE, 0xffff,)
            .bindless();
}

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

#[derive(Debug)]
struct PipelineCompilationData {
    program: Arc<Program>,
    layout: RenderAttachmentLayoutDesc<'static>,
    streams: &'static [InputVertexStreamLayout<'static>],
    specializaton: Vec<(u32, u32)>,
    desc: RasterPipelineCreateDesc,
}

/// Low-level renderer
#[derive(Debug)]
pub struct Renderer {
    pub device: Arc<RenderDevice>,
    images: RwLock<ImagePool>,
    buffers: RwLock<BufferPool>,
    pipelines: RwLock<PipelinePool>,
    descriptors: RwLock<DescriptorPool>,
    bindless: RwLock<BindlessPool>,
    staging: Mutex<Staging>,
    pipelines_to_compile: Mutex<HashMap<PipelineHandle, PipelineCompilationData>>,
    dynamic_memory: Mutex<DynamicGpuMemoryPool>,
    bindless_to_update: Mutex<Vec<BindlessHandle>>,
    bindless_pool: vk::DescriptorPool,
    bindless_descriptor: vk::DescriptorSet,
}

unsafe impl Sync for Renderer {}
unsafe impl Send for Renderer {}

const MAX_RESOURCE_COUNT: usize = 0x7ffff;

impl Renderer {
    pub fn new(device: &Arc<RenderDevice>) -> Result<Arc<Self>, Error> {
        let layout =
            device.get_or_create_layout(vk::ShaderStageFlags::ALL, &BINDLESS_DESCRIPTOR_DESC)?;
        let bindless_pool = unsafe {
            device.raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(
                        &BINDLESS_DESCRIPTOR_DESC
                            .get_descriptor_count()
                            .to_pool_size(1),
                    )
                    .max_sets(1)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
        }?;
        let mut allocation_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(bindless_pool)
            .set_layouts(slice::from_ref(&layout));
        allocation_info.descriptor_set_count = 1;
        let bindless_descriptor =
            unsafe { device.raw.allocate_descriptor_sets(&allocation_info) }?[0];
        device.set_object_name(bindless_descriptor, "Bindless images");
        Ok(Arc::new(Self {
            device: device.clone(),
            staging: Mutex::new(Staging::new(device)?),
            images: RwLock::new(ImagePool::new(MAX_RESOURCE_COUNT)),
            buffers: RwLock::new(BufferPool::new(MAX_RESOURCE_COUNT)),
            pipelines: RwLock::new(PipelinePool::new(MAX_RESOURCE_COUNT)),
            bindless: RwLock::new(BindlessPool::new(MAX_RESOURCE_COUNT)),
            descriptors: RwLock::new(DescriptorPool::new(MAX_RESOURCE_COUNT)),
            pipelines_to_compile: Default::default(),
            dynamic_memory: Default::default(),
            bindless_to_update: Default::default(),
            bindless_pool,
            bindless_descriptor,
        }))
    }

    pub fn import_image(&self, image: Image) -> ImageHandle {
        self.images.write().push(image)
    }

    pub fn create_image(
        &self,
        allocator: &GpuAllocator,
        desc: ImageCreateDesc,
        data: Option<&[ImageUploadData]>,
    ) -> Result<ImageHandle, Error> {
        let image = Image::new(&self.device, allocator, desc)?;
        if let Some(data) = data {
            self.staging.lock().upload_image(&image, data)?;
        }
        Ok(self.import_image(image))
    }

    pub fn replace_image(&self, handle: ImageHandle, image: Image) {
        self.images.write().replace(handle, image);
        self.invalidate_image_views(handle);
    }

    pub fn update_image(
        &self,
        handle: ImageHandle,
        allocator: &GpuAllocator,
        desc: ImageCreateDesc,
        data: Option<&[ImageUploadData]>,
    ) -> Result<(), Error> {
        let image = Image::new(&self.device, allocator, desc)?;
        if let Some(data) = data {
            self.staging.lock().upload_image(&image, data)?;
        }
        self.replace_image(handle, image);
        Ok(())
    }

    pub fn destroy_image(&self, handle: ImageHandle) {
        self.images.write().remove(handle);
    }

    pub fn get_image_view(
        &self,
        handle: ImageHandle,
        desc: ImageViewDesc,
    ) -> Result<vk::ImageView, Error> {
        Ok(self
            .images
            .read()
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .view(desc)?)
    }

    pub fn import_buffer(&self, buffer: Buffer) -> BufferHandle {
        self.buffers.write().push(buffer.raw, buffer)
    }

    pub fn create_buffer(&self, desc: BufferCreateDesc) -> Result<BufferHandle, Error> {
        let buffer = Buffer::new(&self.device, desc)?;
        Ok(self.import_buffer(buffer))
    }

    pub fn upload_buffer<T: Copy>(&self, buffer: BufferPointer, data: &[T]) -> Result<(), Error> {
        let buffers = self.buffers.read();
        let vk_buffer = buffers
            .get_cold(buffer.handle)
            .ok_or(Error::InvalidBufferHandle(buffer.handle))?;
        self.staging
            .lock()
            .upload_buffer(vk_buffer, buffer.offset, data)?;
        Ok(())
    }

    pub fn destroy_buffer(&self, handle: BufferHandle) {
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

    pub fn create_pipeline(
        &self,
        program: &Arc<Program>,
        layout: RenderAttachmentLayoutDesc<'static>,
        streams: &'static [InputVertexStreamLayout<'static>],
        specialization: &[(u32, u32)],
        desc: RasterPipelineCreateDesc,
    ) -> PipelineHandle {
        let handle = self
            .pipelines
            .write()
            .push((vk::Pipeline::null(), program.pipeline_layout));
        self.pipelines_to_compile.lock().insert(
            handle,
            PipelineCompilationData {
                program: program.clone(),
                layout,
                streams,
                specializaton: specialization.to_vec(),
                desc,
            },
        );
        handle
    }

    pub fn destroy_pipeline(&self, handle: PipelineHandle) {
        if let Some((pipeline, _)) = self.pipelines.write().remove(handle) {
            unsafe { self.device.raw.destroy_pipeline(pipeline, None) }
        }
    }

    pub fn create_descriptor_set(
        &self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let data = builder.build(&self.device)?;
        Ok(self
            .descriptors
            .write()
            .push(vk::DescriptorSet::null(), data))
    }

    pub fn destroy_descriptor_set(&self, handle: DescriptorHandle) {
        self.descriptors.write().remove(handle);
    }

    pub fn create_bindless_view(&self, image: ImageHandle, desc: ImageViewDesc) -> BindlessHandle {
        let handle = self
            .bindless
            .write()
            .push(vk::ImageView::null(), (image, desc));
        self.bindless_to_update.lock().push(handle);
        handle
    }

    pub fn update_bindless_view(
        &self,
        handle: BindlessHandle,
        image: ImageHandle,
        desc: ImageViewDesc,
    ) {
        self.bindless
            .write()
            .replace_hot_cold(handle, vk::ImageView::null(), (image, desc));
        self.bindless_to_update.lock().push(handle);
    }

    pub fn destory_bindless_view(&self, handle: BindlessHandle) {
        self.bindless.write().remove(handle);
    }

    pub fn render<RenderCB: FnOnce(&RenderContext) -> Result<ImageHandle, Error>>(
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
        let pipeline_compilation = ComputeTaskPool::get().spawn(Self::compile_pipelines(
            self.device.clone(),
            mem::take(&mut self.pipelines_to_compile.lock()),
        ));
        let frame = self.device.begin_frame()?;
        let semaphore = self.staging.lock().upload()?;
        let mut pipelines = self.pipelines.write();
        let mut dynamic_memory = self.dynamic_memory.lock();
        dynamic_memory.recycle();
        let dynamic = dynamic_memory.get(self)?;
        drop(dynamic_memory);
        // Generate render streams
        let context = RenderContext::new(self, &dynamic, &self.descriptors, target.image);
        let image = render(&context)?;
        // Prepare
        let images = self.images.read();
        let buffers = self.buffers.read();
        self.update_descriptors(&frame, &images, &buffers)?;
        let compiled = block_on(pipeline_compilation);
        for (handle, data) in compiled {
            if let Some((old, _)) = pipelines.replace(handle, data) {
                if old != vk::Pipeline::null() {
                    unsafe { self.device.raw.destroy_pipeline(old, None) };
                }
            }
        }
        // Actual rendering
        let command_buffer =
            frame.get_command_buffer(&self.device.raw, vk::CommandBufferLevel::PRIMARY)?;
        let passes = context.passes.into_inner();
        unsafe {
            self.device.raw.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let mut descriptors = self.descriptors.write();
        let empty_descriptor_set = frame.get_descriptor(
            &self.device.raw,
            self.device.get_or_create_layout(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                &DescriptorSetLayoutDesc::default(),
            )?,
            DescriptorSetCount::default(),
        )?;
        let mut descriptors_to_clean = Vec::new();
        for pass in passes {
            self.device.begin_label(command_buffer, &pass.name);
            record_barriers(
                &self.device.raw,
                command_buffer,
                &images,
                &pass.image_barriers,
            )?;
            descriptors_to_clean.extend(&pass.descriptor_sets);
            let color_attachments = pass
                .color
                .iter()
                .map(|x| x.build(&images, vk::ImageAspectFlags::COLOR).unwrap())
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();
            let area = vk::Rect2D::default().extent(
                vk::Extent2D::default()
                    .width(target.image.desc.dims[0])
                    .height(target.image.desc.dims[1]),
            );
            let mut render_info = vk::RenderingInfo::default()
                .render_area(area)
                .color_attachments(&color_attachments)
                .layer_count(1);
            let depth = pass
                .depth
                .iter()
                .map(|x| x.build(&images, vk::ImageAspectFlags::DEPTH).unwrap())
                .next();
            if let Some(depth) = &depth {
                render_info = render_info.depth_attachment(depth);
            }
            unsafe {
                self.device
                    .raw
                    .cmd_begin_rendering(command_buffer, &render_info);
            }
            let context = DrawStreamExecuteContext {
                device: &self.device.raw,
                pipelines: &pipelines,
                descriptors: &descriptors,
                buffers: &buffers,
                empty: empty_descriptor_set,
                bindless: self.bindless_descriptor,
                render_area: area,
            };
            for stream in pass.streams {
                stream.execute(&context, command_buffer)?;
            }
            unsafe {
                self.device.raw.cmd_end_rendering(command_buffer);
            }
            self.device.end_labe(command_buffer);
        }
        unsafe {
            self.device.raw.end_command_buffer(command_buffer)?;
        }
        // Submit
        if let Some(semaphore) = semaphore {
            self.device.submit(
                &[command_buffer],
                frame.render_fence,
                &[(semaphore, vk::PipelineStageFlags2::VERTEX_INPUT)],
                &[(
                    frame.render_finished,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                )],
            )?;
        } else {
            self.device.submit(
                &[command_buffer],
                frame.render_fence,
                &[],
                &[(
                    frame.render_finished,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                )],
            )?;
        }
        // Present
        self.device
            .present(target, images.get(image).unwrap(), &frame)?;
        // Cleanup
        descriptors_to_clean.drain(..).for_each(|x| {
            descriptors.remove(x);
        });
        self.device.end_frame(frame);

        Ok(FrameState::Rendered)
    }

    async fn compile_pipelines(
        device: Arc<RenderDevice>,
        mut data: HashMap<PipelineHandle, PipelineCompilationData>,
    ) -> HashMap<PipelineHandle, (vk::Pipeline, vk::PipelineLayout)> {
        puffin::profile_function!();
        let result = ComputeTaskPool::get().scope(|s| {
            for (handle, data) in data.drain() {
                s.spawn(Self::compile_pipeline(device.clone(), handle, data));
            }
        });
        result.into_iter().map(|x| x.unwrap()).collect()
    }

    async fn compile_pipeline(
        device: Arc<RenderDevice>,
        handle: PipelineHandle,
        data: PipelineCompilationData,
    ) -> Result<(PipelineHandle, (vk::Pipeline, vk::PipelineLayout)), Error> {
        puffin::profile_function!();
        Ok((
            handle,
            (
                compile_raster_pipeline(
                    &device,
                    vk::PipelineCache::null(),
                    &data.program,
                    data.layout,
                    data.streams,
                    &data.specializaton,
                    data.desc,
                )?,
                data.program.pipeline_layout,
            ),
        ))
    }

    fn update_descriptors(
        &self,
        frame: &Frame,
        images: &ImagePool,
        buffers: &BufferPool,
    ) -> Result<(), Error> {
        puffin::profile_function!();
        frame.with_descriptor_allocator(&self.device.raw, |context| -> Result<(), Error> {
            let mut descriptors = self.descriptors.write();
            let image_writes = TempList::new();
            let buffer_writes = TempList::new();
            let mut writes = Vec::with_capacity(16384);
            let all_handles = descriptors
                .enumerate()
                .map(|(handle, _, _)| handle)
                .collect::<Vec<_>>();
            for handle in all_handles {
                // Allocate and assing new descriptor set
                let data = descriptors.get_cold(handle).unwrap();
                let descriptor_set = context.allocate(data.layout, data.count)?;
                descriptors.replace(handle, descriptor_set);
                let data = descriptors.get_cold_mut(handle).unwrap();

                // Process images
                for image in &mut data.images {
                    // Update image view if needed
                    if image.data.view == vk::ImageView::null() {
                        image.data.view = images
                            .get(image.data.handle)
                            .ok_or(Error::InvalidImageHandle(image.data.handle))?
                            .view(ImageViewDesc::new(image.data.aspect))?;
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
                            .dst_set(descriptor_set),
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
                            .dst_set(descriptor_set),
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
                            .dst_set(descriptor_set),
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
                            .dst_set(descriptor_set),
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
                            .dst_set(descriptor_set),
                    );
                }
            }
            let mut bindless = self.bindless.write();
            let mut bindless_to_update: Vec<_> = mem::take(&mut self.bindless_to_update.lock());
            bindless_to_update.sort();
            bindless_to_update.dedup();
            for handle in bindless_to_update {
                if let Some((image_handle, desc)) = bindless.get_cold(handle).copied() {
                    let image = images
                        .get(image_handle)
                        .ok_or(Error::InvalidImageHandle(image_handle))?;
                    let view = image.view(desc)?;
                    bindless.replace(handle, view);
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .image_info(slice::from_ref(
                                image_writes.add(
                                    vk::DescriptorImageInfo::default()
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(view),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .dst_array_element(handle.index())
                            .dst_binding(0)
                            .dst_set(self.bindless_descriptor),
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
        });
        let mut bindless_to_update = self.bindless_to_update.lock();
        let bindless = self.bindless.write();
        bindless
            .enumerate()
            .filter_map(|(handle, _, (image_handle, _))| {
                if *image_handle == image {
                    Some(handle)
                } else {
                    None
                }
            })
            .for_each(|handle| bindless_to_update.push(handle));
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .raw
                .destroy_descriptor_pool(self.bindless_pool, None)
        };
        self.pipelines
            .write()
            .drain()
            .for_each(|(pipeline, _)| unsafe { self.device.raw.destroy_pipeline(pipeline, None) })
    }
}
