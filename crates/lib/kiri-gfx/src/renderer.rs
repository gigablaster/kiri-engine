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

use ash::vk::{self};
use bevy_tasks::{block_on, ComputeTaskPool};
use kiri_backend::{
    compile_raster_pipeline, AcquiredSurface, Buffer, BufferCreateDesc, Frame, Image,
    ImageCreateDesc, ImageViewDesc, InputVertexStreamLayout, Program, RasterPipelineCreateDesc,
    RenderAttachmentLayoutDesc, RenderDevice, Swapchain,
};
use kiri_common::{Handle, HotColdPool, Pool, SentinelPoolStrategy, TempList};
use parking_lot::{Mutex, RwLock};

use crate::{
    DescriptorSetBuilder, DescriptorSetData, DynamicGpuMemoryPool, Error, ImageUploadData,
    RenderContext, Resolution, Staging, TempImagePool,
};

pub type ImageHandle = Handle<Image>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type PipelineHandle = Handle<(vk::Pipeline, vk::PipelineLayout)>;
pub type DescriptorHandle = Handle<vk::DescriptorSet>;

pub(super) type ImagePool = Pool<Image>;
pub(super) type BufferPool = HotColdPool<vk::Buffer, Buffer>;
pub(super) type PipelinePool = Pool<
    (vk::Pipeline, vk::PipelineLayout),
    SentinelPoolStrategy<(vk::Pipeline, vk::PipelineLayout)>,
>;
pub(super) type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;

pub enum FrameState {
    Rendered,
    NeedRecreateSwapchain,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlice {
    pub buffer: BufferHandle,
    pub offset: u32,
}

impl BufferSlice {
    pub fn new(buffer: BufferHandle, offset: usize) -> BufferSlice {
        Self {
            buffer,
            offset: offset as u32,
        }
    }
}

#[derive(Debug)]
struct PipelineCompilationData {
    program: Arc<Program>,
    layout: RenderAttachmentLayoutDesc<'static>,
    streams: &'static [InputVertexStreamLayout<'static>],
    desc: RasterPipelineCreateDesc,
}

/// Low-level renderer
pub struct Renderer {
    pub device: Arc<RenderDevice>,
    images: RwLock<ImagePool>,
    buffers: RwLock<BufferPool>,
    pipelines: RwLock<PipelinePool>,
    descriptors: RwLock<DescriptorPool>,
    staging: Mutex<Staging>,
    pipelines_to_compile: Mutex<HashMap<PipelineHandle, PipelineCompilationData>>,
    dynamic_memory: Mutex<DynamicGpuMemoryPool>,
    image_pool: TempImagePool,
}

impl Renderer {
    pub fn new(device: &Arc<RenderDevice>) -> Result<Arc<Self>, Error> {
        Ok(Arc::new(Self {
            device: device.clone(),
            staging: Mutex::new(Staging::new(device)?),
            images: Default::default(),
            buffers: Default::default(),
            pipelines: Default::default(),
            descriptors: Default::default(),
            pipelines_to_compile: Default::default(),
            dynamic_memory: Default::default(),
            image_pool: Default::default(),
        }))
    }

    pub fn import_image(&self, image: Image) -> ImageHandle {
        self.images.write().push(image)
    }

    pub fn create_image(
        &self,
        desc: ImageCreateDesc,
        data: Option<&[ImageUploadData]>,
    ) -> Result<ImageHandle, Error> {
        let image = Image::new(&self.device, desc)?;
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
        desc: ImageCreateDesc,
        data: Option<&[ImageUploadData]>,
    ) -> Result<(), Error> {
        let image = Image::new(&self.device, desc)?;
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

    pub fn upload_buffer<T: Copy>(
        &self,
        handle: BufferHandle,
        offset: usize,
        data: &[T],
    ) -> Result<(), Error> {
        let buffers = self.buffers.read();
        let buffer = buffers
            .get_cold(handle)
            .ok_or(Error::InvaludBufferHandle(handle))?;
        self.staging.lock().upload_buffer(buffer, offset, data)?;
        Ok(())
    }

    pub fn destroy_buffer(&self, handle: BufferHandle) {
        self.buffers.write().remove(handle);
    }

    pub fn map_buffer(&self, handle: BufferHandle) -> Result<NonNull<u8>, Error> {
        Ok(self
            .buffers
            .write()
            .get_cold_mut(handle)
            .ok_or(Error::InvaludBufferHandle(handle))?
            .map()?)
    }

    pub fn unmap_buffer(&self, handle: BufferHandle) {
        let mut buffers = self.buffers.write();
        if let Some(buffer) = buffers.get_cold_mut(handle) {
            buffer.unmap()
        }
    }

    pub fn create_pipeline(
        &self,
        program: &Arc<Program>,
        layout: RenderAttachmentLayoutDesc<'static>,
        streams: &'static [InputVertexStreamLayout<'static>],
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

    pub fn remove_descriptor_set(&self, handle: DescriptorHandle) {
        self.descriptors.write().remove(handle);
    }

    pub fn render<RenderCB: FnOnce(&RenderContext)>(
        &self,
        swapchain: &Swapchain,
        format: vk::Format,
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
        let context = RenderContext::new(
            self,
            &self.image_pool,
            &dynamic,
            &self.descriptors,
            target.image,
        );
        render(&context);
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
        let image = self.image_pool.get(
            &self,
            Resolution::Full,
            format,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            target.image,
        )?;
        // TODO!
        // Submit
        if let Some(semaphore) = semaphore {
            self.device.submit(
                &[command_buffer],
                vk::Fence::null(),
                &[(semaphore, vk::PipelineStageFlags2::VERTEX_INPUT)],
                &[(
                    frame.render_finished,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                )],
            )?;
        } else {
            self.device.submit(
                &[command_buffer],
                vk::Fence::null(),
                &[],
                &[(
                    frame.render_finished,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                )],
            )?;
        }
        // Present
        // self.device.present(target, image, frame)
        drop(image);
        // Cleanup
        let mut descriptors = self.descriptors.write();
        context.temp_descriptors.lock().drain(..).for_each(|x| {
            descriptors.remove(x);
        });
        self.device.end_frame(frame);

        Ok(FrameState::Rendered)
    }

    pub fn backbuffer_changed(&self) {
        self.image_pool.purge(&self);
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
                                            Error::InvaludBufferHandle(buffer.data.handle),
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
                                            Error::InvaludBufferHandle(buffer.data.handle),
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
                                            Error::InvaludBufferHandle(buffer.data.handle),
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
                                            Error::InvaludBufferHandle(buffer.data.handle),
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
            unsafe { self.device.raw.update_descriptor_sets(&writes, &[]) };
            Ok(())
        })?;

        todo!()
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
        self.pipelines
            .write()
            .drain()
            .for_each(|(pipeline, _)| unsafe { self.device.raw.destroy_pipeline(pipeline, None) })
    }
}
