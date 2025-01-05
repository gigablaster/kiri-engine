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
use std::{path::PathBuf, ptr::NonNull, sync::Arc};

use kiri_backend::{
    ash::vk::{self},
    compile_raster_pipeline, load_or_create_pipeline_cache, save_pipeline_cache, AcquiredSurface,
    Buffer, BufferCreateDesc, DescriptorSetCount, DescriptorSetLayoutDesc, Frame, Image,
    ImageCreateDesc, ImageViewDesc, InputVertexStreamLayout, Program, RasterPipelineCreateDesc,
    RenderDevice, RenderPassLayout, ShaderDesc, Swapchain,
};
use kiri_common::{GameAppConfig, Handle, HotColdPool, Pool, TempList};
use log::warn;
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{
    DescriptorSetBuilder, DescriptorSetData, DynamicGpuMemoryPool, Error, ImageUploadData,
    RenderContext, RenderResourceResolver, Staging,
};

pub type ImageHandle = Handle<Image>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type PipelineHandle = Handle<(vk::Pipeline, vk::PipelineLayout)>;
pub type DescriptorHandle = Handle<vk::DescriptorSet>;
pub type ProgramHandle = Handle<Program>;

pub(super) type ImagePool = Pool<Image>;
pub(super) type BufferPool = HotColdPool<vk::Buffer, Buffer>;
pub(super) type PipelinePool =
    HotColdPool<(vk::Pipeline, vk::PipelineLayout), PipelineCompilationData>;
pub(super) type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;
pub(super) type ProgramPool = Pool<Program>;

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

#[derive(Debug)]
pub(super) struct PipelineCompilationData {
    program: ProgramHandle,
    render_pass: &'static RenderPassLayout<'static>,
    streams: &'static [InputVertexStreamLayout<'static>],
    specialization: Vec<(u32, u32)>,
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
    programs: RwLock<ProgramPool>,
    staging: Mutex<Staging>,
    pipelines_to_compile: Mutex<Vec<PipelineHandle>>,
    dynamic_memory: Mutex<DynamicGpuMemoryPool>,
    pipeline_cache_path: Option<PathBuf>,
    pipeline_cache: vk::PipelineCache,
}

unsafe impl Sync for Renderer {}
unsafe impl Send for Renderer {}

const MAX_RESOURCE_COUNT: usize = 0xffff;
const MAX_PIPELINES: usize = 8192;
const MAX_PROGRAMS: usize = 1024;
const MAX_DESCRIPTORS: usize = 8192;

impl Renderer {
    pub fn new(device: &Arc<RenderDevice>, config: &GameAppConfig) -> Result<Arc<Self>, Error> {
        let pipeline_cache_path = config.cache();
        let pipeline_cache = pipeline_cache_path
            .clone()
            .map(|path| {
                load_or_create_pipeline_cache(device, &path).unwrap_or(vk::PipelineCache::null())
            })
            .unwrap_or(vk::PipelineCache::null());
        Ok(Arc::new(Self {
            device: device.clone(),
            staging: Mutex::new(Staging::new(device)?),
            images: RwLock::new(ImagePool::new(MAX_RESOURCE_COUNT)),
            buffers: RwLock::new(BufferPool::new(MAX_RESOURCE_COUNT)),
            pipelines: RwLock::new(PipelinePool::new(MAX_PIPELINES)),
            descriptors: RwLock::new(DescriptorPool::new(MAX_DESCRIPTORS)),
            programs: RwLock::new(ProgramPool::new(MAX_PROGRAMS)),
            pipelines_to_compile: Default::default(),
            dynamic_memory: Default::default(),
            pipeline_cache_path,
            pipeline_cache,
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
            self.staging
                .lock()
                .upload_image(image.raw, image.desc, data)?;
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
            self.staging
                .lock()
                .upload_image(image.raw, image.desc, data)?;
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
        let vk_buffer = self
            .buffers
            .read()
            .get(buffer.handle)
            .copied()
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
        program: ProgramHandle,
        render_pass: &'static RenderPassLayout<'static>,
        streams: &'static [InputVertexStreamLayout<'static>],
        specialization: &[(u32, u32)],
        desc: RasterPipelineCreateDesc,
    ) -> PipelineHandle {
        let data = PipelineCompilationData {
            program,
            render_pass,
            streams,
            specialization: specialization.to_vec(),
            desc,
        };
        let handle = self
            .pipelines
            .write()
            .push((vk::Pipeline::null(), vk::PipelineLayout::null()), data);
        self.pipelines_to_compile.lock().push(handle);
        handle
    }

    pub fn destroy_pipeline(&self, handle: PipelineHandle) {
        if let Some(((pipeline, _), _)) = self.pipelines.write().remove(handle) {
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

    pub fn import_program(&self, program: Program) -> ProgramHandle {
        self.programs.write().push(program)
    }

    pub fn create_program(
        &self,
        layout: &'static [DescriptorSetLayoutDesc<'static>],
        shaders: &[ShaderDesc],
    ) -> Result<ProgramHandle, Error> {
        let program = Program::new(&self.device, layout, shaders)?;
        Ok(self.import_program(program))
    }

    pub fn destroy_program(&self, handle: ProgramHandle) {
        self.programs.write().remove(handle);
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
        let frame = self.device.begin_frame()?;
        let mut dynamic_memory = self.dynamic_memory.lock();
        dynamic_memory.recycle();
        let dynamic = dynamic_memory.get(self)?;

        // Generate render streams
        let context = RenderContext::new(self, &dynamic, &self.descriptors, target.image);
        render(&context)?;

        // Prepare
        let staging_wait = self.staging.lock().upload()?;
        self.compile_pipelines()?;
        let images = self.images.write();
        let buffers = self.buffers.write();
        let pipelines = self.pipelines.write();

        self.update_descriptors(&frame, &images, &buffers)?;

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
        let empty_descriptor_set = frame.get_descriptor(
            &self.device.raw,
            self.device.get_or_create_layout(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                DescriptorSetLayoutDesc::default(),
            )?,
            DescriptorSetCount::default(),
        )?;
        let resolver = RenderResourceResolver {
            buffers: &buffers,
            images: &images,
            descriptors: &descriptors,
            pipelines: &pipelines,
            empty_descriptor_set,
            backbuffer: target.image,
        };
        for pass in passes {
            self.device.begin_label(command_buffer, pass.name());
            pass.dispatch(&self.device.raw, command_buffer, &resolver)?;
            self.device.end_labe(command_buffer);
        }
        unsafe {
            self.device.raw.end_command_buffer(command_buffer)?;
        }
        // Submit
        self.device.submit(
            &[command_buffer],
            frame.render_fence,
            &[
                (
                    staging_wait,
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

    fn compile_pipelines(&self) -> Result<(), Error> {
        puffin::profile_function!();
        let programs = self.programs.read();
        let pipelines = self.pipelines.upgradable_read();
        let result = Arc::new(Mutex::new(Vec::default()));
        let pipelines_ref = &pipelines;
        let programs_ref = &programs;
        rayon::scope(|s| {
            for handle in self.pipelines_to_compile.lock().drain(..) {
                let result = result.clone();
                s.spawn(move |_| {
                    let pipeline = self.compile_pipeline(handle, pipelines_ref, programs_ref);
                    result.lock().push(pipeline);
                });
            }
        });
        let mut pipelines = RwLockUpgradableReadGuard::upgrade(pipelines);
        for it in result.lock().drain(..) {
            let (handle, data) = it?;
            pipelines.replace(handle, data);
        }
        Ok(())
    }

    fn compile_pipeline<'a>(
        &self,
        handle: PipelineHandle,
        pipelines: &PipelinePool,
        programs: &ProgramPool,
    ) -> Result<(PipelineHandle, (vk::Pipeline, vk::PipelineLayout)), Error> {
        puffin::profile_function!();
        let data = pipelines
            .get_cold(handle)
            .ok_or(Error::InvalidPipelineHandle(handle))?;
        let program = programs
            .get(data.program)
            .ok_or(Error::InvalidProgramHandle(data.program))?;
        Ok((
            handle,
            (
                compile_raster_pipeline(
                    &self.device,
                    self.pipeline_cache,
                    program,
                    data.render_pass,
                    data.streams,
                    &data.specialization,
                    data.desc,
                )?,
                program.pipeline_layout,
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
        if let Some(path) = &self.pipeline_cache_path {
            save_pipeline_cache(&self.device, self.pipeline_cache, path)
                .map_err(|x| {
                    warn!("Failed to safe pipeline cache: {}", x);
                })
                .ok();
            unsafe {
                self.device
                    .raw
                    .destroy_pipeline_cache(self.pipeline_cache, None);
            }
        }
        self.pipelines
            .write()
            .drain()
            .for_each(|((pipeline, _), _)| unsafe {
                self.device.raw.destroy_pipeline(pipeline, None)
            })
    }
}
