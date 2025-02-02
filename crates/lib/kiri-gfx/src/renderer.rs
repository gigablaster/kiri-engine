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

use std::{mem, path::PathBuf, ptr::NonNull, slice, sync::Arc};

use kiri_backend::{
    ash::{
        self,
        vk::{self},
    },
    compile_raster_pipeline, load_or_create_pipeline_cache, save_pipeline_cache, AcquiredSurface,
    Buffer, BufferCreateDesc, DescriptorSetLayoutDesc, DescriptorTotalCount, GpuDescriptor, Image,
    ImageCreateDesc, ImageDesc, ImageUploadData, ImageViewDesc, InputVertexStreamLayout, Program,
    RasterPipelineCreateDesc, RenderDevice, RenderPassLayout, ShaderDesc, Swapchain,
    EMPTY_DESCRIPTOR_SET,
};
use kiri_common::{BlockAllocator, GameAppConfig, Handle, HotColdPool, Pool, TempList};
use log::debug;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rayon::iter::{ParallelDrainRange, ParallelIterator};

use crate::{DynamicGpuMemory, DynamicGpuMemoryPool, DynamicWriter, Error};

pub type ImageHandle = Handle<Image>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type DescriptorHandle = Handle<vk::DescriptorSet>;
#[derive(Debug, Clone, Copy)]
pub struct ProgramHandle(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterPipelineHandle(u32);

type BufferPool = HotColdPool<vk::Buffer, Buffer>;
type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;
type RasterPipelinePool = Vec<(vk::Pipeline, vk::PipelineLayout)>;
type ProgramPool = Vec<Program>;
type ImagePool = Pool<Image>;

impl From<RasterPipelineHandle> for u32 {
    fn from(value: RasterPipelineHandle) -> Self {
        value.0
    }
}

impl From<u32> for RasterPipelineHandle {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Default for RasterPipelineHandle {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameState {
    Rendered,
    NeedRecreateSwapchain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlice {
    pub handle: BufferHandle,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferPointer {
    pub handle: BufferHandle,
    pub offset: u32,
}

impl Default for BufferSlice {
    fn default() -> Self {
        Self {
            handle: Handle::default(),
            offset: u32::MAX,
            size: u32::MAX,
        }
    }
}

impl BufferSlice {
    pub fn new(handle: BufferHandle, offset: usize, size: usize) -> BufferSlice {
        Self {
            handle,
            offset: offset as u32,
            size: size as u32,
        }
    }
}

impl BufferPointer {
    pub fn new(handle: BufferHandle, offset: usize) -> BufferPointer {
        Self {
            handle,
            offset: offset as u32,
        }
    }
}

impl Default for BufferPointer {
    fn default() -> Self {
        Self {
            handle: Handle::default(),
            offset: u32::MAX,
        }
    }
}

impl From<BufferSlice> for BufferPointer {
    fn from(value: BufferSlice) -> Self {
        Self {
            handle: value.handle,
            offset: value.offset,
        }
    }
}

#[derive(Debug)]
pub struct RasterPipelineDesc {
    pub program: ProgramHandle,
    pub pass_layout: &'static RenderPassLayout<'static>,
    pub input_layout: &'static [InputVertexStreamLayout<'static>],
    pub specialization: Vec<(u32, u32)>,
    pub desc: RasterPipelineCreateDesc,
}

pub struct RenderContext<'a> {
    renderer: &'a Renderer,
    dynamic: &'a DynamicGpuMemory,
    passes: Mutex<Vec<Box<dyn PassDispatcher>>>,
    temp_descriptors: Mutex<Vec<DescriptorHandle>>,
    pub backbuffer: &'a Image,
}

unsafe impl Sync for RenderContext<'_> {}
unsafe impl Send for RenderContext<'_> {}

#[derive(Debug)]
pub struct RenderResourceResolver<'a> {
    buffers: &'a BufferPool,
    images: &'a ImagePool,
    raster_pipelines: &'a RasterPipelinePool,
    descriptor_resolver: DescriptorResolver<'a>,
    pub empty_descriptor_set: vk::DescriptorSet,
    pub backbuffer: &'a Image,
}

impl<'a> RenderResourceResolver<'a> {
    fn new(
        backbuffer: &'a Image,
        buffers: &'a BufferPool,
        images: &'a ImagePool,
        raster_pipelines: &'a RasterPipelinePool,
        descriptor_resolver: DescriptorResolver<'a>,
        empty_descriptor_set: vk::DescriptorSet,
    ) -> Self {
        Self {
            buffers,
            images,
            raster_pipelines,
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
        self.raster_pipelines
            .get(handle.0 as usize)
            .copied()
            .ok_or(Error::InvalidRasterPipelineHandle(handle))
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
        backbuffer: &'a Image,
    ) -> Self {
        Self {
            renderer,
            dynamic,
            passes: Default::default(),
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

    pub fn submit(&self, pass: Box<dyn PassDispatcher>) {
        self.passes.lock().push(pass);
    }

    pub fn create_temp_descriptor(
        &self,
        builder: DescriptorSetCreateDesc,
    ) -> Result<DescriptorHandle, Error> {
        let handle = self
            .renderer
            .with_descriptors()
            .create_descriptor(builder)?;
        self.temp_descriptors.lock().push(handle);
        Ok(handle)
    }

    fn consume(self) -> (Vec<DescriptorHandle>, Vec<Box<dyn PassDispatcher>>) {
        (self.temp_descriptors.into_inner(), self.passes.into_inner())
    }
}

#[derive(Debug, Clone, Copy)]
struct Binding<T: Copy> {
    pub slot: u32,
    pub element: u32,
    pub data: T,
}

#[derive(Debug, Clone, Copy)]
struct ImageBindingData {
    image: ImageHandle,
    desc: ImageViewDesc,
    ty: vk::DescriptorType,
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
    uniforms: Vec<Binding<BufferSlice>>,
    storages: Vec<Binding<BufferSlice>>,
    dynamic_uniforms: Vec<Binding<DynamicBufferBindingData>>,
    dynamic_storages: Vec<Binding<DynamicBufferBindingData>>,
    name: Option<String>,
}

#[derive(Debug, Default)]
pub struct DescriptorSetCreateDesc<'a> {
    pub layout: DescriptorSetLayoutDesc<'static>,
    pub stages: vk::ShaderStageFlags,
    pub images: &'a [ImageHandle],
    pub unifoms: &'a [BufferSlice],
    pub storages: &'a [BufferSlice],
    pub dynamic_uniforms: &'a [(BufferHandle, usize)],
    pub dynamic_storage_buffers: &'a [(BufferHandle, usize)],
    pub name: Option<&'a str>,
}

impl DescriptorSetCreateDesc<'_> {
    fn build(self, device: &RenderDevice) -> Result<DescriptorSetData, Error> {
        let images = self
            .layout
            .by_types(&[
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ])
            .zip(self.images)
            .map(|((slot, desc), image)| Binding {
                slot: slot as u32,
                element: 0,
                data: ImageBindingData {
                    image: *image,
                    desc: ImageViewDesc::color(),
                    ty: desc.ty,
                },
            })
            .collect();
        let uniforms = self
            .layout
            .by_types(&[vk::DescriptorType::UNIFORM_BUFFER])
            .zip(self.unifoms)
            .map(|((slot, _), buffer)| Binding {
                slot: slot as u32,
                element: 0,
                data: *buffer,
            })
            .collect();
        let storages = self
            .layout
            .by_types(&[vk::DescriptorType::STORAGE_BUFFER])
            .zip(self.storages)
            .map(|((slot, _), buffer)| Binding {
                slot: slot as u32,
                element: 0,
                data: *buffer,
            })
            .collect();
        let dynamic_uniforms = self
            .layout
            .by_types(&[vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC])
            .zip(self.dynamic_uniforms)
            .map(|((slot, _), (buffer, size))| Binding {
                slot: slot as u32,
                element: 0,
                data: DynamicBufferBindingData {
                    buffer: *buffer,
                    size: *size as u32,
                },
            })
            .collect();
        let dynamic_storages = self
            .layout
            .by_types(&[vk::DescriptorType::STORAGE_BUFFER_DYNAMIC])
            .zip(self.dynamic_storage_buffers)
            .map(|((slot, _), (buffer, size))| Binding {
                slot: slot as u32,
                element: 0,
                data: DynamicBufferBindingData {
                    buffer: *buffer,
                    size: *size as u32,
                },
            })
            .collect();
        Ok(DescriptorSetData {
            descriptor: None,
            count: self.layout.get_descriptor_count(),
            layout: device.get_or_create_layout(self.stages, self.layout)?,
            images,
            uniforms,
            storages,
            dynamic_uniforms,
            dynamic_storages,
            name: self.name.map(|x| x.to_owned()),
        })
    }
}

const MAX_DESCRIPTORS: usize = 16384;

#[derive(Debug)]
struct DescriptorResolver<'a> {
    descriptors: &'a DescriptorPool,
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
        builder: DescriptorSetCreateDesc,
    ) -> Result<DescriptorHandle, Error> {
        let data = builder.build(self.device)?;
        let handle = self.descriptors.push(vk::DescriptorSet::null(), data);
        self.dirty.push(handle);
        Ok(handle)
    }

    pub fn destroy_descriptor(&mut self, handle: DescriptorHandle) {
        self.to_destroy.push(handle);
    }
}

const UNIFORM_PAGE_SIZE: usize = 64536;
const MAX_UNIFOMR_SIZE: usize = 16384;

#[derive(Debug)]
struct UniformPage {
    handle: BufferHandle,
    size_range: (usize, usize),
    allocator: BlockAllocator,
}

type RastePipelineData = (
    Vec<(vk::Pipeline, vk::PipelineLayout)>,
    Vec<RasterPipelineDesc>,
);
/// Low-level renderer
#[derive(Debug)]
pub struct Renderer {
    pub device: Arc<RenderDevice>,
    buffers: RwLock<BufferPool>,
    images: RwLock<ImagePool>,
    raster_programs: RwLock<ProgramPool>,
    raster_pipelines: Mutex<RastePipelineData>,
    dynamic_memory: Mutex<DynamicGpuMemoryPool>,
    descriptors: RwLock<DescriptorPool>,
    dirty_descriptors: Mutex<Vec<DescriptorHandle>>,
    descriptors_to_destroy: Mutex<Vec<DescriptorHandle>>,
    buffers_to_destroy: Mutex<Vec<BufferHandle>>,
    images_to_destroy: Mutex<Vec<ImageHandle>>,
    uniforms: Mutex<Vec<UniformPage>>,
    cache_path: Option<PathBuf>,
    cache: vk::PipelineCache,
}

unsafe impl Sync for Renderer {}
unsafe impl Send for Renderer {}

const MAX_RESOURCE_COUNT: usize = 64536;

impl Renderer {
    pub fn new(device: Arc<RenderDevice>, config: &GameAppConfig) -> Result<Arc<Self>, Error> {
        let cache_path = config.cache();
        let cache = cache_path
            .iter()
            .map(|path| load_or_create_pipeline_cache(&device, path).unwrap_or_default())
            .next()
            .unwrap_or_default();
        Ok(Arc::new(Self {
            buffers: RwLock::new(BufferPool::new(MAX_RESOURCE_COUNT)),
            images: RwLock::new(ImagePool::new(MAX_RESOURCE_COUNT)),
            dynamic_memory: Default::default(),
            raster_pipelines: Default::default(),
            raster_programs: Default::default(),
            descriptors: RwLock::new(HotColdPool::new(MAX_DESCRIPTORS)),
            dirty_descriptors: Default::default(),
            descriptors_to_destroy: Default::default(),
            buffers_to_destroy: Default::default(),
            images_to_destroy: Default::default(),
            uniforms: Default::default(),
            device,
            cache_path,
            cache,
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
            .upload(target.offset as usize, data)?;
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

    pub fn update_image(&self, handle: ImageHandle, image: Image) -> Result<Image, Error> {
        self.invalidate_descriptors_with_image(handle);
        self.images
            .write()
            .replace(handle, image)
            .ok_or(Error::InvalidImageHandle(handle))
    }

    pub fn destroy_image(&self, handle: ImageHandle) {
        self.images_to_destroy.lock().push(handle);
    }

    pub fn create_program(
        &self,
        layout: &'static [DescriptorSetLayoutDesc<'static>],
        shaders: &[ShaderDesc],
    ) -> Result<ProgramHandle, Error> {
        let program = Program::new(self.device.clone(), layout, shaders)?;
        let mut programs = self.raster_programs.write();
        let index = programs.len() as u32;
        programs.push(program);
        Ok(ProgramHandle(index))
    }

    pub fn create_raster_pipeline(&self, desc: RasterPipelineDesc) -> RasterPipelineHandle {
        let mut pipelines = self.raster_pipelines.lock();
        debug_assert_eq!(pipelines.0.len(), pipelines.1.len());
        let index = pipelines.0.len() as u32;
        pipelines
            .0
            .push((vk::Pipeline::null(), vk::PipelineLayout::null()));
        pipelines.1.push(desc);
        RasterPipelineHandle(index)
    }

    pub fn with_descriptors(&self) -> DescriptorUpdateContext {
        let descriptors = self.descriptors.write();
        let dirty = self.dirty_descriptors.lock();
        let to_destroy = self.descriptors_to_destroy.lock();
        DescriptorUpdateContext {
            device: &self.device,
            descriptors,
            dirty,
            to_destroy,
        }
    }

    pub fn allocate_uniform<T: Copy>(&self, data: T) -> Result<BufferSlice, Error> {
        debug_assert!(mem::size_of::<T>() < MAX_UNIFOMR_SIZE);
        let mut pages = self.uniforms.lock();
        let item_size = mem::size_of::<T>();
        let upper_bound = item_size.next_power_of_two();
        let lower_bound = item_size.next_power_of_two() / 2 - 1;
        let allocated = pages
            .iter_mut()
            .find_map(|x| {
                if x.size_range.0 > lower_bound && x.size_range.1 <= upper_bound {
                    x.allocator
                        .allocate()
                        .map(|offset| BufferSlice::new(x.handle, offset, item_size as _))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                // Fixme:: unwrap
                let buffer = Buffer::new(
                    &self.device,
                    BufferCreateDesc::gpu(UNIFORM_PAGE_SIZE as _)
                        .transfer_destination()
                        .uniform_buffer(),
                )
                .unwrap();

                let mut allocator = BlockAllocator::new(
                    UNIFORM_PAGE_SIZE as _,
                    (UNIFORM_PAGE_SIZE / upper_bound) as _,
                );
                let offset = allocator.allocate().unwrap();
                let handle = self.register_buffer(buffer);
                pages.push(UniformPage {
                    handle,
                    allocator,
                    size_range: (lower_bound, upper_bound),
                });
                BufferSlice::new(handle, offset, item_size as _)
            });
        drop(pages);
        self.upload_buffer_data(allocated.into(), &[data])?;
        Ok(allocated)
    }

    pub fn free_uniform(&self, uniform: BufferSlice) {
        let mut pages = self.uniforms.lock();
        pages
            .iter_mut()
            .find(|page| page.handle == uniform.handle)
            .iter_mut()
            .for_each(|page| page.allocator.dealloc(uniform.offset as usize));
    }

    fn compile_pipelines(&self) -> Result<(), Error> {
        let mut pipelines = self.raster_pipelines.lock();
        debug_assert_eq!(pipelines.0.len(), pipelines.1.len());
        let mut to_compile = Vec::new();
        for index in 0..pipelines.0.len() {
            if pipelines.0[index].0 == vk::Pipeline::null() {
                to_compile.push((index, &pipelines.1[index]));
            }
        }
        let compiled = to_compile
            .par_drain(..)
            .map(|(index, desc)| self.compile_pipeline(index, desc))
            .collect::<Vec<_>>();
        for result in compiled {
            let (index, pipeline, pipeline_layout) = result?;
            pipelines.0[index] = (pipeline, pipeline_layout);
        }
        Ok(())
    }

    fn compile_pipeline(
        &self,
        index: usize,
        desc: &RasterPipelineDesc,
    ) -> Result<(usize, vk::Pipeline, vk::PipelineLayout), Error> {
        let (pipeline, pipeline_layout) = compile_raster_pipeline(
            &self.device,
            self.cache,
            self.raster_programs
                .read()
                .get(desc.program.0 as usize)
                .ok_or(Error::InvalidRasterProgramHandle(desc.program))?,
            desc.pass_layout,
            desc.input_layout,
            &desc.specialization,
            desc.desc,
            None,
        )?;
        Ok((index, pipeline, pipeline_layout))
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
                                .descriptor_type(image.data.ty)
                                .dst_array_element(image.element)
                                .dst_binding(image.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process uniform buffers
                    for buffer in &data.uniforms {
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
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process storage buffers
                    for buffer in &data.storages {
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
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process dynamic uniform buffers
                    for buffer in &data.dynamic_uniforms {
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
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process dynamic storage buffers
                    for buffer in &data.dynamic_storages {
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
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
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
            if let Some(ds) = descriptors.get_cold(handle) {
                if let Some(ds) = &ds.descriptor {
                    descriptors.replace(handle, *ds.raw());
                }
            }
        }
        dirty.clear();
        self.device.drop_descriptors(drop_list);
        Ok(())
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
        let mut descriptors_to_drop = Vec::new();
        dynamic_memory.recycle();
        let dynamic = dynamic_memory.take(self)?;
        drop(dynamic_memory);

        // Generate render streams
        let mut context = RenderContext::new(self, &dynamic, target.image);
        render(&mut context)?;
        let (mut temp_descriptors, passes) = context.consume();
        descriptors_to_drop.append(&mut temp_descriptors);
        let mut descriptors = self.descriptors.write();
        let mut buffers = self.buffers.write();
        let mut images = self.images.write();

        self.update_descriptors(&mut descriptors, &buffers, &images)?;

        // Prepare
        self.compile_pipelines()?;

        let raster_pipelines = self.raster_pipelines.lock();
        let mut buffers_to_destroy = self.buffers_to_destroy.lock();
        let mut images_to_destroy = self.images_to_destroy.lock();
        descriptors_to_drop.append(&mut self.descriptors_to_destroy.lock());

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
            &raster_pipelines.0,
            DescriptorResolver {
                descriptors: &descriptors,
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
        let mut descriptors_to_drop = descriptors_to_drop
            .drain(..)
            .filter_map(|handle| descriptors.remove(handle))
            .filter_map(|(_, data)| data.descriptor)
            .collect::<Vec<_>>();
        descriptors_to_drop.push(empty_descriptor_set);
        self.device.drop_descriptors(descriptors_to_drop);
        buffers_to_destroy.drain(..).for_each(|handle| {
            buffers.remove(handle);
        });
        images_to_destroy.drain(..).for_each(|handle| {
            images.remove(handle);
        });
        drop(raster_pipelines);
        drop(descriptors);
        drop(buffers);
        drop(images);
        drop(buffers_to_destroy);
        drop(images_to_destroy);
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

    fn invalidate_descriptors_with_image(&self, image: ImageHandle) {
        let mut dirty = self.dirty_descriptors.lock();
        let descriptors = self.descriptors.read();
        descriptors.enumerate().for_each(|(handle, _, data)| {
            if data.images.iter().any(|x| x.data.image == image) {
                dirty.push(handle);
            }
        });
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe { self.device.raw.device_wait_idle() }.ok();
        self.raster_pipelines
            .lock()
            .0
            .drain(..)
            .for_each(|(pipeline, _)| unsafe {
                self.device.raw.destroy_pipeline(pipeline, None);
            });
        let descriptors = self
            .descriptors
            .write()
            .drain()
            .filter_map(|(_, data)| data.descriptor)
            .collect::<Vec<_>>();
        self.device.drop_descriptors(descriptors);
        if let Some(path) = &self.cache_path {
            save_pipeline_cache(&self.device, self.cache, path).ok();
        }
        if self.cache != vk::PipelineCache::null() {
            unsafe {
                self.device.raw.destroy_pipeline_cache(self.cache, None);
            }
        }
    }
}
