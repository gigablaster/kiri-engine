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

use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    mem,
    path::PathBuf,
    slice,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk::{self, RenderPassBeginInfo};
use bevy_tasks::ComputeTaskPool;
use directories::ProjectDirs;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use kiri_common::{Handle, HotColdPool, SentinelPoolStrategy};
use log::error;
use parking_lot::{Mutex, RwLock};
use std::fmt::Debug;

use crate::{
    vulkan::{AcquiredSurface, Buffer, DrawStreamExecuteContext, RenderContext, MAX_ATTACHMENTS},
    Error, Instance, RenderDeviceProperties, ShaderStage, Swapchain,
};

use super::{
    create_descriptor_set_layout, drop_list::DropList, frame::Frame, image::Image,
    load_or_create_pipeline_cache, physical_device::PhysicalDevice, save_pipeline_cache,
    staging::Staging, BindGroupData, BindGroupDesc, CompilePipelineData, DescriptorSetLayout,
    DrawStream, FindSuitableDevice, GpuAllocator, GpuDescriptor, GpuDescriptorAllocator, GpuMemory,
    PhysicalDeviceType, Program, RenderPass, Surface, SwapchainImage, Uniforms, TEMP_BUFFER_SIZE,
    UNIFORM_BUFFER_SIZE,
};

pub type ImageHandle = Handle<vk::Image>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type BindGroupHandle = Handle<vk::DescriptorSet>;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlice(pub BufferHandle, pub u32);

impl BufferSlice {
    pub fn new(buffer: BufferHandle, offset: usize) -> BufferSlice {
        Self(buffer, offset as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramHandle(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RenderPassHandle(pub(crate) u32);

impl Default for RenderPassHandle {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PipelineHandle(pub(crate) u32);

impl Default for PipelineHandle {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

pub(crate) type ImagePool = HotColdPool<vk::Image, Image, SentinelPoolStrategy<vk::Image>>;
pub(crate) type BufferPool = HotColdPool<vk::Buffer, Buffer, SentinelPoolStrategy<vk::Buffer>>;
pub(crate) type ProgramPool = Vec<Program>;
pub(crate) type PipelinePool = Vec<(vk::Pipeline, vk::PipelineLayout)>;
pub(crate) type BindGroupPool =
    HotColdPool<vk::DescriptorSet, BindGroupData, SentinelPoolStrategy<vk::DescriptorSet>>;
pub(crate) type RenderPassPool = Vec<RenderPass>;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode: vk::SamplerAddressMode,
    pub anisotropy_level: u32,
}

pub enum FrameState {
    Rendered,
    NeedRecreateSwapchain,
}

pub const EMPTY_BIND_GROUP: BindGroupDesc = BindGroupDesc {
    stage: ShaderStage::Graphics,
    set: &[],
};

pub struct RenderDevice {
    pub(crate) instance: Arc<Instance>,
    pub(crate) pdevice: PhysicalDevice,
    pub(crate) device: ash::Device,
    debug: Option<ash::ext::debug_utils::Device>,
    memory_allocator: Mutex<GpuAllocator>,
    descriptor_allocator: Mutex<GpuDescriptorAllocator>,
    current_drop_list: Mutex<DropList>,
    pub(crate) images: RwLock<ImagePool>,
    pub(crate) buffers: RwLock<BufferPool>,
    pub(crate) programs: RwLock<ProgramPool>,
    pub(crate) render_passes: RwLock<RenderPassPool>,
    pub(crate) pipelines: RwLock<PipelinePool>,
    pub(crate) pipelines_to_compile: Mutex<HashMap<PipelineHandle, CompilePipelineData>>,
    frames: [Mutex<Arc<Frame>>; 2],
    pub(crate) samplers: HashMap<SamplerDesc, vk::Sampler>,
    universal_queue: Arc<Mutex<vk::Queue>>,
    transfer_queue: Arc<Mutex<vk::Queue>>,
    pub(crate) universal_queue_index: u32,
    pub(crate) transfer_queue_index: u32,
    pub(crate) staging: Mutex<Staging>,
    pub(crate) cache: vk::PipelineCache,
    pub(crate) temp_buffer_handle: BufferHandle,
    pub(crate) temp_buffer: vk::Buffer,
    pub(crate) bind_groups: Mutex<BindGroupPool>,
    pub(crate) dirty_bind_groups: Mutex<HashSet<BindGroupHandle>>,
    pub(crate) uniforms: Mutex<Uniforms>,
    pub(crate) layouts: Mutex<HashMap<BindGroupDesc<'static>, Arc<DescriptorSetLayout>>>,
    empty: Option<GpuDescriptor>,
}

impl Debug for RenderDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VkDevice({})", vk::Handle::as_raw(self.device.handle()))
    }
}

impl RenderDevice {
    pub fn new(
        instance: &Arc<Instance>,
        surface: &Surface,
        preferences: &[PhysicalDeviceType],
    ) -> Result<Arc<Self>, Error> {
        let physical_devices = instance.enumerate_physical_devices()?;
        let pdevice = physical_devices
            .find_suitable_device(surface, preferences)
            .ok_or(Error::NoSuitableDevice)?;

        if !pdevice.is_queue_flag_supported(vk::QueueFlags::GRAPHICS) {
            return Err(Error::NoSuitableDevice);
        };

        let device_extension_names = vec![ash::khr::swapchain::NAME];

        for ext in &device_extension_names {
            let ext = ext.to_str().unwrap();
            if !pdevice.is_extensions_sipported(ext) {
                return Err(Error::ExtensionNotFound(ext.into()));
            }
        }

        let device_extension_names = device_extension_names
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();

        let universal_queue_family = pdevice
            .find_queue(vk::QueueFlags::GRAPHICS, &[])
            .ok_or(Error::NoSuitableQueue)?;
        let transfer_queue_family =
            pdevice.find_queue(vk::QueueFlags::TRANSFER, &[universal_queue_family.index]);

        let universal_queue_index = universal_queue_family.index;
        let transfer_queue_index = transfer_queue_family
            .unwrap_or(universal_queue_family)
            .index;

        let queue_priorities = [1.0];
        let mut queue_info = Vec::new();
        queue_info.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(universal_queue_family.index)
                .queue_priorities(&queue_priorities),
        );
        if let Some(transfer_queue_family) = transfer_queue_family {
            queue_info.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_queue_family.index)
                    .queue_priorities(&queue_priorities),
            )
        }

        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true));
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extension_names)
            .push_next(&mut features);

        let device = unsafe {
            instance
                .raw
                .create_device(pdevice.raw, &device_create_info, None)?
        };

        let universal_queue = Arc::new(Mutex::new(unsafe {
            device.get_device_queue(universal_queue_index, 0)
        }));
        let transfer_queue = transfer_queue_family
            .map(|x| Arc::new(Mutex::new(unsafe { device.get_device_queue(x.index, 0) })))
            .unwrap_or(universal_queue.clone());

        let allocator_config = gpu_alloc::Config {
            dedicated_threshold: 64 * 1024 * 1024,
            preferred_dedicated_threshold: 16 * 1024 * 1024,
            transient_dedicated_threshold: 32 * 1024 * 1024,
            final_free_list_chunk: 1024 * 1024,
            minimal_buddy_size: 256,
            starting_free_list_chunk: 256 * 1024,
            initial_buddy_dedicated_size: 128 * 1024 * 1024,
        };
        let allocator_props =
            unsafe { device_properties(instance.get(), Instance::vulkan_version(), pdevice.raw) }?;
        let mut memory_allocator = GpuAllocator::new(allocator_config, allocator_props);
        let temp_buffer_info = vk::BufferCreateInfo::default()
            .size((TEMP_BUFFER_SIZE * 2) as _)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            );
        let temp_buffer = unsafe { device.create_buffer(&temp_buffer_info, None) }?;
        let temp_buffer_requirements =
            unsafe { device.get_buffer_memory_requirements(temp_buffer) };
        let mut temp_memory = Self::allocate_impl(
            &device,
            &mut memory_allocator,
            temp_buffer_requirements,
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS | gpu_alloc::UsageFlags::HOST_ACCESS,
            true,
        )?;
        let temp_map = unsafe {
            temp_memory.map(
                AshMemoryDevice::wrap(&device),
                0,
                (TEMP_BUFFER_SIZE * 2) as _,
            )
        }?;

        let frames = [
            Mutex::new(Arc::new(Frame::new(
                &device,
                &pdevice,
                universal_queue_index,
                temp_map,
                0,
            )?)),
            Mutex::new(Arc::new(Frame::new(
                &device,
                &pdevice,
                universal_queue_index,
                temp_map,
                TEMP_BUFFER_SIZE,
            )?)),
        ];

        let debug = instance
            .debug_utils
            .iter()
            .map(|_| ash::ext::debug_utils::Device::new(instance.get(), &device))
            .next();

        let staging = Mutex::new(Staging::new(
            &device,
            transfer_queue_index,
            universal_queue_index,
            &mut memory_allocator,
        )?);
        let samplers = Self::generate_samplers(&device);

        let cache = if let Some(path) = Self::get_pipelines_path(instance) {
            load_or_create_pipeline_cache(&device, &pdevice, &path)?
        } else {
            vk::PipelineCache::null()
        };

        let mut descriptor_allocator = GpuDescriptorAllocator::new(0);
        let mut buffers = BufferPool::default();
        let temp_buffer_handle = buffers.push(
            temp_buffer,
            Buffer {
                raw: temp_buffer,
                size: TEMP_BUFFER_SIZE * 2,
                memory: Some(temp_memory),
            },
        );

        let uniforms = Uniforms::new(
            &device,
            UNIFORM_BUFFER_SIZE as _,
            &mut memory_allocator,
            &pdevice,
        )?;
        let empty_layout = create_descriptor_set_layout(&device, &samplers, &EMPTY_BIND_GROUP)?;
        let mut layouts = HashMap::default();
        layouts.insert(EMPTY_BIND_GROUP, empty_layout.clone());
        let empty = unsafe {
            descriptor_allocator.allocate(
                AshDescriptorDevice::wrap(&device),
                &empty_layout.raw,
                DescriptorSetLayoutCreateFlags::empty(),
                &DescriptorTotalCount::default(),
                1,
            )
        }?
        .remove(0);
        Ok(Arc::new(Self {
            staging,
            instance: instance.clone(),
            samplers,
            pdevice,
            memory_allocator: Mutex::new(memory_allocator),
            descriptor_allocator: Mutex::new(descriptor_allocator),
            universal_queue,
            transfer_queue,
            frames,
            current_drop_list: Mutex::default(),
            device,
            universal_queue_index,
            transfer_queue_index,
            debug,
            images: Default::default(),
            buffers: RwLock::new(buffers),
            programs: Default::default(),
            render_passes: Default::default(),
            pipelines: Default::default(),
            pipelines_to_compile: Default::default(),
            cache,
            temp_buffer,
            temp_buffer_handle,
            bind_groups: Default::default(),
            dirty_bind_groups: Default::default(),
            layouts: Mutex::new(layouts),
            uniforms: Mutex::new(uniforms),
            empty: Some(empty),
        }))
    }

    fn get_pipelines_path(instance: &Instance) -> Option<PathBuf> {
        ProjectDirs::from(&instance.title[0], &instance.title[1], &instance.title[2])
            .map(|dirs| dirs.cache_dir().join("pipelines.bin"))
    }

    fn generate_samplers(device: &ash::Device) -> HashMap<SamplerDesc, vk::Sampler> {
        let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
        let mipmap_modes = [
            vk::SamplerMipmapMode::NEAREST,
            vk::SamplerMipmapMode::LINEAR,
        ];
        let address_modes = [
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_BORDER,
            vk::SamplerAddressMode::MIRRORED_REPEAT,
        ];
        let aniso_levels = [0, 1, 2, 3, 4];
        let mut result = HashMap::new();
        texel_filters.into_iter().for_each(|texel_filter| {
            mipmap_modes.into_iter().for_each(|mipmap_mode| {
                address_modes.into_iter().for_each(|address_mode| {
                    aniso_levels.into_iter().for_each(|aniso_level| {
                        let anisotropy = aniso_level > 0 && texel_filter == vk::Filter::LINEAR;
                        let anisotropy_level = if anisotropy { 1 << aniso_level } else { 0 };
                        let desc = SamplerDesc {
                            texel_filter,
                            mipmap_mode,
                            address_mode,
                            anisotropy_level,
                        };
                        result.entry(desc).or_insert_with(|| {
                            let sampler_create_info = vk::SamplerCreateInfo::default()
                                .mag_filter(texel_filter)
                                .min_filter(texel_filter)
                                .mipmap_mode(mipmap_mode)
                                .address_mode_u(address_mode)
                                .address_mode_v(address_mode)
                                .address_mode_w(address_mode)
                                .max_lod(vk::LOD_CLAMP_NONE)
                                .max_anisotropy(anisotropy_level as _)
                                .anisotropy_enable(anisotropy);
                            unsafe { device.create_sampler(&sampler_create_info, None).unwrap() }
                        });
                    })
                })
            })
        });

        result
    }

    pub(crate) fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        location: gpu_alloc::UsageFlags,
        dedicated: bool,
    ) -> Result<GpuMemory, Error> {
        Self::allocate_impl(
            &self.device,
            &mut self.memory_allocator.lock(),
            requirements,
            location,
            dedicated,
        )
    }

    pub(crate) fn allocate_impl(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        requirements: vk::MemoryRequirements,
        location: gpu_alloc::UsageFlags,
        dedicated: bool,
    ) -> Result<GpuMemory, Error> {
        let request = gpu_alloc::Request {
            size: requirements.size,
            align_mask: requirements.alignment,
            usage: location,
            memory_types: requirements.memory_type_bits,
        };

        Ok(if dedicated {
            unsafe {
                allocator.alloc_with_dedicated(
                    AshMemoryDevice::wrap(device),
                    request,
                    gpu_alloc::Dedicated::Required,
                )
            }
        } else {
            unsafe { allocator.alloc(AshMemoryDevice::wrap(device), request) }
        }?)
    }

    pub(crate) fn with_drop_list<CB: FnOnce(&mut DropList)>(&self, cb: CB) {
        cb(&mut self.current_drop_list.lock());
    }

    pub(crate) fn with_descriptor_allocator<
        CB: FnOnce(&mut GpuDescriptorAllocator) -> Result<(), Error>,
    >(
        &self,
        cb: CB,
    ) -> Result<(), Error> {
        cb(&mut self.descriptor_allocator.lock())
    }

    pub(crate) fn set_object_name<T: vk::Handle, S: AsRef<str>>(&self, object: T, name: S) {
        if let Some(debug_utils) = &self.debug {
            let name = CString::new(name.as_ref()).unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(object)
                .object_name(&name);
            unsafe { debug_utils.set_debug_utils_object_name(&name_info) }.unwrap();
        }
    }

    pub fn submit_graphics(
        &self,
        cb: (vk::CommandBuffer, vk::Fence),
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        triggers: &[vk::Semaphore],
    ) -> Result<(), Error> {
        self.submit(*self.universal_queue.lock(), cb.0, cb.1, wait, triggers)
    }

    pub fn submit_transfer(
        &self,
        cb: (vk::CommandBuffer, vk::Fence),
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        triggers: &[vk::Semaphore],
    ) -> Result<(), Error> {
        self.submit(*self.transfer_queue.lock(), cb.0, cb.1, wait, triggers)
    }

    fn submit(
        &self,
        queue: vk::Queue,
        cb: vk::CommandBuffer,
        fence: vk::Fence,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        triggers: &[vk::Semaphore],
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let wait_semaphores = wait.iter().map(|x| x.0).collect::<ArrayVec<_, 8>>();
        let wait_stages = wait.iter().map(|x| x.1).collect::<ArrayVec<_, 8>>();
        let command_bufers = [cb];
        let info = vk::SubmitInfo::default()
            .command_buffers(&command_bufers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(triggers)
            .wait_dst_stage_mask(&wait_stages);
        unsafe { self.device.queue_submit(queue, &[info], fence) }?;
        Ok(())
    }

    fn begin_frame(&self) -> Result<Arc<Frame>, Error> {
        puffin::profile_function!();
        let mut frame = self.frames[0].lock();
        {
            let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
            unsafe {
                self.device
                    .wait_for_fences(slice::from_ref(&frame.fence), true, u64::MAX)?
            };
            frame.reset(
                &self.device,
                &mut self.memory_allocator.lock(),
                &mut self.descriptor_allocator.lock(),
                &mut self.uniforms.lock(),
            )?;
        }
        Ok(frame.clone())
    }

    pub(crate) fn end_frame(&self, frame: Arc<Frame>) {
        drop(frame);

        let mut frame = self.frames[0].lock();
        let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
        let mut next_frame = self.frames[1].lock();
        let next_frame = Arc::get_mut(&mut next_frame).unwrap();
        frame.assign_drop_list(mem::take(&mut self.current_drop_list.lock()));
        mem::swap(frame, next_frame);
    }

    pub fn frame<F: FnOnce(&mut RenderContext) -> Result<(), Error>>(
        &self,
        target: &Swapchain,
        f: F,
    ) -> Result<FrameState, Error> {
        puffin::profile_function!();
        let compile_pipelines = Self::compile_all_pipelines(self);
        let target = match target.acquire_next_image()? {
            AcquiredSurface::NeedRecreate => return Ok(FrameState::NeedRecreateSwapchain),
            AcquiredSurface::Image(image) => image,
        };
        let frame = self.begin_frame()?;

        let passes = {
            puffin::profile_scope!("Generate frame");
            let mut context = RenderContext {
                frame: &frame,
                passes: Default::default(),
                backbuffer: target.image,
                temp_buffer: self.temp_buffer_handle,
            };
            f(&mut context)?;
            context.finish()
        };
        self.update_descriptors()?;
        bevy_tasks::block_on(compile_pipelines)?;
        {
            puffin::profile_scope!("Execute frame");
            let mut staging = self.staging.lock();
            let upload = staging.upload(self)?;
            let images = self.images.read();
            let bind_groups = self.bind_groups.lock();
            let buffers = self.buffers.read();
            let render_passes = self.render_passes.read();
            let pipelines = self.pipelines.read();

            unsafe {
                self.device
                    .begin_command_buffer(frame.cb, &vk::CommandBufferBeginInfo::default())
            }?;

            staging.execute_pending_barriers(self, frame.cb);
            for pass in passes {
                let (pass, subpass, streams, targets) = pass.consume();
                let pass = render_passes
                    .get(pass.0 as usize)
                    .ok_or(Error::InvalidRenderPassHandle(pass))?;
                let (fbo, size) = pass.framebuffer(&self.device, &images, &targets)?;
                let render_area = vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: size[0],
                        height: size[1],
                    },
                };
                let clear_values = targets
                    .iter()
                    .copied()
                    .map(|x| x.clear.into())
                    .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
                let begin_info = RenderPassBeginInfo::default()
                    .clear_values(&clear_values)
                    .framebuffer(fbo)
                    .render_pass(pass.raw)
                    .render_area(render_area);
                unsafe {
                    self.device.cmd_begin_render_pass(
                        frame.cb,
                        &begin_info,
                        vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
                    );
                }

                let executed = ComputeTaskPool::get().scope(|s| {
                    streams.into_iter().for_each(|stream| {
                        s.spawn(Self::execute_single_stream(
                            stream,
                            DrawStreamExecuteContext {
                                device: &self.device,
                                frame: &frame,
                                pipelines: &pipelines,
                                bind_groups: &bind_groups,
                                buffers: &buffers,
                                empty: *self.empty.as_ref().unwrap().raw(),
                                fbo,
                                pass: pass.raw,
                                subpass,
                                render_area,
                            },
                        ))
                    })
                });
                let mut cbs = Vec::with_capacity(executed.len());
                for cb in executed {
                    cbs.push(cb?);
                }
                unsafe {
                    if !cbs.is_empty() {
                        self.device.cmd_execute_commands(frame.cb, &cbs);
                    }
                    self.device.cmd_end_render_pass(frame.cb);
                }
            }
            unsafe { self.device.end_command_buffer(frame.cb) }?;
            let wait = [
                upload,
                (
                    target.acquire_semaphore,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ),
            ];
            let trigger = [target.rendering_finished];
            self.submit_graphics((frame.cb, frame.fence), &wait, &trigger)?;
            self.end_frame(frame);
        }
        self.present(target);
        Ok(FrameState::Rendered)
    }

    #[allow(clippy::needless_lifetimes)]
    async fn execute_single_stream<'a>(
        stream: DrawStream,
        context: DrawStreamExecuteContext<'a>,
    ) -> Result<vk::CommandBuffer, Error> {
        stream.execute(context)
    }

    fn present(&self, image: SwapchainImage) {
        puffin::profile_function!();
        let binding = image.swapchain.raw;
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&image.rendering_finished))
            .swapchains(slice::from_ref(&binding))
            .image_indices(slice::from_ref(&image.image_index));

        match unsafe {
            image
                .swapchain
                .loader()
                .queue_present(*self.universal_queue.lock(), &present_info)
        } {
            Ok(_) => (),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {}
            Err(err) => panic!("Can't present image: {}", err),
        }
    }

    pub fn properties(&self) -> RenderDeviceProperties {
        RenderDeviceProperties {
            buffer_allocation_granularity: self
                .pdevice
                .properties
                .limits
                .min_storage_buffer_offset_alignment
                as usize,
        }
    }
}

impl Drop for RenderDevice {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle() }.expect("device_wait_idle isn't supposed to fail");
        self.staging.lock().free(self);
        let mut memory_allocator = self.memory_allocator.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        let mut drop_list = self.current_drop_list.lock();
        let mut uniforms = self.uniforms.lock();
        if let Some(empty) = self.empty.take() {
            drop_list.drop_descriptor(empty);
        }
        self.images.write().drain().for_each(|(_, image)| {
            image.free(&mut drop_list);
        });
        self.buffers
            .write()
            .drain()
            .for_each(|(_, buffer)| buffer.free(&mut drop_list));
        drop_list.purge(
            &self.device,
            &mut memory_allocator,
            &mut descriptor_allocator,
            &mut uniforms,
        );
        self.frames.iter().for_each(|frame| {
            Arc::get_mut(&mut frame.lock())
                .expect("Nothing should hold a frame at point when we destroy rendering context")
                .reset(
                    &self.device,
                    &mut memory_allocator,
                    &mut descriptor_allocator,
                    &mut uniforms,
                )
                .unwrap();
        });
        self.pipelines
            .write()
            .drain(..)
            .for_each(|(pipeline, _)| unsafe {
                self.device.destroy_pipeline(pipeline, None);
            });
        self.programs
            .write()
            .drain(..)
            .for_each(|x| x.free(&self.device));
        self.frames.iter_mut().for_each(|x| {
            Arc::get_mut(&mut x.lock())
                .expect("Nothing should hold frame at this point")
                .free(
                    &self.device,
                    &mut memory_allocator,
                    &mut descriptor_allocator,
                    &mut uniforms,
                )
        });
        uniforms.free(&self.device, &mut memory_allocator);
        self.samplers
            .drain()
            .for_each(|(_, sampler)| unsafe { self.device.destroy_sampler(sampler, None) });
        self.render_passes
            .write()
            .drain(..)
            .for_each(|x| x.free(&self.device));
        if self.cache != vk::PipelineCache::null() {
            if let Some(path) = Self::get_pipelines_path(&self.instance) {
                if let Err(err) = save_pipeline_cache(&self.device, &self.pdevice, self.cache, path)
                {
                    error!("Failed to save pipeline cache: {}", err);
                }
            }
            unsafe { self.device.destroy_pipeline_cache(self.cache, None) };
        }
        unsafe {
            descriptor_allocator.free(
                AshDescriptorDevice::wrap(&self.device),
                self.bind_groups
                    .lock()
                    .drain()
                    .filter_map(|(_, data)| data.set),
            );
            descriptor_allocator.cleanup(AshDescriptorDevice::wrap(&self.device));
            memory_allocator.cleanup(AshMemoryDevice::wrap(&self.device));
        };
        self.layouts.lock().drain().for_each(|(_, mut x)| {
            Arc::get_mut(&mut x).unwrap().free(&self.device);
        });
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

pub(crate) struct PipelineCompilationContext<'a> {
    pub device: &'a ash::Device,
    pub programs: &'a ProgramPool,
    pub render_passes: &'a RenderPassPool,
}

impl<'a> PipelineCompilationContext<'a> {
    pub fn resolve_program(&self, handle: ProgramHandle) -> Option<&Program> {
        self.programs.get(handle.0 as usize)
    }

    pub fn resolve_render_pass(&self, handle: RenderPassHandle) -> Option<&RenderPass> {
        self.render_passes.get(handle.0 as usize)
    }
}
