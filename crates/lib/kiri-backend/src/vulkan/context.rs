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
    u32, u64,
};

use arrayvec::ArrayVec;
use ash::vk::{self, DescriptorBufferInfo, DescriptorImageInfo, WriteDescriptorSet};
use directories::ProjectDirs;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use kiri_common::{Handle, HotColdPool, SentinelPoolStrategy, TempList, MAX_POOL_INDEX};
use log::error;
use parking_lot::{Mutex, RwLock};
use std::fmt::Debug;

use crate::{
    vulkan::{
        barrier::{image_barrier, ImageBarrier, ImageBarrierType},
        AcquiredSurface, Buffer, DrawStreamExecuteContext, FrameRecorder, MAX_ATTACHMENTS,
        MAX_COLOR_ATTACHMENTS,
    },
    Error, Instance, RasterPipelineCreateDesc, Swapchain,
};

use super::{
    create_descriptor_set_layout, drop_list::DropList, frame::Frame, image::Image,
    load_or_create_pipeline_cache, physical_device::PhysicalDevice, save_pipeline_cache,
    staging::Staging, DescriptorBindingDesc, DescriptorSetLayoutDesc, GpuAllocator, GpuMemory,
    Program, RenderPassLayout, SwapchainImage,
};

pub type ImageHandle = Handle<vk::ImageView>;
pub type BufferHandle = Handle<vk::DeviceAddress>;

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct BufferSlice(pub BufferHandle, pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramHandle(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PipelineHandle(pub(crate) u32);

impl Default for PipelineHandle {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

pub(crate) type ImagePool = HotColdPool<vk::ImageView, Image, SentinelPoolStrategy<vk::ImageView>>;
pub(crate) type BufferPool =
    HotColdPool<vk::DeviceAddress, Buffer, SentinelPoolStrategy<vk::DeviceAddress>>;
pub(crate) type ProgramPool = Vec<Program>;
pub(crate) type PipelinePool = Vec<(vk::Pipeline, vk::PipelineLayout)>;

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

pub struct RenderContext<'game> {
    pub(crate) instance: &'game Instance,
    pub(crate) pdevice: PhysicalDevice,
    pub(crate) device: ash::Device,
    debug: Option<ash::ext::debug_utils::Device>,
    memory_allocator: Mutex<GpuAllocator>,
    current_drop_list: Mutex<DropList>,
    pub(crate) images: RwLock<ImagePool>,
    pub(crate) buffers: RwLock<BufferPool>,
    pub(crate) programs: RwLock<ProgramPool>,
    pub(crate) pipelines: RwLock<PipelinePool>,
    pub(crate) pipelines_to_compile: Mutex<
        HashMap<
            PipelineHandle,
            (
                ProgramHandle,
                RenderPassLayout<'static>,
                RasterPipelineCreateDesc,
            ),
        >,
    >,
    frames: [Mutex<Arc<Frame>>; 2],
    pub(crate) samplers: HashMap<SamplerDesc, vk::Sampler>,
    universal_queue: Arc<Mutex<vk::Queue>>,
    transfer_queue: Arc<Mutex<vk::Queue>>,
    pub(crate) universal_queue_index: u32,
    pub(crate) transfer_queue_index: u32,
    pub(crate) staging: Mutex<Staging>,
    bindless_layout: vk::DescriptorSetLayout,
    bindless_pool: vk::DescriptorPool,
    bindless_ds: vk::DescriptorSet,
    sampler_layout: vk::DescriptorSetLayout,
    sampler_pool: vk::DescriptorPool,
    sampler_ds: vk::DescriptorSet,
    pub(crate) sampled_images_to_update: Mutex<HashSet<ImageHandle>>,
    pub(crate) storage_images_to_update: Mutex<HashSet<ImageHandle>>,
    pub(crate) storage_buffers_to_update: Mutex<HashSet<BufferHandle>>,
    pub(crate) cache: vk::PipelineCache,
}

impl<'game> Debug for RenderContext<'game> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VkDevice({})", vk::Handle::as_raw(self.device.handle()))
    }
}

const SAMPLED_IMAGES_SLOT: u32 = 0;
const STORAGE_IMAGES_SLOT: u32 = 1;
const STORAGE_BUFFERS_SLOT: u32 = 2;

const BINDLESS_SET: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    bindless: true,
    stage: vk::ShaderStageFlags::ALL,
    set: &[
        DescriptorBindingDesc {
            name: "sampled_images",
            slot: SAMPLED_IMAGES_SLOT,
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            count: MAX_POOL_INDEX,
        },
        DescriptorBindingDesc {
            name: "storage_images",
            slot: STORAGE_IMAGES_SLOT,
            ty: vk::DescriptorType::STORAGE_IMAGE,
            count: MAX_POOL_INDEX,
        },
        DescriptorBindingDesc {
            name: "storage_buffers",
            slot: STORAGE_BUFFERS_SLOT,
            ty: vk::DescriptorType::STORAGE_BUFFER,
            count: MAX_POOL_INDEX,
        },
    ],
};

const SAMPLER_SET: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    bindless: false,
    stage: vk::ShaderStageFlags::FRAGMENT,
    set: &[
        DescriptorBindingDesc {
            name: "sampler_lr",
            slot: 0,
            ty: vk::DescriptorType::SAMPLER,
            count: 1,
        },
        DescriptorBindingDesc {
            name: "sampler_lb",
            slot: 0,
            ty: vk::DescriptorType::SAMPLER,
            count: 1,
        },
        DescriptorBindingDesc {
            name: "sampler_lm",
            slot: 0,
            ty: vk::DescriptorType::SAMPLER,
            count: 1,
        },
        DescriptorBindingDesc {
            name: "sampler_nr",
            slot: 0,
            ty: vk::DescriptorType::SAMPLER,
            count: 1,
        },
        DescriptorBindingDesc {
            name: "sampler_nb",
            slot: 0,
            ty: vk::DescriptorType::SAMPLER,
            count: 1,
        },
        DescriptorBindingDesc {
            name: "sampler_nm",
            slot: 0,
            ty: vk::DescriptorType::SAMPLER,
            count: 1,
        },
    ],
};

impl<'game> RenderContext<'game> {
    pub(crate) fn new(instance: &'game Instance, pdevice: PhysicalDevice) -> Result<Self, Error> {
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

        let mut dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut synchronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true);
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true);
        let mut buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut dynamic_rendering)
            .push_next(&mut synchronization2)
            .push_next(&mut descriptor_indexing)
            .push_next(&mut maintenance4)
            .push_next(&mut buffer_device_address)
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

        let frames = [
            Mutex::new(Arc::new(Frame::new(
                &device,
                &pdevice,
                &mut memory_allocator,
                universal_queue_index,
            )?)),
            Mutex::new(Arc::new(Frame::new(
                &device,
                &pdevice,
                &mut memory_allocator,
                universal_queue_index,
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
        let bindless_layout = create_descriptor_set_layout(&device, &samplers, &BINDLESS_SET)?;
        let sampler_layout = create_descriptor_set_layout(&device, &samplers, &SAMPLER_SET)?;

        let sizes = BINDLESS_SET.to_pool_size(1);
        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&sizes);
        let bindless_pool = unsafe { device.create_descriptor_pool(&pool_create_info, None) }?;
        let sizes = SAMPLER_SET.to_pool_size(1);
        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        let sampler_pool = unsafe { device.create_descriptor_pool(&pool_create_info, None) }?;

        let layouts = [bindless_layout];
        let mut allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(bindless_pool)
            .set_layouts(&layouts);
        allocate_info.descriptor_set_count = 1;
        let bindless_ds = unsafe { device.allocate_descriptor_sets(&allocate_info) }?.remove(0);
        let layouts = [sampler_layout];
        let mut allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(sampler_pool)
            .set_layouts(&layouts);
        allocate_info.descriptor_set_count = 1;
        let sampler_ds = unsafe { device.allocate_descriptor_sets(&allocate_info) }?.remove(0);

        let cache = if let Some(path) = Self::get_pipelines_path(&instance) {
            load_or_create_pipeline_cache(&device, &pdevice, &path)?
        } else {
            vk::PipelineCache::null()
        };
        Ok(Self {
            staging,
            instance,
            samplers,
            pdevice,
            memory_allocator: Mutex::new(memory_allocator),
            universal_queue,
            transfer_queue,
            frames,
            current_drop_list: Mutex::default(),
            device,
            universal_queue_index,
            transfer_queue_index,
            debug,
            images: Default::default(),
            buffers: Default::default(),
            programs: Default::default(),
            pipelines: Default::default(),
            pipelines_to_compile: Default::default(),
            bindless_layout,
            bindless_pool,
            bindless_ds,
            sampler_layout,
            sampler_pool,
            sampler_ds,
            sampled_images_to_update: Default::default(),
            storage_images_to_update: Default::default(),
            storage_buffers_to_update: Default::default(),
            cache,
        })
    }

    fn get_pipelines_path(instance: &Instance) -> Option<PathBuf> {
        if let Some(dirs) =
            ProjectDirs::from(&instance.title[0], &instance.title[1], &instance.title[2])
        {
            Some(dirs.cache_dir().join("pipelines.bin"))
        } else {
            None
        }
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

    pub fn with_drop_list<CB: FnOnce(&mut DropList)>(&self, cb: CB) {
        cb(&mut self.current_drop_list.lock());
    }

    pub fn set_object_name<T: vk::Handle, S: AsRef<str>>(&self, object: T, name: S) {
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
        wait: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        triggers: &[(vk::Semaphore, vk::PipelineStageFlags2)],
    ) -> Result<(), Error> {
        self.submit(*self.universal_queue.lock(), cb.0, cb.1, wait, triggers)
    }

    pub fn submit_transfer(
        &self,
        cb: (vk::CommandBuffer, vk::Fence),
        wait: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        triggers: &[(vk::Semaphore, vk::PipelineStageFlags2)],
    ) -> Result<(), Error> {
        self.submit(*self.transfer_queue.lock(), cb.0, cb.1, wait, triggers)
    }

    fn submit(
        &self,
        queue: vk::Queue,
        cb: vk::CommandBuffer,
        fence: vk::Fence,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        triggers: &[(vk::Semaphore, vk::PipelineStageFlags2)],
    ) -> Result<(), Error> {
        puffin::profile_function!();
        // let wait_semaphores = wait.iter().map(|x| x.0).collect::<ArrayVec<_, 8>>();
        let wait = wait
            .iter()
            .map(|x| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(x.0)
                    .stage_mask(x.1)
            })
            .collect::<ArrayVec<_, 16>>();
        let signal = triggers
            .iter()
            .map(|x| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(x.0)
                    .stage_mask(x.1)
            })
            .collect::<ArrayVec<_, 16>>();
        let command_bufers = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_bufers)
            .wait_semaphore_infos(&wait)
            .signal_semaphore_infos(&signal);
        unsafe { self.device.queue_submit2(queue, &[info], fence) }?;
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
            frame.reset(&self.device, &mut self.memory_allocator.lock())?;
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

    pub fn frame<F: FnOnce(&mut FrameRecorder) -> Result<(), Error>>(
        &self,
        target: &Swapchain,
        f: F,
    ) -> Result<FrameState, Error> {
        puffin::profile_function!();
        let compile_pipelines = Self::compile_all_pipelines(&self);
        let target = match target.acquire_next_image()? {
            AcquiredSurface::NeedRecreate => return Ok(FrameState::NeedRecreateSwapchain),
            AcquiredSurface::Image(image) => image,
        };
        let frame = self.begin_frame()?;

        let passes = {
            puffin::profile_scope!("Generate frame");
            let mut context = FrameRecorder {
                frame: &frame,
                passes: Default::default(),
                backbuffer: target.image,
            };
            f(&mut context)?;
            context.finish()
        };
        self.update_descriptors();
        {
            let mut staging = self.staging.lock();
            let upload = staging.upload(&self)?;
            let images = self.images.read();
            bevy_tasks::block_on(compile_pipelines)?;
            unsafe {
                self.device
                    .begin_command_buffer(frame.cb, &vk::CommandBufferBeginInfo::default())
            }?;
            staging.execute_pending_barriers(&self, frame.cb);
            let pipelines = self.pipelines.read();
            for pass in passes {
                let (pass, streams, image_barriers) = pass.consume();
                let sizes = pass
                    .color
                    .iter()
                    .map(|x| images.get_cold(x.image).unwrap().desc.dims)
                    .chain(
                        pass.depth
                            .map(|x| images.get_cold(x.image).unwrap().desc.dims),
                    )
                    .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
                assert!(!sizes.is_empty());
                let size = sizes[0];
                assert!(sizes.iter().all(|x| *x == size));
                let color_attachments = pass
                    .color
                    .iter()
                    .map(|x| x.build(&images).unwrap())
                    .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();
                let depth_attachment = pass.depth.iter().map(|x| x.build(&images).unwrap()).next();
                let render_area = vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: size[0],
                        height: size[1],
                    },
                };
                let mut rendering_info = vk::RenderingInfo::default()
                    .color_attachments(&color_attachments)
                    .layer_count(1)
                    .render_area(render_area);
                if let Some(depth) = &depth_attachment {
                    rendering_info = rendering_info.depth_attachment(depth);
                }
                image_barrier(&self.device, frame.cb, &images, &image_barriers);
                unsafe {
                    self.device.cmd_begin_rendering(frame.cb, &rendering_info);
                    self.device.cmd_set_viewport(
                        frame.cb,
                        0,
                        &[vk::Viewport::default()
                            .width(size[0] as _)
                            .height(size[1] as _)
                            .max_depth(0.0)
                            .max_depth(1.0)],
                    );
                    self.device.cmd_set_scissor(frame.cb, 0, &[render_area]);
                }
                streams.into_iter().try_for_each(|x| {
                    x.execute(DrawStreamExecuteContext {
                        device: &self.device,
                        cb: frame.cb,
                        pipelines: &pipelines,
                        descriptors: &[self.bindless_ds, self.sampler_ds],
                    })
                })?;
                unsafe { self.device.cmd_end_rendering(frame.cb) };
            }
            image_barrier(
                &self.device,
                frame.cb,
                &images,
                &[ImageBarrier::new(target.image, ImageBarrierType::ToPresent)],
            );
            unsafe { self.device.end_command_buffer(frame.cb) }?;
            let wait = [
                upload,
                (
                    target.acquire_semaphore,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                ),
            ];
            let trigger = [(
                target.rendering_finished,
                vk::PipelineStageFlags2::ALL_GRAPHICS,
            )];
            self.submit_graphics((frame.cb, frame.fence), &wait, &trigger)?;
            self.end_frame(frame);
        }
        self.present(target);
        Ok(FrameState::Rendered)
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

    fn update_descriptors(&self) {
        puffin::profile_function!();
        let images = self.images.read();
        let buffers = self.buffers.read();
        let sampled_images = self
            .sampled_images_to_update
            .lock()
            .drain()
            .map(|x| (x.index(), *images.get(x).unwrap()))
            .collect::<Vec<_>>();
        let storage_images = self
            .storage_images_to_update
            .lock()
            .drain()
            .map(|x| (x.index(), *images.get(x).unwrap()))
            .collect::<Vec<_>>();
        let storage_buffers = self
            .storage_buffers_to_update
            .lock()
            .drain()
            .map(|x| (x.index(), buffers.get_cold(x).unwrap().raw))
            .collect::<Vec<_>>();
        drop(images);
        drop(buffers);
        let mut writes =
            Vec::with_capacity(sampled_images.len() + storage_buffers.len() + storage_images.len());
        let image_info = TempList::new();
        let buffer_info = TempList::new();
        sampled_images.into_iter().for_each(|(index, view)| {
            writes.push(
                WriteDescriptorSet::default()
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_array_element(index)
                    .dst_set(self.bindless_ds)
                    .dst_binding(SAMPLED_IMAGES_SLOT)
                    .image_info(slice::from_ref(
                        image_info.add(
                            DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                                .image_view(view),
                        ),
                    )),
            )
        });
        storage_images.into_iter().for_each(|(index, view)| {
            writes.push(
                WriteDescriptorSet::default()
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_array_element(index)
                    .dst_set(self.bindless_ds)
                    .dst_binding(STORAGE_IMAGES_SLOT)
                    .image_info(slice::from_ref(
                        image_info.add(
                            DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                                .image_view(view),
                        ),
                    )),
            )
        });
        storage_buffers.into_iter().for_each(|(index, buffer)| {
            writes.push(
                WriteDescriptorSet::default()
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_array_element(index)
                    .dst_set(self.bindless_ds)
                    .dst_binding(STORAGE_BUFFERS_SLOT)
                    .buffer_info(slice::from_ref(
                        buffer_info.add(DescriptorBufferInfo::default().buffer(buffer)),
                    )),
            )
        });

        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }
}

impl<'game> Drop for RenderContext<'game> {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle() }.expect("device_wait_idle isn't supposed to fail");
        self.staging.lock().free(&self);
        let mut memory_allocator = self.memory_allocator.lock();
        let mut drop_list = self.current_drop_list.lock();
        self.images.write().drain().for_each(|(view, image)| {
            drop_list.drop_view(view);
            image.free(&mut drop_list);
        });
        self.buffers
            .write()
            .drain()
            .for_each(|(_, buffer)| buffer.free(&mut drop_list));
        drop_list.purge(&self.device, &mut memory_allocator);
        self.frames.iter().for_each(|frame| {
            Arc::get_mut(&mut frame.lock())
                .expect("Nothing should hold a frame at point when we destroy rendering context")
                .reset(&self.device, &mut memory_allocator)
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
                .free(&self.device, &mut memory_allocator)
        });
        self.samplers
            .drain()
            .for_each(|(_, sampler)| unsafe { self.device.destroy_sampler(sampler, None) });
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
            self.device
                .destroy_descriptor_pool(self.bindless_pool, None);
            self.device.destroy_descriptor_pool(self.sampler_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.bindless_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.sampler_layout, None);
            self.device.destroy_device(None);
        }
    }
}

pub(crate) struct PipelineCompilationContext<'a> {
    pub device: &'a ash::Device,
    pub programs: &'a ProgramPool,
}

impl<'a> PipelineCompilationContext<'a> {
    pub fn resolve_program(&self, handle: ProgramHandle) -> Option<&Program> {
        self.programs.get(handle.0 as usize)
    }
}
