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
    collections::HashMap,
    ffi::{c_void, CStr, CString},
    mem, slice,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use kiri_common::{Handle, HotColdPool, SentinelPoolStrategy};
use parking_lot::{Mutex, RwLock};
use std::fmt::Debug;

use crate::{AcquiredSurface, BufferDesc, Error, Instance, Swapchain, SwapchainImage};

use super::{
    drop_list::DropList, frame::Frame, image::Image, physical_device::PhysicalDevice,
    staging::Staging, GpuAllocator, GpuDescriptor, GpuDescriptorAllocator, GpuMemory, Program,
};

pub type ImageHandle = Handle<vk::ImageView>;
pub type BufferHandle = Handle<vk::Buffer>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramHandle(pub(crate) u32);

pub(crate) type ImagePool = HotColdPool<vk::ImageView, Image, SentinelPoolStrategy<vk::ImageView>>;
pub(crate) type BufferPool =
    HotColdPool<vk::Buffer, (GpuMemory, BufferDesc), SentinelPoolStrategy<vk::Buffer>>;
pub(crate) type ProgramPool = Vec<Program>;

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

pub struct RenderContext {
    pub(crate) instance: Instance,
    pub(crate) pdevice: PhysicalDevice,
    pub(crate) device: ash::Device,
    debug: Option<ash::ext::debug_utils::Device>,
    memory_allocator: Mutex<GpuAllocator>,
    descriptor_allocator: Mutex<GpuDescriptorAllocator>,
    current_drop_list: Mutex<DropList>,
    pub(crate) images: RwLock<ImagePool>,
    pub(crate) buffers: RwLock<BufferPool>,
    pub(crate) programs: RwLock<ProgramPool>,
    frames: [Mutex<Arc<Frame>>; 2],
    pub(crate) samplers: HashMap<SamplerDesc, vk::Sampler>,
    universal_queue: Arc<Mutex<vk::Queue>>,
    transfer_queue: Arc<Mutex<vk::Queue>>,
    pub(crate) universal_queue_index: u32,
    pub(crate) transfer_queue_index: u32,
    pub(crate) staging: Mutex<Staging>,
}

impl Debug for RenderContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VkDevice({})", vk::Handle::as_raw(self.device.handle()))
    }
}

impl RenderContext {
    fn new(instance: Instance, pdevice: PhysicalDevice) -> Result<Self, Error> {
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

        let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true);
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut dynamic_rendering)
            .push_next(&mut synchronization2)
            .push_next(&mut descriptor_indexing)
            .push_next(&mut maintenance4)
            .push_next(&mut buffer_device_address);
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
        let memory_allocator = Mutex::new(GpuAllocator::new(allocator_config, allocator_props));
        let descriptor_allocator = Mutex::new(GpuDescriptorAllocator::new(1));

        let frames = [
            Mutex::new(Arc::new(Frame::new(&device, universal_queue_index)?)),
            Mutex::new(Arc::new(Frame::new(&device, universal_queue_index)?)),
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
            &mut memory_allocator.lock(),
        )?);

        Ok(Self {
            staging,
            instance,
            samplers: Self::generate_samplers(&device),
            pdevice,
            memory_allocator,
            descriptor_allocator,
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
        })
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
            .collect::<ArrayVec<_, 8>>();
        let signal = triggers
            .iter()
            .map(|x| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(x.0)
                    .stage_mask(x.1)
            })
            .collect::<ArrayVec<_, 9>>();
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
            frame.reset(
                &self.device,
                &mut self.memory_allocator.lock(),
                &mut self.descriptor_allocator.lock(),
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

    pub fn frame<F: FnOnce(&mut FrameRecordContext) -> Result<(), Error>>(
        &self,
        target: &Swapchain,
        f: F,
    ) -> Result<FrameState, Error> {
        puffin::profile_function!();
        let target = match target.acquire_next_image()? {
            AcquiredSurface::NeedRecreate => return Ok(FrameState::NeedRecreateSwapchain),
            AcquiredSurface::Image(image) => image,
        };
        let frame = self.begin_frame()?;
        let mut context = FrameRecordContext { context: &self };
        {
            puffin::profile_scope!("Generate frame");
            f(&mut context)?;
        }
        let mut staging = self.staging.lock();
        let upload = staging.upload(&self)?;
        let images = self.images.read();
        let buffers = self.buffers.read();
        unsafe {
            self.device
                .begin_command_buffer(frame.cb, &vk::CommandBufferBeginInfo::default())
        }?;
        staging.execute_pending_barriers(&self, frame.cb);
        // TODO:: passes
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
}

pub struct FrameRecordContext<'a> {
    context: &'a RenderContext,
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle() }.expect("device_wait_idle isn't supposed to fail");
        self.staging.lock().free(&self);
        let mut memory_allocator = self.memory_allocator.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        let mut drop_list = self.current_drop_list.lock();
        self.images.write().drain().for_each(|(view, image)| {
            drop_list.drop_view(view);
            image.free(&mut drop_list);
        });
        self.buffers
            .write()
            .drain()
            .for_each(|(buffer, (memory, _))| {
                drop_list.drop_buffer(buffer);
                drop_list.drop_memory(memory);
            });
        drop_list.purge(
            &self.device,
            &mut memory_allocator,
            &mut descriptor_allocator,
        );
        self.frames.iter().for_each(|frame| {
            Arc::get_mut(&mut frame.lock())
                .expect("Nothing should hold a frame at point when we destroy rendering context")
                .reset(
                    &self.device,
                    &mut memory_allocator,
                    &mut descriptor_allocator,
                )
                .unwrap();
        });
        self.programs
            .write()
            .drain(..)
            .for_each(|x| x.free(&self.device));
    }
}

pub(crate) struct RenderingContext<'a> {
    pub images: &'a ImagePool,
    pub buffers: &'a BufferPool,
}

pub(crate) struct PipelineCompilationContext<'a> {
    pub device: &'a ash::Device,
    programs: &'a ProgramPool,
}

impl<'a> PipelineCompilationContext<'a> {
    pub fn resolve_program(&self, handle: ProgramHandle) -> Option<&Program> {
        self.programs.get(handle.0 as usize)
    }
}
