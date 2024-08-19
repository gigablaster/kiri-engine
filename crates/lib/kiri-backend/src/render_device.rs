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

use std::{collections::HashMap, ffi::CString, mem, slice, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use gpu_descriptor_ash::AshDescriptorDevice;
use parking_lot::Mutex;
use std::fmt::Debug;

use crate::{AsVulkan, Error, Instance};

use super::{
    drop_list::DropList, frame::Frame, physical_device::PhysicalDevice, FindSuitableDevice,
    GpuAllocator, GpuDescriptorAllocator, GpuMemory, PhysicalDeviceType, Surface, SwapchainImage,
};

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

pub struct RenderDevice {
    instance: Arc<Instance>,
    pdevice: PhysicalDevice,
    device: ash::Device,
    debug: Option<ash::ext::debug_utils::Device>,
    memory_allocator: Mutex<GpuAllocator>,
    descriptor_allocator: Mutex<GpuDescriptorAllocator>,
    current_drop_list: Mutex<DropList>,
    frames: [Mutex<Arc<Frame>>; 2],
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    universal_queue: Arc<Mutex<vk::Queue>>,
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

        for ext in device_extension_names.iter().copied() {
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
            .find_queue(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER, &[])
            .ok_or(Error::NoSuitableQueue)?;

        let universal_queue_index = universal_queue_family.index;

        let queue_priorities = [1.0];
        let mut queue_info = Vec::new();
        queue_info.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(universal_queue_family.index)
                .queue_priorities(&queue_priorities),
        );

        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true));
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extension_names)
            .push_next(&mut features);

        let device = unsafe {
            instance
                .get()
                .create_device(pdevice.raw, &device_create_info, None)?
        };

        let universal_queue = Arc::new(Mutex::new(unsafe {
            device.get_device_queue(universal_queue_index, 0)
        }));

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

        let debug = instance
            .debug_utils()
            .iter()
            .map(|_| ash::ext::debug_utils::Device::new(instance.get(), &device))
            .next();

        let samplers = Self::generate_samplers(&device);

        Ok(Arc::new(Self {
            instance: instance.clone(),
            samplers,
            pdevice,
            memory_allocator: Mutex::new(GpuAllocator::new(allocator_config, allocator_props)),
            descriptor_allocator: Mutex::new(GpuDescriptorAllocator::new(0)),
            universal_queue,
            frames: [
                Mutex::new(Arc::new(Frame::new(&device)?)),
                Mutex::new(Arc::new(Frame::new(&device)?)),
            ],
            current_drop_list: Mutex::default(),
            device,
            debug,
        }))
    }

    pub fn get(&self) -> &ash::Device {
        &self.device
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn sampler(&self, desc: SamplerDesc) -> Option<vk::Sampler> {
        self.samplers.get(&desc).copied()
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

    pub(super) fn allocate_memory(
        &self,
        requirements: vk::MemoryRequirements,
        location: gpu_alloc::UsageFlags,
        dedicated: bool,
    ) -> Result<GpuMemory, Error> {
        let mut allocator = self.memory_allocator.lock();
        let request = gpu_alloc::Request {
            size: requirements.size,
            align_mask: requirements.alignment,
            usage: location,
            memory_types: requirements.memory_type_bits,
        };

        Ok(if dedicated {
            unsafe {
                allocator.alloc_with_dedicated(
                    AshMemoryDevice::wrap(&self.device),
                    request,
                    gpu_alloc::Dedicated::Required,
                )
            }
        } else {
            unsafe { allocator.alloc(AshMemoryDevice::wrap(&self.device), request) }
        }?)
    }

    pub fn with_drop_list<CB: FnOnce(&mut DropList)>(&self, cb: CB) {
        cb(&mut self.current_drop_list.lock());
    }

    pub fn with_descriptor_allocator<
        CB: FnOnce(&mut GpuDescriptorAllocator) -> Result<(), Error>,
    >(
        &self,
        cb: CB,
    ) -> Result<(), Error> {
        cb(&mut self.descriptor_allocator.lock())
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

    pub fn submit(
        &self,
        cb: (vk::CommandBuffer, vk::Fence),
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        triggers: &[vk::Semaphore],
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let wait_semaphores = wait.iter().map(|x| x.0).collect::<ArrayVec<_, 8>>();
        let wait_stages = wait.iter().map(|x| x.1).collect::<ArrayVec<_, 8>>();
        let command_bufers = [cb.0];
        let info = vk::SubmitInfo::default()
            .command_buffers(&command_bufers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(triggers)
            .wait_dst_stage_mask(&wait_stages);
        unsafe {
            self.device
                .queue_submit(*self.universal_queue.lock(), &[info], cb.1)
        }?;
        Ok(())
    }

    pub fn begin_frame(&self) -> Result<Arc<Frame>, Error> {
        puffin::profile_function!();
        let mut frame = self.frames[0].lock();
        {
            let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
            unsafe {
                self.device
                    .wait_for_fences(slice::from_ref(&frame.fence()), true, u64::MAX)?
            };
            frame.reset(
                &self.device,
                &mut self.memory_allocator.lock(),
                &mut self.descriptor_allocator.lock(),
            )?;
        }
        Ok(frame.clone())
    }

    pub fn end_frame(&self, frame: Arc<Frame>) {
        drop(frame);

        let mut frame = self.frames[0].lock();
        let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
        let mut next_frame = self.frames[1].lock();
        let next_frame = Arc::get_mut(&mut next_frame).unwrap();
        frame.assign_drop_list(mem::take(&mut self.current_drop_list.lock()));
        mem::swap(frame, next_frame);
    }

    pub fn present(&self, image: SwapchainImage) {
        puffin::profile_function!();
        let binding = image.swapchain.as_vulkan();
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

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.pdevice
    }
}

impl Drop for RenderDevice {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle() }.expect("device_wait_idle isn't supposed to fail");
        let mut memory_allocator = self.memory_allocator.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        let mut drop_list = self.current_drop_list.lock();
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
        self.frames.iter_mut().for_each(|x| {
            Arc::get_mut(&mut x.lock())
                .expect("Nothing should hold frame at this point")
                .free(
                    &self.device,
                    &mut memory_allocator,
                    &mut descriptor_allocator,
                )
        });
        self.samplers
            .drain()
            .for_each(|(_, sampler)| unsafe { self.device.destroy_sampler(sampler, None) });
        unsafe {
            descriptor_allocator.cleanup(AshDescriptorDevice::wrap(&self.device));
            memory_allocator.cleanup(AshMemoryDevice::wrap(&self.device));
            self.device.destroy_device(None);
        }
    }
}
