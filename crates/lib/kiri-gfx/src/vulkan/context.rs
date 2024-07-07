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
    sync::Arc,
};

use ash::{
    khr::maintenance3,
    vk::{self, Bool32},
};
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use kiri_common::{Handle, HotColdPool, SentinelPoolStrategy};
use log::info;
use parking_lot::{Mutex, RwLock};
use raw_window_handle::RawDisplayHandle;
use std::fmt::Debug;

use crate::{BufferDesc, Error, Instance};

use super::{
    drop_list::DropList, frame::Frame, image::Image, physical_device::PhysicalDevice, GpuAllocator,
    GpuDescriptorAllocator, GpuMemory,
};

pub(crate) type ImageHandle = Handle<vk::ImageView>;
pub(crate) type BufferHandle = Handle<vk::Buffer>;

pub(crate) type ImagePool = HotColdPool<vk::ImageView, Image, SentinelPoolStrategy<vk::ImageView>>;
pub(crate) type BufferPool =
    HotColdPool<vk::Buffer, (GpuMemory, BufferDesc), SentinelPoolStrategy<vk::Buffer>>;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode: vk::SamplerAddressMode,
    pub anisotropy_level: u32,
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
    frames: [Mutex<Arc<Frame>>; 2],
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    universal_queue: Arc<Mutex<vk::Queue>>,
    transfer_queue: Arc<Mutex<vk::Queue>>,
    universal_queue_index: u32,
    transfer_queue_index: u32,
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
        let descriptor_allocator = Mutex::new(GpuDescriptorAllocator::new(0));

        let frames = [
            Mutex::new(Arc::new(Frame::new(&device, universal_queue_index)?)),
            Mutex::new(Arc::new(Frame::new(&device, universal_queue_index)?)),
        ];

        let debug = instance
            .debug_utils
            .iter()
            .map(|_| ash::ext::debug_utils::Device::new(instance.get(), &device))
            .next();

        Ok(Self {
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
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle() }.expect("device_wait_idle isn't supposed to fail");
        let mut drop_list = self.current_drop_list.lock();
        self.images.write().drain().for_each(|(view, image)| {
            drop_list.drop_view(view);
            image.destroy(&mut drop_list);
        });
        self.buffers
            .write()
            .drain()
            .for_each(|(buffer, (memory, _))| {
                drop_list.drop_buffer(buffer);
                drop_list.drop_memory(memory);
            })
    }
}
