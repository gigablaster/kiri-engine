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

use std::{collections::HashMap, ffi::CString, marker::PhantomData, mem, slice, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use gpu_alloc_ash::AshMemoryDevice;
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockUpgradableReadGuard};
use std::fmt::Debug;

use crate::{
    create_descriptor_layout, staging::Staging, DescriptorSetLayoutDesc, Error, GpuDescriptor,
    GpuDescriptorAllocator, GpuMemoryBlock, Instance,
};

use super::{
    drop_list::DropList, frame::Frame, physical_device::PhysicalDevice, FindSuitableDevice,
    GpuAllocator, PhysicalDeviceType, Surface, SwapchainImage,
};

const MAX_SUBMITS: usize = 32;
pub(super) const MAX_RESOURCES: u32 = 64536;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode: vk::SamplerAddressMode,
    pub anisotropy_level: u32,
}

pub struct RenderDevice {
    pub instance: Arc<Instance>,
    pub physical_device: PhysicalDevice,
    pub raw: ash::Device,
    debug: Option<ash::ext::debug_utils::Device>,
    current_drop_list: Mutex<DropList>,
    frames: [Mutex<Arc<Frame>>; 2],
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    universal_queue: Arc<Queue>,
    layouts: RwLock<HashMap<DescriptorSetLayoutDesc, vk::DescriptorSetLayout>>,
    memory_allocator: Mutex<GpuAllocator>,
    descriptor_allocator: Mutex<GpuDescriptorAllocator>,
    staging: Mutex<Staging>,
}

impl Debug for RenderDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VkDevice({})", vk::Handle::as_raw(self.raw.handle()))
    }
}

#[derive(Debug)]
pub(super) struct Queue {
    pub family_index: u32,
    raw: Mutex<vk::Queue>,
}

impl Queue {
    /// Submits execution to main queue
    ///
    /// Thread-safe.
    pub fn submit(
        &self,
        device: &ash::Device,
        cbs: &[vk::CommandBuffer],
        fence: vk::Fence,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        signal: &[vk::Semaphore],
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let wait_sems = wait
            .iter()
            .map(|(semaphore, _)| *semaphore)
            .collect::<ArrayVec<_, MAX_SUBMITS>>();
        let wait_stages = wait
            .iter()
            .map(|(_, stage)| *stage)
            .collect::<ArrayVec<_, MAX_SUBMITS>>();
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(cbs)
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(signal);
        unsafe { device.queue_submit(*self.raw.lock(), &[submit_info], fence) }?;
        Ok(())
    }
}

pub struct DescriptorAllocatorContext<'a, E: From<Error>> {
    device: &'a ash::Device,
    allocator: &'a mut GpuDescriptorAllocator,
    phantom_data: PhantomData<E>,
}

impl<E: From<Error>> DescriptorAllocatorContext<'_, E> {
    pub fn allocate(
        &mut self,
        layout: vk::DescriptorSetLayout,
        layout_descriptor_count: &DescriptorTotalCount,
        count: usize,
    ) -> Result<Vec<GpuDescriptor>, E> {
        Ok(self.allocate_impl(layout, layout_descriptor_count, count, false)?)
    }

    pub fn allocate_bindless(
        &mut self,
        layout: vk::DescriptorSetLayout,
        layout_descriptor_count: &DescriptorTotalCount,
        count: usize,
    ) -> Result<Vec<GpuDescriptor>, Error> {
        Ok(self.allocate_impl(layout, layout_descriptor_count, count, true)?)
    }

    fn allocate_impl(
        &mut self,
        layout: vk::DescriptorSetLayout,
        layout_descriptor_count: &DescriptorTotalCount,
        count: usize,
        bindless: bool,
    ) -> Result<Vec<GpuDescriptor>, Error> {
        let flags = if bindless {
            DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND
        } else {
            DescriptorSetLayoutCreateFlags::empty()
        };
        Ok(unsafe {
            self.allocator.allocate(
                AshDescriptorDevice::wrap(self.device),
                &layout,
                flags,
                layout_descriptor_count,
                count as _,
            )
        }?)
    }
}

impl RenderDevice {
    pub fn new(
        instance: &Arc<Instance>,
        surface: &Surface,
        preferences: &[PhysicalDeviceType],
        max_update_after_bind_descriptors_in_all_pools: Option<usize>,
    ) -> Result<Arc<Self>, Error> {
        let physical_devices = instance.enumerate_physical_devices()?;
        let pdevice = physical_devices
            .find_suitable_device(surface, preferences)
            .ok_or(Error::NoSuitableDevice)?;
        if !pdevice.is_queue_flag_supported(vk::QueueFlags::GRAPHICS) {
            return Err(Error::NoSuitableDevice);
        };
        let device_extension_names = [
            ash::khr::swapchain::NAME,
            ash::khr::maintenance1::NAME,
            ash::khr::maintenance2::NAME,
            ash::khr::maintenance3::NAME,
            ash::khr::maintenance4::NAME,
        ];

        for ext in device_extension_names.iter() {
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
            .find_queue(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER | vk::QueueFlags::COMPUTE,
                &[],
            )
            .ok_or(Error::NoSuitableQueue)?;

        let universal_queue_index = universal_queue_family.index;

        let queue_priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(universal_queue_family.index)
            .queue_priorities(&queue_priorities)];

        let mut buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true);
        let mut dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut synchornization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true);

        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true))
            .push_next(&mut buffer_device_address)
            .push_next(&mut maintenance4)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut synchornization2)
            .push_next(&mut descriptor_indexing);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extension_names)
            .push_next(&mut features);

        let device = unsafe {
            instance
                .raw
                .create_device(pdevice.raw, &device_create_info, None)?
        };

        let universal_queue = Arc::new(Queue {
            family_index: universal_queue_index,
            raw: Mutex::new(unsafe { device.get_device_queue(universal_queue_index, 0) }),
        });

        let debug = instance
            .debug_utils()
            .iter()
            .map(|_| ash::ext::debug_utils::Device::new(&instance.raw, &device))
            .next();

        let samplers = Self::generate_samplers(&device);

        let allocator_config = gpu_alloc::Config {
            dedicated_threshold: 128 * 1024 * 1024,
            preferred_dedicated_threshold: 64 * 1024 * 1024,
            transient_dedicated_threshold: 128 * 1024 * 1024,
            starting_free_list_chunk: 4 * 1024 * 1024,
            final_free_list_chunk: 32 * 1024 * 1024,
            minimal_buddy_size: 256 * 1024,
            initial_buddy_dedicated_size: 256 * 1024 * 1024,
        };

        let mut allocator = GpuAllocator::new(allocator_config, unsafe {
            gpu_alloc_ash::device_properties(&instance.raw, Instance::vulkan_version(), pdevice.raw)
        }?);

        Ok(Arc::new(Self {
            instance: instance.clone(),
            staging: Mutex::new(Staging::new(
                &device,
                &pdevice,
                &mut allocator,
                universal_queue.clone(),
            )?),
            samplers,
            universal_queue,
            frames: [
                Mutex::new(Arc::new(Frame::new(&device)?)),
                Mutex::new(Arc::new(Frame::new(&device)?)),
            ],
            current_drop_list: Mutex::default(),
            raw: device,
            debug,
            layouts: Default::default(),
            memory_allocator: Mutex::new(allocator),
            physical_device: pdevice,
            descriptor_allocator: Mutex::new(GpuDescriptorAllocator::new(
                max_update_after_bind_descriptors_in_all_pools.unwrap_or(0) as _,
            )),
        }))
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

    pub fn submit(
        &self,
        cbs: &[vk::CommandBuffer],
        fence: vk::Fence,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        signal: &[vk::Semaphore],
    ) -> Result<(), Error> {
        self.universal_queue
            .submit(&self.raw, cbs, fence, wait, signal)
    }

    pub(super) fn with_drop_list<CB: FnOnce(&mut DropList)>(&self, cb: CB) {
        cb(&mut self.current_drop_list.lock());
    }

    pub fn drop_descriptors(&self, descriptors: impl IntoIterator<Item = GpuDescriptor>) {
        self.with_drop_list(|drop_list| {
            for descriptor in descriptors {
                drop_list.drop_descriptor(descriptor);
            }
        });
    }

    pub fn with_descriptor_allocator<
        CB: FnOnce(&mut DescriptorAllocatorContext<E>) -> Result<(), E>,
        E: From<Error>,
    >(
        &self,
        cb: CB,
    ) -> Result<(), E> {
        let mut allocator = self.descriptor_allocator.lock();
        cb(&mut DescriptorAllocatorContext {
            device: &self.raw,
            allocator: &mut allocator,
            phantom_data: PhantomData,
        })
    }

    pub fn allocate_descriptor_sets(
        &self,
        layout: vk::DescriptorSetLayout,
        layout_descriptor_count: &DescriptorTotalCount,
        count: usize,
    ) -> Result<Vec<GpuDescriptor>, Error> {
        let mut allocator = self.descriptor_allocator.lock();
        DescriptorAllocatorContext::<Error> {
            device: &self.raw,
            allocator: &mut allocator,
            phantom_data: PhantomData,
        }
        .allocate(layout, layout_descriptor_count, count)
    }

    pub fn allocate_bindless_descriptor_sets(
        &self,
        layout: vk::DescriptorSetLayout,
        layout_descriptor_count: &DescriptorTotalCount,
        count: usize,
    ) -> Result<Vec<GpuDescriptor>, Error> {
        let mut allocator = self.descriptor_allocator.lock();
        DescriptorAllocatorContext::<Error> {
            device: &self.raw,
            allocator: &mut allocator,
            phantom_data: PhantomData,
        }
        .allocate_bindless(layout, layout_descriptor_count, count)
    }

    pub(super) fn with_staging<CB: FnOnce(&mut Staging) -> Result<(), Error>>(
        &self,
        cb: CB,
    ) -> Result<(), Error> {
        cb(&mut self.staging.lock())
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

    pub fn begin_label(&self, command_buffer: vk::CommandBuffer, name: &str) {
        if let Some(debug_utils) = &self.debug {
            unsafe {
                debug_utils.cmd_begin_debug_utils_label(
                    command_buffer,
                    &vk::DebugUtilsLabelEXT::default().label_name(&CString::new(name).unwrap()),
                )
            };
        }
    }

    pub fn end_label(&self, command_buffer: vk::CommandBuffer) {
        if let Some(debug_utils) = &self.debug {
            unsafe { debug_utils.cmd_end_debug_utils_label(command_buffer) };
        }
    }

    /// Begins frame
    ///
    /// Waiting for last frame to finish rendering, them resets fences and frame state.
    /// Returns frame data and staging semaphore
    pub fn begin_frame(&self) -> Result<(Arc<Frame>, vk::Semaphore), Error> {
        puffin::profile_function!();
        let mut frame = self.frames[0].lock();
        {
            puffin::profile_scope!("Waiting for frame to be finished");
            let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
            unsafe {
                self.raw
                    .wait_for_fences(&[frame.render_fence], true, u64::MAX)?
            };
            frame.reset(
                &self.raw,
                &mut self.memory_allocator.lock(),
                &mut self.descriptor_allocator.lock(),
            )?;
        }
        let upload_finished = self.staging.lock().upload(&self.raw)?;
        Ok((frame.clone(), upload_finished))
    }

    /// Ends frame
    ///
    /// Current frame marked for execution, last frame moved to be waited.
    pub fn end_frame(&self, frame: Arc<Frame>) {
        drop(frame);

        let mut frame = self.frames[0].lock();
        let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
        let mut next_frame = self.frames[1].lock();
        let next_frame = Arc::get_mut(&mut next_frame).unwrap();
        frame.assign_drop_list(mem::take(&mut self.current_drop_list.lock()));
        mem::swap(frame, next_frame);
    }

    pub fn present(&self, target: SwapchainImage, frame: &Frame) -> Result<(), Error> {
        puffin::profile_function!();

        let binding = target.swapchain.raw;
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&frame.render_finished))
            .swapchains(slice::from_ref(&binding))
            .image_indices(slice::from_ref(&target.image_index));

        match unsafe {
            target
                .swapchain
                .loader()
                .queue_present(*self.universal_queue.raw.lock(), &present_info)
        } {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => Ok(()),
            Err(err) => panic!("Can't present image: {}", err),
            _ => Ok(()),
        }
    }

    pub fn get_or_create_layout(
        &self,
        stage: vk::ShaderStageFlags,
        desc: &DescriptorSetLayoutDesc,
    ) -> Result<vk::DescriptorSetLayout, Error> {
        let layouts = self.layouts.upgradable_read();
        if let Some(layout) = layouts.get(&desc) {
            Ok(*layout)
        } else {
            let mut layouts = RwLockUpgradableReadGuard::upgrade(layouts);
            if let Some(layout) = layouts.get(&desc) {
                Ok(*layout)
            } else {
                let layout = create_descriptor_layout(self, stage, desc)?;
                layouts.insert(desc.clone(), layout);
                Ok(layout)
            }
        }
    }

    pub(super) fn allocate(
        &self,
        requirement: vk::MemoryRequirements,
        usage: gpu_alloc::UsageFlags,
        dedicated: bool,
    ) -> Result<GpuMemoryBlock, Error> {
        let mut allocator = self.memory_allocator.lock();
        let request = gpu_alloc::Request {
            size: requirement.size.max(requirement.alignment),
            align_mask: requirement.alignment,
            memory_types: requirement.memory_type_bits,
            usage,
        };
        let block = if dedicated {
            unsafe {
                allocator.alloc_with_dedicated(
                    AshMemoryDevice::wrap(&self.raw),
                    request,
                    gpu_alloc::Dedicated::Required,
                )
            }?
        } else {
            unsafe { allocator.alloc(AshMemoryDevice::wrap(&self.raw), request) }?
        };
        Ok(block)
    }
}

impl Drop for RenderDevice {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle() }.expect("device_wait_idle isn't supposed to fail");
        let mut drop_list = self.current_drop_list.lock();
        let mut memory_allocator = self.memory_allocator.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        self.staging.lock().free(&self.raw, &mut memory_allocator);
        drop_list.purge(&self.raw, &mut memory_allocator, &mut descriptor_allocator);
        self.frames.iter().for_each(|frame| {
            Arc::get_mut(&mut frame.lock())
                .expect("Nothing should hold a frame at point when we destroy rendering context")
                .reset(&self.raw, &mut memory_allocator, &mut descriptor_allocator)
                .unwrap();
        });
        self.frames.iter_mut().for_each(|x| {
            Arc::get_mut(&mut x.lock())
                .expect("Nothing should hold frame at this point")
                .free(&self.raw, &mut memory_allocator, &mut descriptor_allocator);
        });
        self.samplers
            .drain()
            .for_each(|(_, sampler)| unsafe { self.raw.destroy_sampler(sampler, None) });
        self.layouts.write().drain().for_each(|(_, layout)| unsafe {
            self.raw.destroy_descriptor_set_layout(layout, None)
        });
        unsafe {
            memory_allocator.cleanup(AshMemoryDevice::wrap(&self.raw));
            descriptor_allocator.cleanup(AshDescriptorDevice::wrap(&self.raw));
            self.raw.destroy_device(None);
        }
    }
}
