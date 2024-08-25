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
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};
use std::fmt::Debug;

use crate::{
    create_descriptor_layout, DescriptorSetLayoutDesc, Error, GpuMemoryPage, Image, Instance,
};

use super::{
    drop_list::DropList, frame::Frame, physical_device::PhysicalDevice, FindSuitableDevice,
    GpuAllocator, PhysicalDeviceType, Surface, SwapchainImage,
};

const MAX_SUBMITS: usize = 32;
const MEMORY_PAGE_SIZE: u64 = 256 * 1024 * 1024;

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
    universal_queue: Arc<Mutex<vk::Queue>>,
    layouts: RwLock<HashMap<DescriptorSetLayoutDesc, vk::DescriptorSetLayout>>,
    allocated_memory: Mutex<HashMap<u32, Vec<GpuMemoryPage>>>,
}

impl Debug for RenderDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VkDevice({})", vk::Handle::as_raw(self.raw.handle()))
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

        let mut synchronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true);
        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true))
            .push_next(&mut synchronization2)
            .push_next(&mut buffer_device_address)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut maintenance4)
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

        let universal_queue = Arc::new(Mutex::new(unsafe {
            device.get_device_queue(universal_queue_index, 0)
        }));

        let debug = instance
            .debug_utils()
            .iter()
            .map(|_| ash::ext::debug_utils::Device::new(&instance.raw, &device))
            .next();

        let samplers = Self::generate_samplers(&device);

        Ok(Arc::new(Self {
            instance: instance.clone(),
            samplers,
            physical_device: pdevice,
            universal_queue,
            frames: [
                Mutex::new(Arc::new(Frame::new(&device)?)),
                Mutex::new(Arc::new(Frame::new(&device)?)),
            ],
            current_drop_list: Mutex::default(),
            raw: device,
            debug,
            layouts: Default::default(),
            allocated_memory: Default::default(),
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

    pub(super) fn with_drop_list<CB: FnOnce(&mut DropList)>(&self, cb: CB) {
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

    pub fn end_labe(&self, command_buffer: vk::CommandBuffer) {
        if let Some(debug_utils) = &self.debug {
            unsafe { debug_utils.cmd_end_debug_utils_label(command_buffer) };
        }
    }

    /// Submits execution to main queue
    ///
    /// Thread-safe.
    pub fn submit(
        &self,
        cbs: &[vk::CommandBuffer],
        fence: vk::Fence,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        signal: &[(vk::Semaphore, vk::PipelineStageFlags2)],
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let command_info = cbs
            .iter()
            .map(|x| vk::CommandBufferSubmitInfo::default().command_buffer(*x))
            .collect::<ArrayVec<_, MAX_SUBMITS>>();
        let wait_info = wait
            .iter()
            .map(|(semaphore, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*semaphore)
                    .stage_mask(*stage)
            })
            .collect::<ArrayVec<_, MAX_SUBMITS>>();
        let signal_info = signal
            .iter()
            .map(|(semaphore, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*semaphore)
                    .stage_mask(*stage)
            })
            .collect::<ArrayVec<_, MAX_SUBMITS>>();
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_info)
            .wait_semaphore_infos(&wait_info)
            .signal_semaphore_infos(&signal_info);
        unsafe {
            self.raw
                .queue_submit2(*self.universal_queue.lock(), &[submit_info], fence)
        }?;
        Ok(())
    }

    /// Begins frame
    ///
    /// Waiting for last frame to finish rendering, them resets fences and frame state.
    pub fn begin_frame(&self) -> Result<Arc<Frame>, Error> {
        puffin::profile_function!();
        let mut frame = self.frames[0].lock();
        {
            let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
            unsafe {
                self.raw.wait_for_fences(
                    &[frame.present_fence, frame.render_fence],
                    true,
                    u64::MAX,
                )?
            };
            frame.reset(&self.raw)?;
        }
        Ok(frame.clone())
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

    /// Gets image and copy it to backbuffer.
    ///
    /// As far as I understand it will let us to execute redering commands while waiting for
    /// vsync. So as soon as back bffer is there we can just copy result and go for another
    /// frame.    
    pub fn present(
        &self,
        target: SwapchainImage,
        image: &Image,
        frame: &Frame,
    ) -> Result<(), Error> {
        puffin::profile_function!();
        unsafe {
            let cb = frame.get_command_buffer(&self.raw, vk::CommandBufferLevel::PRIMARY)?;
            self.raw.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.begin_label(cb, "prsent");
            let barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE) // ?
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(target.image.raw)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                vk::ImageMemoryBarrier2::default()
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE) // ?
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(image.raw)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            ];
            self.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&barriers),
            );
            self.raw.cmd_blit_image(
                cb,
                image.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target.image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D::default()
                            .x(image.desc.dims[0] as _)
                            .y(image.desc.dims[1] as _)
                            .z(1),
                    ])
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D::default()
                            .x(target.image.desc.dims[0] as _)
                            .y(target.image.desc.dims[1] as _)
                            .z(1),
                    ])
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
                vk::Filter::LINEAR,
            );
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE) // ?
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(target.image.raw)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&[barrier]),
            );
            self.end_labe(cb);
            self.raw.end_command_buffer(cb)?;
            self.submit(
                &[cb],
                frame.present_fence,
                &[
                    (
                        frame.render_finished,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    ),
                    (
                        target.acquire_semaphore,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    ),
                ],
                &[(
                    target.present_finished,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                )],
            )?;
        }
        let binding = target.swapchain.raw;
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&target.present_finished))
            .swapchains(slice::from_ref(&binding))
            .image_indices(slice::from_ref(&target.image_index));

        match unsafe {
            target
                .swapchain
                .loader()
                .queue_present(*self.universal_queue.lock(), &present_info)
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
        let key = desc.normalize();
        let layouts = self.layouts.upgradable_read();
        if let Some(layout) = layouts.get(desc) {
            Ok(*layout)
        } else {
            let mut layouts = RwLockUpgradableReadGuard::upgrade(layouts);
            if let Some(layout) = layouts.get(desc) {
                Ok(*layout)
            } else {
                let layout = create_descriptor_layout(self, stage, desc)?;
                layouts.insert(key, layout);
                Ok(layout)
            }
        }
    }

    pub(super) fn get_suitable_memory_index(
        &self,
        required_type_bits: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        unsafe {
            self.instance
                .raw
                .get_physical_device_memory_properties(self.physical_device.raw)
                .memory_types_as_slice()
                .iter()
                .enumerate()
                .find_map(|(index, data)| {
                    let type_bits = 1 << index;
                    let is_required_type = required_type_bits & type_bits != 0;
                    let has_required_properties = data.property_flags & flags == flags;
                    if is_required_type && has_required_properties {
                        Some(index as u32)
                    } else {
                        None
                    }
                })
        }
    }

    pub(super) fn get_memory_page(&self, index: u32) -> Result<GpuMemoryPage, Error> {
        let mut pages = self.allocated_memory.lock();
        let group = pages.entry(index).or_default();
        if let Some(page) = group.pop() {
            Ok(page)
        } else {
            self.allocate_page(index, MEMORY_PAGE_SIZE)
        }
    }

    pub(super) fn release_memory_page(&self, page: GpuMemoryPage) {
        // We only do it when there's state transition. So it's fine to stall
        unsafe { self.raw.device_wait_idle() }.unwrap();
        page.reset();
        self.allocated_memory
            .lock()
            .entry(page.index)
            .or_default()
            .push(page);
    }

    pub(super) fn use_allocator_or_dedicated(
        &self,
        allocator: Option<&GpuAllocator>,
        requirements: vk::MemoryRequirements,
        memory_location: vk::MemoryPropertyFlags,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize, Option<GpuMemoryPage>), Error> {
        if let Some(allocator) = allocator {
            let (memory, offset) = allocator.allocate(requirements, memory_location)?;

            Ok((memory, offset, None))
        } else {
            let index = self
                .get_suitable_memory_index(requirements.memory_type_bits, memory_location)
                .ok_or(Error::NoSuitableMemoryType)?;
            let page = self.allocate_page(index, requirements.size)?;
            self.set_object_name(page.memory, format!("Memory page index {}", index));
            Ok((page.memory, 0, Some(page)))
        }
    }

    pub(super) fn allocate_page(&self, index: u32, size: u64) -> Result<GpuMemoryPage, Error> {
        GpuMemoryPage::new(&self.raw, index, size)
    }
}

impl Drop for RenderDevice {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle() }.expect("device_wait_idle isn't supposed to fail");
        let mut drop_list = self.current_drop_list.lock();
        drop_list.purge(&self.raw);
        self.frames.iter().for_each(|frame| {
            Arc::get_mut(&mut frame.lock())
                .expect("Nothing should hold a frame at point when we destroy rendering context")
                .reset(&self.raw)
                .unwrap();
        });
        self.frames.iter_mut().for_each(|x| {
            Arc::get_mut(&mut x.lock())
                .expect("Nothing should hold frame at this point")
                .free(&self.raw)
        });
        self.samplers
            .drain()
            .for_each(|(_, sampler)| unsafe { self.raw.destroy_sampler(sampler, None) });
        self.layouts.write().drain().for_each(|(_, layout)| unsafe {
            self.raw.destroy_descriptor_set_layout(layout, None)
        });
        self.allocated_memory
            .lock()
            .drain()
            .for_each(|(_, mut group)| group.drain(..).for_each(|page| page.free(&self.raw)));
        unsafe {
            self.raw.destroy_device(None);
        }
    }
}
