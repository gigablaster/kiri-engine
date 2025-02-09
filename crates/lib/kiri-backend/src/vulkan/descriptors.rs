// Copyright (C) 2023-2025 gigablaster

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

use std::slice;

use ash::vk;
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use kiri_common::{HotColdPool, TempList};
use parking_lot::{MutexGuard, RwLockWriteGuard};

use crate::Error;

use super::{
    buffer::BufferPool, image::ImagePool, BufferHandle, BufferSlice, DescriptorHandle,
    GpuDescriptor, GraphicsDevice, ImageHandle, ImageViewDesc,
};

pub(crate) type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;

pub const PASS_DESCRIPTOR_SLOT_INDEX: usize = 0;
pub const OBJECT_DESCRIPTOR_SLOT_INDEX: usize = 1;
pub const MATERIAL_DESCRIPTOR_SLOT_IDNEX: usize = 2;
pub const DYNAMIC_DESCRIPTOR_SLOT_INDEX: usize = 3;
pub const MAX_DESCRIPTOR_SETS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorDesc<'a> {
    pub name: &'a str,
    pub ty: vk::DescriptorType,
    pub count: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorLayoutDesc<'a> {
    pub layout: &'a [(usize, DescriptorDesc<'a>)],
    pub compute_groups_size: Option<(u32, u32, u32)>,
}

impl<'a> DescriptorLayoutDesc<'a> {
    pub fn has_slot(&self, index: usize) -> bool {
        self.layout.iter().any(|(x, _)| *x == index)
    }

    pub fn get_slot(&self, name: &str) -> Option<usize> {
        self.layout
            .iter()
            .find_map(|(slot, desc)| (desc.name == name).then_some(*slot))
    }

    pub fn get_desc(&self, slot: usize) -> Option<&DescriptorDesc> {
        self.layout
            .iter()
            .find_map(|(x, data)| if slot == *x { Some(data) } else { None })
    }

    pub fn get_layout(&self) -> &[(usize, DescriptorDesc<'a>)] {
        self.layout
    }

    pub fn by_types(
        &self,
        ty: &'a [vk::DescriptorType],
    ) -> impl Iterator<Item = (usize, DescriptorDesc)> {
        self.layout
            .iter()
            .copied()
            .filter(move |x| ty.contains(&x.1.ty))
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ShaderDesc<'a> {
    pub stage: vk::ShaderStageFlags,
    pub entry: &'a str,
    pub code: &'a [u8],
}

impl<'a> ShaderDesc<'a> {
    pub fn new(stage: vk::ShaderStageFlags, code: &'a [u8]) -> Self {
        Self {
            stage,
            entry: "main",
            code,
        }
    }

    pub fn vertex(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::VERTEX,
            entry: "main",
            code,
        }
    }

    pub fn fragment(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::FRAGMENT,
            entry: "main",
            code,
        }
    }

    pub fn compute(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::COMPUTE,
            entry: "main",
            code,
        }
    }

    pub fn entry(mut self, entry: &'a str) -> Self {
        self.entry = entry;
        self
    }
}

#[derive(Debug, Default)]
pub struct DescriptorSetCreateData<'a> {
    pub layout: DescriptorLayoutDesc<'static>,
    pub stages: vk::ShaderStageFlags,
    pub images: &'a [ImageHandle],
    pub unifoms: &'a [BufferSlice],
    pub storages: &'a [BufferSlice],
    pub dynamic_uniforms: &'a [(BufferHandle, usize)],
    pub dynamic_storage_buffers: &'a [(BufferHandle, usize)],
    pub name: Option<&'a str>,
}

pub struct DescriptorUpdateContext<'a> {
    device: &'a GraphicsDevice,
    descriptors: RwLockWriteGuard<'a, DescriptorPool>,
    dirty: MutexGuard<'a, Vec<DescriptorHandle>>,
    to_destroy: MutexGuard<'a, Vec<DescriptorHandle>>,
}

impl DescriptorUpdateContext<'_> {
    pub fn create_descriptor(
        &mut self,
        data: DescriptorSetCreateData,
    ) -> Result<DescriptorHandle, Error> {
        let data = data.build(self.device)?;
        let handle = self.descriptors.push(vk::DescriptorSet::null(), data);
        self.dirty.push(handle);
        Ok(handle)
    }

    pub fn destroy_descriptor(&mut self, handle: DescriptorHandle) {
        self.to_destroy.push(handle);
    }
}

pub const EMPTY_DESCRIPTOR_LAYOUT: DescriptorLayoutDesc = DescriptorLayoutDesc {
    layout: &[],
    compute_groups_size: None,
};

impl<'a> DescriptorLayoutDesc<'a> {
    pub(crate) fn get_descriptor_count(&self) -> DescriptorTotalCount {
        let mut count = DescriptorTotalCount::default();
        for (_, data) in self.layout.iter() {
            match data.ty {
                vk::DescriptorType::SAMPLED_IMAGE => count.sampled_image += data.count as u32,
                vk::DescriptorType::UNIFORM_BUFFER => count.uniform_buffer += data.count as u32,
                vk::DescriptorType::STORAGE_BUFFER => count.storage_buffer += data.count as u32,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                    count.uniform_buffer_dynamic += data.count as u32
                }
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    count.uniform_buffer_dynamic += data.count as u32
                }
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    count.combined_image_sampler += data.count as u32
                }
                ty => panic!("Binding type {:?} not supported", ty),
            }
        }
        count
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
pub(crate) struct DescriptorSetData {
    pub descriptor: Option<GpuDescriptor>,
    count: DescriptorTotalCount,
    layout: vk::DescriptorSetLayout,
    images: Vec<Binding<ImageBindingData>>,
    uniforms: Vec<Binding<BufferSlice>>,
    storages: Vec<Binding<BufferSlice>>,
    dynamic_uniforms: Vec<Binding<DynamicBufferBindingData>>,
    dynamic_storages: Vec<Binding<DynamicBufferBindingData>>,
    name: Option<String>,
}

impl DescriptorSetCreateData<'_> {
    pub(crate) fn build(self, device: &GraphicsDevice) -> Result<DescriptorSetData, Error> {
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

    // We want descriptors, buffers and images to be locked at this point. So we pass them from outside.
}

const DEFAULT_UPDATES: usize = 16384;

impl GraphicsDevice {
    pub(crate) fn update_descriptors(
        &self,
        descriptors: &mut DescriptorPool,
        buffers: &BufferPool,
        images: &ImagePool,
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let mut drop_list = Vec::new();
        let mut dirty = self.dirty_descriptors.lock();
        let mut allocator = self.descriptor_allocator.lock();

        let image_writes = TempList::new();
        let buffer_writes = TempList::new();
        let mut writes = Vec::with_capacity(DEFAULT_UPDATES);
        // Process all dirty descriptors
        for handle in dirty.iter().copied() {
            // Allocate and assing new descriptor set
            let data = if let Some(data) = descriptors.get_cold_mut(handle) {
                data
            } else {
                // Skip invalid descriptors
                continue;
            };
            let descriptor_set = unsafe {
                allocator.allocate(
                    AshDescriptorDevice::wrap(&self.raw),
                    &data.layout,
                    DescriptorSetLayoutCreateFlags::empty(),
                    &data.count,
                    1,
                )
            }?
            .remove(0);
            let ds = *descriptor_set.raw();
            // Remove old descriptor if any
            if let Some(descriptor) = data.descriptor.replace(descriptor_set) {
                drop_list.push(descriptor);
            }
            if let Some(name) = &data.name {
                self.set_object_name(ds, name);
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
                                            .ok_or(Error::InvalidImageHandle(image.data.image))?
                                            .get_or_create_view(&self.raw, image.data.desc)?,
                                    )
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
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
                                    .buffer(
                                        *buffers.get(buffer.data.handle).ok_or(
                                            Error::InvalidBufferHandle(buffer.data.handle),
                                        )?,
                                    )
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
                                    .buffer(
                                        *buffers.get(buffer.data.handle).ok_or(
                                            Error::InvalidBufferHandle(buffer.data.handle),
                                        )?,
                                    )
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
                                    .buffer(
                                        *buffers.get(buffer.data.buffer).ok_or(
                                            Error::InvalidBufferHandle(buffer.data.buffer),
                                        )?,
                                    )
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
                                    .buffer(
                                        *buffers.get(buffer.data.buffer).ok_or(
                                            Error::InvalidBufferHandle(buffer.data.buffer),
                                        )?,
                                    )
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
        unsafe { self.raw.update_descriptor_sets(&writes, &[]) };
        // Update actual vulkan objects for all dirty descriptors
        for handle in dirty.iter().copied() {
            if let Some(ds) = descriptors.get_cold(handle) {
                if let Some(ds) = &ds.descriptor {
                    descriptors.replace(handle, *ds.raw());
                }
            }
        }
        dirty.clear();
        self.current_drop_list.lock().drop_descriptors(drop_list);
        Ok(())
    }

    pub fn descriptors(&self) -> DescriptorUpdateContext {
        DescriptorUpdateContext {
            device: self,
            descriptors: self.descriptors.write(),
            dirty: self.dirty_descriptors.lock(),
            to_destroy: self.descriptors_to_destroy.lock(),
        }
    }
}
