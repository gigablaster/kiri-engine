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
use parking_lot::RwLockWriteGuard;

use crate::Error;

use super::{
    buffer::BufferPool, image::ImagePool, BufferHandle, BufferSlice, DescriptorLayoutDesc,
    DescriptorSetCreateDesc, GpuDescriptor, ImageHandle, ImageViewDesc, RenderDevice,
};

pub type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;

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
pub struct DescriptorSetData {
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

impl DescriptorSetCreateDesc<'_> {
    pub fn build(self, device: &RenderDevice) -> Result<DescriptorSetData, Error> {
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

impl RenderDevice {
    pub fn update_descriptors(
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
}
