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

use std::{slice, sync::Arc};

use kiri_backend::{
    ash::vk, Buffer, DescriptorSetLayoutDesc, DescriptorTotalCount, GpuDescriptor, Image,
    ImageViewDesc, RenderDevice,
};
use kiri_common::{Handle, HotColdPool, TempList};
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::Error;

pub type DescriptorHandle = Handle<vk::DescriptorSet>;
type DescriptorPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;

#[derive(Debug)]
pub struct Binding<T> {
    pub slot: u32,
    pub element: u32,
    pub ty: vk::DescriptorType,
    pub data: T,
}

#[derive(Debug)]
pub struct ImageBindingData {
    pub image: Arc<Image>,
    pub view: vk::ImageView,
}

#[derive(Debug)]
pub struct StaticBufferBindingData {
    pub buffer: Arc<Buffer>,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug)]
pub(super) struct DynamicBufferBindingData {
    pub buffer: Arc<Buffer>,
    pub size: u32,
}

#[derive(Debug)]
pub struct DescriptorSetData {
    pub descriptor: Option<GpuDescriptor>,
    pub count: DescriptorTotalCount,
    pub layout: vk::DescriptorSetLayout,
    pub images: Vec<Binding<ImageBindingData>>,
    pub unifom_buffers: Vec<Binding<StaticBufferBindingData>>,
    pub storage_buffers: Vec<Binding<StaticBufferBindingData>>,
    pub dynamic_uniform_buffers: Vec<Binding<DynamicBufferBindingData>>,
    pub dynamic_storage_buffers: Vec<Binding<DynamicBufferBindingData>>,
    pub name: Option<String>,
}

#[derive(Debug)]
pub struct DescriptorSetBuilder<'a> {
    layout: DescriptorSetLayoutDesc<'static>,
    stages: vk::ShaderStageFlags,
    images: Vec<Binding<ImageBindingData>>,
    unifom_buffers: Vec<Binding<StaticBufferBindingData>>,
    storage_buffers: Vec<Binding<StaticBufferBindingData>>,
    dynamic_uniform_buffers: Vec<Binding<DynamicBufferBindingData>>,
    dynamic_storage_buffers: Vec<Binding<DynamicBufferBindingData>>,
    name: Option<&'a str>,
}

impl<'a> DescriptorSetBuilder<'a> {
    pub fn new(stages: vk::ShaderStageFlags, layout: DescriptorSetLayoutDesc<'static>) -> Self {
        let count = layout.get_descriptor_count();
        Self {
            layout,
            stages,
            images: Vec::with_capacity((count.sampled_image + count.combined_image_sampler) as _),
            unifom_buffers: Vec::with_capacity(count.uniform_buffer as _),
            storage_buffers: Vec::with_capacity(count.storage_buffer as _),
            dynamic_uniform_buffers: Vec::with_capacity(count.uniform_buffer_dynamic as _),
            dynamic_storage_buffers: Vec::with_capacity(count.storage_buffer_dynamic as _),
            name: None,
        }
    }

    pub fn bind_image(mut self, slot: &str, image: Arc<Image>) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::TextureSlotNotFound(slot.to_owned()))?;
        self.images.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: ImageBindingData {
                view: image.view(ImageViewDesc::new(vk::ImageAspectFlags::COLOR))?,
                image,
            },
        });
        Ok(self)
    }

    pub fn bind_uniform_buffer(
        mut self,
        slot: &str,
        buffer: Arc<Buffer>,
        offset: usize,
        size: usize,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.unifom_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: StaticBufferBindingData {
                offset: offset as u32,
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn bind_storage_buffer(
        mut self,
        slot: &str,
        buffer: Arc<Buffer>,
        offset: usize,
        size: u32,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.storage_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: StaticBufferBindingData {
                offset: offset as u32,
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn bind_dynamic_uniform_buffer(
        mut self,
        slot: &str,
        buffer: Arc<Buffer>,
        size: usize,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.dynamic_uniform_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData {
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn bind_dynamic_storage_buffer(
        mut self,
        slot: &str,
        buffer: Arc<Buffer>,
        size: usize,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.dynamic_storage_buffers.push(Binding {
            slot: slot as u32,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData {
                size: size as u32,
                buffer,
            },
        });
        Ok(self)
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    fn build(self, device: &RenderDevice) -> Result<DescriptorSetData, Error> {
        Ok(DescriptorSetData {
            descriptor: None,
            count: self.layout.get_descriptor_count(),
            layout: device.get_or_create_layout(self.stages, self.layout)?,
            images: self.images,
            unifom_buffers: self.unifom_buffers,
            storage_buffers: self.storage_buffers,
            dynamic_uniform_buffers: self.dynamic_uniform_buffers,
            dynamic_storage_buffers: self.dynamic_storage_buffers,
            name: self.name.map(|x| x.to_owned()),
        })
    }
}

const MAX_DESCRIPTORS: usize = 16384;

#[derive(Debug)]
pub(super) struct DescriptorManager {
    device: Arc<RenderDevice>,
    descriptors: RwLock<DescriptorPool>,
    dirty: Mutex<Vec<DescriptorHandle>>,
    to_destroy: Mutex<Vec<DescriptorHandle>>,
}

#[derive(Debug)]
pub(super) struct DescriptorResolver<'a> {
    descriptors: RwLockReadGuard<'a, DescriptorPool>,
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
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let data = builder.build(self.device)?;
        let handle = self.descriptors.push(vk::DescriptorSet::null(), data);
        self.dirty.push(handle);
        Ok(handle)
    }

    pub fn update_descriptor(
        &mut self,
        handle: DescriptorHandle,
        builder: DescriptorSetBuilder,
    ) -> Result<(), Error> {
        let data = builder.build(self.device)?;
        if self.descriptors.replace_cold(handle, data).is_some() {
            self.dirty.push(handle);
        }
        Ok(())
    }

    pub fn destroy_descriptor(&mut self, handle: DescriptorHandle) {
        self.to_destroy.push(handle);
    }
}

impl DescriptorManager {
    pub fn new(device: Arc<RenderDevice>) -> Self {
        Self {
            device,
            descriptors: RwLock::new(HotColdPool::new(MAX_DESCRIPTORS)),
            dirty: Default::default(),
            to_destroy: Default::default(),
        }
    }

    pub fn create_descriptor(
        &self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let data = builder.build(&self.device)?;
        let handle = self
            .descriptors
            .write()
            .push(vk::DescriptorSet::null(), data);
        self.dirty.lock().push(handle);
        Ok(handle)
    }

    pub fn update_descriptor(
        &self,
        handle: DescriptorHandle,
        builder: DescriptorSetBuilder,
    ) -> Result<(), Error> {
        let data = builder.build(&self.device)?;
        self.descriptors.write().replace_cold(handle, data);
        self.dirty.lock().push(handle);
        Ok(())
    }

    pub fn destroy_descriptor(&self, handle: DescriptorHandle) {
        self.to_destroy.lock().push(handle);
    }

    pub fn update(&self) -> DescriptorUpdateContext {
        let descriptors = self.descriptors.write();
        let dirty = self.dirty.lock();
        let to_destroy = self.to_destroy.lock();
        DescriptorUpdateContext {
            device: &self.device,
            descriptors: descriptors,
            dirty: dirty,
            to_destroy: to_destroy,
        }
    }

    pub fn resolve(&self) -> DescriptorResolver {
        DescriptorResolver {
            descriptors: self.descriptors.read(),
        }
    }

    pub fn update_descriptors(&self) -> Result<(), Error> {
        puffin::profile_function!();
        let mut descriptors = self.descriptors.write();
        let mut drop_list = Vec::new();
        let mut dirty = self.dirty.lock();
        for handle in self.to_destroy.lock().drain(..) {
            if let Some((_, mut data)) = descriptors.remove(handle) {
                if let Some(descriptor) = data.descriptor.take() {
                    drop_list.push(descriptor);
                }
            }
        }
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
                                            .image_view(image.data.view)
                                            .image_layout(
                                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                            ),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(image.ty)
                                .dst_array_element(image.element)
                                .dst_binding(image.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process uniform buffers
                    for buffer in &data.unifom_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(buffer.data.buffer.raw)
                                            .offset(buffer.data.offset as _)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process storage buffers
                    for buffer in &data.storage_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(buffer.data.buffer.raw)
                                            .offset(buffer.data.offset as _)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process dynamic uniform buffers
                    for buffer in &data.dynamic_uniform_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(buffer.data.buffer.raw)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
                                .dst_array_element(buffer.element)
                                .dst_binding(buffer.slot)
                                .dst_set(ds),
                        );
                    }
                    // Process dynamic storage buffers
                    for buffer in &data.dynamic_storage_buffers {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .buffer_info(slice::from_ref(
                                    buffer_writes.add(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(buffer.data.buffer.raw)
                                            .range(buffer.data.size as _),
                                    ),
                                ))
                                .descriptor_count(1)
                                .descriptor_type(buffer.ty)
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
            let ds = *descriptors
                .get_cold(handle)
                .unwrap()
                .descriptor
                .as_ref()
                .unwrap()
                .raw();
            descriptors.replace(handle, ds);
        }
        dirty.clear();
        self.device.drop_descriptors(drop_list);
        Ok(())
    }
}
