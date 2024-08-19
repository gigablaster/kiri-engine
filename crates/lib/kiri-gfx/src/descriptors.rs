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

use core::slice;
use std::{collections::HashMap, sync::Arc};

use ash::vk::{self};
use gpu_descriptor::DescriptorSetLayoutCreateFlags;
use kiri_backend::{DescriptorSetLayout, DescriptorSetLayoutDesc};
use kiri_common::{Handle, HotColdPool, SentinelPoolStrategy, TempList};

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptorSetSlot<'a> {
    Name(&'a str),
    Index(usize),
}

impl<'a> DescriptorSetSlot<'a> {
    fn get_slot<T: Copy>(
        self,
        bindings: &[Binding<T>],
        layout: &DescriptorSetLayout,
    ) -> Result<usize, Error> {
        let slot = match self {
            DescriptorSetSlot::Name(name) => layout
                .slot_by_name(name)
                .ok_or(Error::SlotWithNameNotFound(name.to_owned()))?,
            DescriptorSetSlot::Index(index) => index,
        };
        bindings
            .iter()
            .enumerate()
            .find_map(|(index, x)| if x.slot == slot { Some(index) } else { None })
            .ok_or(Error::SlotWithIndexNotFound(slot))
    }
}

#[derive(Debug)]
pub struct DescriptorSetBuilder<'a, 'b> {
    pub layout: &'a DescriptorSetLayoutDesc<'a>,
    pub images: HashMap<DescriptorSetSlot<'b>, (ImageHandle, vk::ImageAspectFlags)>,
    pub uniform_buffers: HashMap<DescriptorSetSlot<'b>, (BufferSlice, usize)>,
    pub storage_buffers: HashMap<DescriptorSetSlot<'b>, (BufferSlice, usize)>,
    pub dynamic_uniform_buffers: HashMap<DescriptorSetSlot<'b>, (BufferHandle, usize)>,
    pub dynamic_storage_buffers: HashMap<DescriptorSetSlot<'b>, (BufferHandle, usize)>,
}

impl<'a, 'b> DescriptorSetBuilder<'a, 'b> {
    pub fn new(layout: &'a DescriptorSetLayoutDesc<'a>) -> Self {
        Self {
            layout,
            images: Default::default(),
            uniform_buffers: Default::default(),
            storage_buffers: Default::default(),
            dynamic_storage_buffers: Default::default(),
            dynamic_uniform_buffers: Default::default(),
        }
    }

    pub fn image(
        mut self,
        slot: DescriptorSetSlot<'b>,
        image: ImageHandle,
        aspect: vk::ImageAspectFlags,
    ) -> Self {
        self.images.insert(slot, (image, aspect));
        self
    }

    pub fn uniform_buffer(
        mut self,
        slot: DescriptorSetSlot<'b>,
        buffer: BufferSlice,
        size: usize,
    ) -> Self {
        self.uniform_buffers.insert(slot, (buffer, size));
        self
    }

    pub fn storage_buffer(
        mut self,
        slot: DescriptorSetSlot<'b>,
        buffer: BufferSlice,
        size: usize,
    ) -> Self {
        self.storage_buffers.insert(slot, (buffer, size));
        self
    }

    pub fn dynamic_uniform_buffer(
        mut self,
        slot: DescriptorSetSlot<'b>,
        buffer: BufferHandle,
        size: usize,
    ) -> Self {
        self.dynamic_uniform_buffers.insert(slot, (buffer, size));
        self
    }

    pub fn dynamic_storage_buffer(
        mut self,
        slot: DescriptorSetSlot<'b>,
        buffer: BufferHandle,
        size: usize,
    ) -> Self {
        self.dynamic_storage_buffers.insert(slot, (buffer, size));
        self
    }
}

#[derive(Debug, Clone, Copy)]
struct Binding<T: Copy> {
    slot: usize,
    ty: vk::DescriptorType,
    data: Option<T>,
}

impl<T: Copy> Binding<T> {
    pub fn new(slot: u32, ty: vk::DescriptorType) -> Self {
        Self {
            slot,
            ty,
            data: None,
        }
    }
}

impl<T: Copy> Binding<T> {
    pub fn validate(&self) -> Result<(), Error> {
        if self.data.is_none() {
            Err(Error::EmptyDescriptorSetSlot(self.slot as _))
        } else {
            Ok(())
        }
    }
}

type ImageBinding = Binding<(ImageHandle, vk::ImageAspectFlags)>;
type BufferBinding = Binding<(BufferHandle, u64, u64)>;
type DynamicBufferBinding = Binding<(BufferHandle, u64)>;

#[derive(Debug)]
struct DescriptorSetBindData {
    pub set: Option<GpuDescriptor>,
    layout: Arc<DescriptorSetLayout>,
    images: Vec<ImageBinding>,
    uniform_buffers: Vec<BufferBinding>,
    storage_buffers: Vec<BufferBinding>,
    dynamic_uniform_buffers: Vec<DynamicBufferBinding>,
    dynamic_storage_buffers: Vec<DynamicBufferBinding>,
}

impl DescriptorSetBindData {
    pub fn new<'a>(layout: Arc<DescriptorSetLayout>) -> Self {
        let images = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::SAMPLED_IMAGE
                    || *ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    || *ty == vk::DescriptorType::INPUT_ATTACHMENT
                {
                    Some(ImageBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let uniform_buffers = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER {
                    Some(BufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let storage_buffers = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::STORAGE_BUFFER {
                    Some(BufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let dynamic_uniform_buffers = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                    Some(DynamicBufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let dynamic_storage_buffers = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::STORAGE_BUFFER_DYNAMIC {
                    Some(DynamicBufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        Self {
            set: None,
            layout,
            images,
            uniform_buffers,
            dynamic_uniform_buffers,
            storage_buffers,
            dynamic_storage_buffers,
        }
    }

    pub fn free(mut self, drop_list: &mut DropList) {
        if let Some(set) = self.set.take() {
            drop_list.drop_descriptor(set);
        }
    }

    pub fn bind_image(
        &mut self,
        slot: DescriptorSetSlot,
        image: ImageHandle,
        aspect: vk::ImageAspectFlags,
    ) -> Result<(), Error> {
        let slot = slot.get_slot(&self.images, &self.layout)?;
        self.images[slot].data = Some((image, aspect));
        Ok(())
    }

    pub fn bind_uniform_buffer(
        &mut self,
        slot: DescriptorSetSlot,
        buffer: BufferSlice,
        size: usize,
    ) -> Result<(), Error> {
        let slot = slot.get_slot(&self.uniform_buffers, &self.layout)?;
        self.uniform_buffers[slot].data = Some((buffer.0, buffer.1 as u64, size as u64));
        Ok(())
    }

    pub fn bind_storage_buffer(
        &mut self,
        slot: DescriptorSetSlot,
        buffer: BufferSlice,
        size: usize,
    ) -> Result<(), Error> {
        let slot = slot.get_slot(&self.storage_buffers, &self.layout)?;
        self.storage_buffers[slot].data = Some((buffer.0, buffer.1 as u64, size as u64));
        Ok(())
    }

    pub fn bind_dynamic_uniform_buffer(
        &mut self,
        slot: DescriptorSetSlot,
        buffer: BufferHandle,
        size: usize,
    ) -> Result<(), Error> {
        let slot = slot.get_slot(&self.dynamic_uniform_buffers, &self.layout)?;
        self.dynamic_uniform_buffers[slot].data = Some((buffer, size as u64));
        Ok(())
    }

    pub fn bind_dynamic_storage_buffer(
        &mut self,
        slot: DescriptorSetSlot,
        buffer: BufferHandle,
        size: usize,
    ) -> Result<(), Error> {
        let slot = slot.get_slot(&self.dynamic_storage_buffers, &self.layout)?;
        self.dynamic_storage_buffers[slot].data = Some((buffer, size as u64));
        Ok(())
    }

    pub fn bind<'a>(&mut self, builder: DescriptorSetBuilder<'static, 'a>) -> Result<(), Error> {
        builder
            .images
            .into_iter()
            .try_for_each(|(slot, (image, aspect))| self.bind_image(slot, image, aspect))?;
        builder
            .uniform_buffers
            .into_iter()
            .try_for_each(|(slot, (buffer, size))| self.bind_uniform_buffer(slot, buffer, size))?;
        builder
            .storage_buffers
            .into_iter()
            .try_for_each(|(slot, (buffer, size))| self.bind_storage_buffer(slot, buffer, size))?;
        builder
            .dynamic_uniform_buffers
            .into_iter()
            .try_for_each(|(slot, (buffer, size))| {
                self.bind_dynamic_uniform_buffer(slot, buffer, size)
            })?;
        builder
            .dynamic_storage_buffers
            .into_iter()
            .try_for_each(|(slot, (buffer, size))| {
                self.bind_dynamic_storage_buffer(slot, buffer, size)
            })?;
        self.images.iter().try_for_each(Binding::validate)?;
        self.uniform_buffers
            .iter()
            .try_for_each(Binding::validate)?;
        self.storage_buffers
            .iter()
            .try_for_each(Binding::validate)?;
        self.dynamic_uniform_buffers
            .iter()
            .try_for_each(Binding::validate)?;
        self.dynamic_storage_buffers
            .iter()
            .try_for_each(Binding::validate)?;
        Ok(())
    }
}

pub type DescriptorSetHandle = Handle<vk::DescriptorSet>;
type DescriptorSetPool =
    HotColdPool<vk::DescriptorSet, DescriptorSetBindData, SentinelPoolStrategy<vk::DescriptorSet>>;

#[derive(Debug, Default)]
pub struct DescriptorSetManager {
    descriptors: DescriptorSetPool,
    dirty_descriptor_handles: Vec<DescriptorSetHandle>,
    layouts: HashMap<DescriptorSetLayoutDesc<'static>, Arc<DescriptorSetLayout>>,
}

impl DescriptorSetManager {
    pub fn free(&mut self, device: &ash::Device, drop_list: &mut DropList) {
        self.layouts.drain().for_each(|x| x.1.free(device));
        self.descriptors
            .drain()
            .for_each(|(_, data)| data.free(drop_list));
    }

    pub fn create_descriptor_set<'a>(
        &mut self,
        device: &ash::Device,
        samplers: &HashMap<SamplerDesc, vk::Sampler>,
        builder: DescriptorSetBuilder<'static, 'a>,
    ) -> Result<DescriptorSetHandle, Error> {
        let layout = self.get_or_create_layout(device, samplers, &builder.layout)?;
        self.create_descriptor_set_from_layout(layout, builder)
    }

    fn create_descriptor_set_from_layout<'a>(
        &mut self,
        layout: Arc<DescriptorSetLayout>,
        builder: DescriptorSetBuilder<'static, 'a>,
    ) -> Result<DescriptorSetHandle, Error> {
        let mut data = DescriptorSetBindData::new(layout);
        data.bind(builder)?;
        let handle = self
            .descriptors
            .push(<vk::DescriptorSet as vk::Handle>::from_raw(u64::MAX), data);
        self.dirty_descriptor_handles.push(handle);
        Ok(handle)
    }

    pub fn destroy_descriptor_set(
        &mut self,
        drop_list: &mut DropList,
        handle: DescriptorSetHandle,
    ) {
        if let Some((_, data)) = self.descriptors.remove(handle) {
            data.free(drop_list);
        }
    }

    pub(super) fn get_or_create_layout(
        &mut self,
        device: &ash::Device,
        samplers: &HashMap<SamplerDesc, vk::Sampler>,
        layout: &'static DescriptorSetLayoutDesc<'static>,
    ) -> Result<Arc<DescriptorSetLayout>, Error> {
        if let Some(layout) = self.layouts.get(layout).cloned() {
            Ok(layout)
        } else {
            let result = create_descriptor_set_layout(device, samplers, layout)?;
            self.layouts.insert(*layout, result.clone());
            Ok(result)
        }
    }

    pub fn resolve(&self, handle: DescriptorSetHandle) -> Result<vk::DescriptorSet, Error> {
        self.descriptors
            .get(handle)
            .copied()
            .ok_or(Error::InvalidDescriptorSetHandle(handle))
    }

    pub(super) fn update_descriptors(
        &mut self,
        device: &ash::Device,
        drop_list: &mut DropList,
        allocator: &mut GpuDescriptorAllocator,
        images: &ImagePool,
        buffers: &BufferPool,
    ) -> Result<(), Error> {
        puffin::profile_function!();
        self.dirty_descriptor_handles.dedup();
        let mut to_destroy = Vec::new();
        // Remove old descriptors
        self.dirty_descriptor_handles.iter().for_each(|handle| {
            if let Some(data) = self.descriptors.get_cold_mut(*handle) {
                if let Some(set) = data.set.take() {
                    to_destroy.push(set);
                }
            }
        });
        to_destroy.drain(..).for_each(|x| {
            drop_list.drop_descriptor(x);
        });
        // Allocate new descriptors
        self.dirty_descriptor_handles
            .iter()
            .try_for_each(|handle| -> Result<(), Error> {
                let set = if let Some(data) = self.descriptors.get_cold_mut(*handle) {
                    let set = unsafe {
                        allocator.allocate(
                            AshDescriptorDevice::wrap(device),
                            &data.layout.raw,
                            DescriptorSetLayoutCreateFlags::empty(),
                            &data.layout.count,
                            1,
                        )
                    }?
                    .remove(0);
                    let raw = *set.raw();

                    data.set = Some(set);
                    Some(raw)
                } else {
                    None
                };
                if let Some(set) = set {
                    self.descriptors.replace(*handle, set);
                }
                Ok(())
            })?;
        // Bind everything
        let mut writes = Vec::with_capacity(8192);
        let image_writes = TempList::new();
        let buffer_writes = TempList::new();
        for handle in &self.dirty_descriptor_handles {
            let data = self
                .descriptors
                .get_cold(*handle)
                .ok_or(Error::InvalidDescriptorSetHandle(*handle))?;
            let descriptor = if let Some(set) = &data.set {
                *set.raw()
            } else {
                unreachable!();
            };
            data.images.iter().try_for_each(|x| -> Result<(), Error> {
                let (handle, aspect) = x.data.unwrap();
                let image = images
                    .get_cold(handle)
                    .ok_or(Error::InvalidImageHandle(handle))?;
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .image_info(slice::from_ref(
                            image_writes.add(
                                vk::DescriptorImageInfo::default()
                                    .image_view(image.view(device, ImageViewDesc::new(aspect))?)
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                            ),
                        ))
                        .descriptor_count(1)
                        .descriptor_type(x.ty)
                        .dst_binding(x.slot)
                        .dst_set(descriptor),
                );
                Ok(())
            })?;
            data.uniform_buffers
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (handle, offset, range) = x.data.unwrap();

                    let buffer = buffers
                        .get(handle)
                        .copied()
                        .ok_or(Error::InvalidBufferHandle(handle))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(offset)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
            data.storage_buffers
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (handle, offset, range) = x.data.unwrap();
                    let buffer = buffers
                        .get(handle)
                        .copied()
                        .ok_or(Error::InvalidBufferHandle(handle))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(offset)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
            data.dynamic_uniform_buffers
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (handle, range) = x.data.unwrap();
                    let buffer = buffers
                        .get(handle)
                        .copied()
                        .ok_or(Error::InvalidBufferHandle(handle))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(0)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
            data.dynamic_storage_buffers
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (handle, range) = x.data.unwrap();
                    let buffer = buffers
                        .get(handle)
                        .copied()
                        .ok_or(Error::InvalidBufferHandle(handle))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(0)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
        }
        // And now update all descriptors
        unsafe {
            puffin::profile_scope!("Update_descriptor_sets");
            device.update_descriptor_sets(&writes, &[])
        };
        Ok(())
    }

    pub fn image_updated(&mut self, image: ImageHandle) {
        self.descriptors.enumerate().for_each(|(handle, _, data)| {
            if data.images.iter().any(|x| {
                if let Some((image_handle, _)) = x.data {
                    image_handle == image
                } else {
                    false
                }
            }) {
                self.dirty_descriptor_handles.push(handle);
            }
        });
    }
}
