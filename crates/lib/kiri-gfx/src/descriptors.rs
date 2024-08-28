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

use ash::vk;
use kiri_backend::{DescriptorCount, DescriptorSetLayoutDesc, Program, RenderDevice};

use crate::{BufferHandle, BufferSlice, Error, ImageHandle};

#[derive(Debug, Clone, Copy)]
pub(super) struct Binding<T: Copy> {
    pub slot: u32,
    pub element: u32,
    pub ty: vk::DescriptorType,
    pub data: T,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ImageBindingData {
    pub handle: ImageHandle,
    pub aspect: vk::ImageAspectFlags,
    pub view: vk::ImageView,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct StaticBufferBindingData {
    pub handle: BufferHandle,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DynamicBufferBindingData {
    pub handle: BufferHandle,
    pub size: u64,
}

#[derive(Debug)]
pub(super) struct DescriptorSetData {
    pub count: DescriptorCount,
    pub layout: vk::DescriptorSetLayout,
    pub images: Vec<Binding<ImageBindingData>>,
    pub unifom_buffers: Vec<Binding<StaticBufferBindingData>>,
    pub storage_buffers: Vec<Binding<StaticBufferBindingData>>,
    pub dynamic_uniform_buffers: Vec<Binding<DynamicBufferBindingData>>,
    pub dynamic_storage_buffers: Vec<Binding<DynamicBufferBindingData>>,
}

#[derive(Debug)]
pub struct DescriptorSetBuilder<'a> {
    layout: &'a DescriptorSetLayoutDesc,
    stages: vk::ShaderStageFlags,
    images: Vec<Binding<ImageBindingData>>,
    unifom_buffers: Vec<Binding<StaticBufferBindingData>>,
    storage_buffers: Vec<Binding<StaticBufferBindingData>>,
    dynamic_uniform_buffers: Vec<Binding<DynamicBufferBindingData>>,
    dynamic_storage_buffers: Vec<Binding<DynamicBufferBindingData>>,
}

impl<'a> DescriptorSetBuilder<'a> {
    pub fn new(stages: vk::ShaderStageFlags, layout: &'a DescriptorSetLayoutDesc) -> Self {
        let count = layout.get_descriptor_count();
        Self {
            layout,
            stages,
            images: Vec::with_capacity((count.sampled_images + count.combined_image_samplers) as _),
            unifom_buffers: Vec::with_capacity(count.unifroms_buffers as _),
            storage_buffers: Vec::with_capacity(count.storage_buffers as _),
            dynamic_uniform_buffers: Vec::with_capacity(count.dynamic_uniform_buffers as _),
            dynamic_storage_buffers: Vec::with_capacity(count.dynamic_storage_buffers as _),
        }
    }

    pub fn from_program(program: &'a Program, binding: usize) -> Self {
        Self::new(program.stages, &program.desc[binding])
    }

    pub fn bind_image(
        mut self,
        slot: BindingSlot,
        image: ImageHandle,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, Error> {
        let slot = slot.resolve(&self.layout)?;
        self.images.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: ImageBindingData {
                handle: image,
                aspect,
                view: vk::ImageView::null(),
            },
        });
        Ok(self)
    }

    pub fn bind_uniform_buffer(mut self, slot: BindingSlot, buffer: BufferSlice) -> Result<Self, Error> {
        let slot = slot.resolve(&self.layout)?;
        self.unifom_buffers.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: StaticBufferBindingData {
                handle: buffer.handle,
                offset: buffer.offset,
                size: buffer.size,
            },
        });
        Ok(self)
    }

    pub fn bind_storage_buffer(mut self, slot: BindingSlot, buffer: BufferSlice) ->Result<Self, Error> {
        let slot = slot.resolve(&self.layout)?;
        self.storage_buffers.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: StaticBufferBindingData {
                handle: buffer.handle,
                offset: buffer.offset,
                size: buffer.size,
            },
        });
        Ok(self)
    }

    pub fn bind_dynamic_uniform_buffer(
        mut self,
        slot: BindingSlot,
        handle: BufferHandle,
        size: u64,
    ) -> Result<Self, Error> {
        let slot = slot.resolve(&self.layout)?;
        self.dynamic_uniform_buffers.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData { handle, size },
        });
        Ok(self)
    }

    pub fn bind_dynamic_storage_buffer(
        mut self,
        slot: BindingSlot,
        handle: BufferHandle,
        size: u64,
    ) -> Result<Self, Error> {
        let slot = slot.resolve(&self.layout)?;
        self.dynamic_storage_buffers.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData { handle, size },
        });
        Ok(self)
    }

    pub(super) fn build(self, device: &RenderDevice) -> Result<DescriptorSetData, Error> {
        Ok(DescriptorSetData {
            count: self.layout.get_descriptor_count(),
            layout: device.get_or_create_layout(self.stages, self.layout)?,
            images: self.images,
            unifom_buffers: self.unifom_buffers,
            storage_buffers: self.storage_buffers,
            dynamic_uniform_buffers: self.dynamic_uniform_buffers,
            dynamic_storage_buffers: self.dynamic_storage_buffers,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BindingSlot<'a> {
    Index(u32),
    Name(&'a str),
}

impl<'a> BindingSlot<'a> {
    pub fn resolve(self, layout: &DescriptorSetLayoutDesc) -> Result<u32, Error> {
        match self {
            BindingSlot::Index(index) => {
                if !layout.has_slot(index) {
                    Err(Error::InvalidDescriptorSlotIndex(index))
                } else {
                    Ok(index)
                }
            }
            BindingSlot::Name(name) => layout
                .get_slot_by_name(name)
                .ok_or(Error::InvalidDescriptorSlotName(name.to_owned())),
        }
    }
}

pub trait SkipMissingSlots {
    fn skip_missing_losts(self) -> Result<(), Error>;
}

impl SkipMissingSlots for Result<(), Error> {
    fn skip_missing_losts(self)  -> Result<(), Error>{
        match  self {
            Ok(_) => Ok(()),
            Err(Error::InvalidDescriptorSlotIndex(_)) => Ok(()),
            Err(Error::InvalidDescriptorSlotName(_)) => Ok(()),
            Err(err) => Err(err)
        }
    }
}
