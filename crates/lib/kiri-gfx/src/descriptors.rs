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

use kiri_backend::{ash::vk, DescriptorSetLayoutDesc, DescriptorTotalCount, RenderDevice};

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
    layout: &'a DescriptorSetLayoutDesc,
    stages: vk::ShaderStageFlags,
    images: Vec<Binding<ImageBindingData>>,
    unifom_buffers: Vec<Binding<StaticBufferBindingData>>,
    storage_buffers: Vec<Binding<StaticBufferBindingData>>,
    dynamic_uniform_buffers: Vec<Binding<DynamicBufferBindingData>>,
    dynamic_storage_buffers: Vec<Binding<DynamicBufferBindingData>>,
    name: Option<&'a str>,
}

impl<'a> DescriptorSetBuilder<'a> {
    pub fn new(stages: vk::ShaderStageFlags, layout: &'a DescriptorSetLayoutDesc) -> Self {
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

    pub fn bind_image(
        mut self,
        slot: &str,
        image: ImageHandle,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::TextureSlotNotFound(slot.to_owned()))?;
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

    pub fn bind_uniform_buffer(mut self, slot: &str, buffer: BufferSlice) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
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

    pub fn bind_storage_buffer(mut self, slot: &str, buffer: BufferSlice) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
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
        slot: &str,
        handle: BufferHandle,
        size: u64,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
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
        slot: &str,
        handle: BufferHandle,
        size: u64,
    ) -> Result<Self, Error> {
        let slot = self
            .layout
            .get_slot(slot)
            .ok_or(Error::BindingSlotNotFound(slot.to_owned()))?;
        self.dynamic_storage_buffers.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData { handle, size },
        });
        Ok(self)
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
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
            name: self.name.map(|x| x.to_owned()),
        })
    }
}
