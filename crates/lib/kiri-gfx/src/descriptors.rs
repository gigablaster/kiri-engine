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

use kiri_backend::{ash::vk, DescriptorSetCount, DescriptorSetLayoutDesc, RenderDevice};

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
    pub count: DescriptorSetCount,
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
            images: Vec::with_capacity((count.sampled_images + count.combined_image_samplers) as _),
            unifom_buffers: Vec::with_capacity(count.unifroms_buffers as _),
            storage_buffers: Vec::with_capacity(count.storage_buffers as _),
            dynamic_uniform_buffers: Vec::with_capacity(count.dynamic_uniform_buffers as _),
            dynamic_storage_buffers: Vec::with_capacity(count.dynamic_storage_buffers as _),
            name: None,
        }
    }

    pub fn bind_image(
        mut self,
        slot: u32,
        image: ImageHandle,
        aspect: vk::ImageAspectFlags,
    ) -> Self {
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
        self
    }

    pub fn bind_uniform_buffer(mut self, slot: u32, buffer: BufferSlice) -> Self {
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
        self
    }

    pub fn bind_storage_buffer(mut self, slot: u32, buffer: BufferSlice) -> Self {
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
        self
    }

    pub fn bind_dynamic_uniform_buffer(
        mut self,
        slot: u32,
        handle: BufferHandle,
        size: u64,
    ) -> Self {
        self.dynamic_uniform_buffers.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData { handle, size },
        });
        self
    }

    pub fn bind_dynamic_storage_buffer(
        mut self,
        slot: u32,
        handle: BufferHandle,
        size: u64,
    ) -> Self {
        self.dynamic_storage_buffers.push(Binding {
            slot,
            element: 0,
            ty: self.layout.get_desc(slot).unwrap().ty,
            data: DynamicBufferBindingData { handle, size },
        });
        self
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
