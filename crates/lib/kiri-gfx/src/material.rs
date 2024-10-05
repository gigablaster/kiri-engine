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

use std::{collections::HashMap, mem, sync::Arc};

use crate::{BufferSlice, DescriptorHandle, DescriptorSetBuilder, PipelineHandle, Renderer};
use byte_slice_cast::AsByteSlice;
use kiri_backend::{
    ash::vk::{self},
    DescriptorSetLayoutDesc,
};
use kiri_common::Align;
use kiri_math::Vec4;

use crate::{ConstUniformBuffer, Error, Texture};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderMaterialOrder {
    Opaque,
    Masked,
    Transparent,
}

#[derive(Debug, Clone)]
pub struct RenderMaterialUniformLayout<'a> {
    pub scalars: &'a [(&'a str, usize)],
    pub vectors: &'a [(&'a str, usize)],
}

impl<'a> RenderMaterialUniformLayout<'a> {
    pub fn count(&self) -> usize {
        let mut max = self
            .scalars
            .iter()
            .map(|(_, offset)| offset + 1)
            .max()
            .unwrap_or(0);
        max = max.max(
            self.vectors
                .iter()
                .map(|(_, offset)| offset + 4)
                .max()
                .unwrap_or(0),
        );
        max.align(4)
    }
}

#[derive(Debug, Clone)]
pub struct RenderMaterialInstanceDesc {
    pub textures: HashMap<String, Arc<Texture>>,
    pub scalars: HashMap<String, f32>,
    pub vectors: HashMap<String, Vec4>,
}

impl RenderMaterialInstanceDesc {
    pub fn write(&self, layout: &RenderMaterialUniformLayout) -> Vec<f32> {
        let count = layout.count();
        let mut data = vec![0f32; count];
        for (name, offset) in layout.scalars.iter().copied() {
            let value = self
                .scalars
                .get(name)
                .expect(&format!("Scalar value {} isn't found", name));
            Self::write_f32(&mut data, offset, *value);
        }
        for (name, offset) in layout.vectors.iter().copied() {
            let value = self
                .vectors
                .get(name)
                .expect(&format!("Vector value {} isn't found", name));
            Self::write_vec(&mut data, offset, *value);
        }

        data
    }

    fn write_f32(data: &mut [f32], offset: usize, value: f32) {
        data[offset] = value;
    }

    fn write_vec(data: &mut [f32], offset: usize, value: Vec4) {
        let values = value.to_array();
        for i in 0..4 {
            data[offset + i] = values[i];
        }
    }
}

pub trait RenderMaterial {
    fn create_instance(
        &self,
        desc: RenderMaterialInstanceDesc,
    ) -> Result<RenderMaterialInstance, Error>;
}

#[derive(Debug)]
pub struct RenderMaterialBase {
    renderer: Arc<Renderer>,
    uniform_layout: RenderMaterialUniformLayout<'static>,
    descriptor_layout: DescriptorSetLayoutDesc<'static>,
    depth: Option<PipelineHandle>,
    main: PipelineHandle,
    order: RenderMaterialOrder,
    unifroms: ConstUniformBuffer,
}

#[derive(Debug)]
pub struct RenderMaterialInstance {
    material: Arc<RenderMaterialBase>,
    pub desc: RenderMaterialInstanceDesc,
    pub ds: DescriptorHandle,
    pub uniform: Option<BufferSlice>,
}

impl Drop for RenderMaterialInstance {
    fn drop(&mut self) {
        if let Some(uniform) = self.uniform.take() {
            self.material.renderer.destroy_descriptor_set(self.ds);
            self.material.unifroms.free(uniform);
        }
    }
}

#[derive(Debug)]
pub struct RenderMaterialBuilder {
    pub uniform_layout: RenderMaterialUniformLayout<'static>,
    pub descriptor_layout: DescriptorSetLayoutDesc<'static>,
    pub order: RenderMaterialOrder,
    pub depth: Option<PipelineHandle>,
    pub main: PipelineHandle,
}

impl RenderMaterialBuilder {
    pub fn new(
        uniform_layout: RenderMaterialUniformLayout<'static>,
        descriptor_layout: DescriptorSetLayoutDesc<'static>,
        main: PipelineHandle,
    ) -> Self {
        Self {
            uniform_layout,
            descriptor_layout,
            order: RenderMaterialOrder::Opaque,
            depth: None,
            main,
        }
    }

    pub fn depth(mut self, pipeline: PipelineHandle) -> Self {
        self.depth = Some(pipeline);
        self
    }

    pub fn order(mut self, order: RenderMaterialOrder) -> Self {
        self.order = order;
        self
    }

    pub fn build(self, renderer: &Arc<Renderer>) -> Result<RenderMaterialBase, Error> {
        Ok(RenderMaterialBase {
            renderer: renderer.clone(),
            unifroms: ConstUniformBuffer::new(
                renderer,
                (self.uniform_layout.count() * mem::size_of::<f32>()) as _,
            ),
            uniform_layout: self.uniform_layout,
            descriptor_layout: self.descriptor_layout,
            depth: self.depth,
            main: self.main,
            order: self.order,
        })
    }
}

impl RenderMaterial for Arc<RenderMaterialBase> {
    fn create_instance(
        &self,
        desc: RenderMaterialInstanceDesc,
    ) -> Result<RenderMaterialInstance, Error> {
        let data = desc.write(&self.uniform_layout);
        let mut descriptor =
            DescriptorSetBuilder::new(vk::ShaderStageFlags::ALL_GRAPHICS, self.descriptor_layout);
        let uniform = if !data.is_empty() {
            let unifrom = self.unifroms.allocate(data.as_byte_slice())?;
            descriptor = descriptor.bind_uniform_buffer(
                self.descriptor_layout
                    .get_slot("material")
                    .expect("Binding named 'material' is a must"),
                unifrom,
            );
            Some(unifrom)
        } else {
            None
        };
        for (name, texture) in &desc.textures {
            if let Some(slot) = self.descriptor_layout.get_slot(&name) {
                descriptor = descriptor.bind_image(slot, texture.image, vk::ImageAspectFlags::COLOR)
            }
        }
        let ds = self.renderer.create_descriptor_set(descriptor)?;
        Ok(RenderMaterialInstance {
            material: self.clone(),
            desc,
            ds,
            uniform,
        })
    }
}
