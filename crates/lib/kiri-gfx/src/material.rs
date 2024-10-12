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

use std::{collections::HashMap, fmt::Debug, mem, sync::Arc};

use crate::{
    BufferSlice, DescriptorHandle, DescriptorSetBuilder, PipelineCache, PipelineHandle,
    RasterPipelineDesc, Renderer,
};
use byte_slice_cast::AsByteSlice;
use kiri_backend::{
    ash::vk::{self},
    DescriptorSetLayoutDesc, InputVertexStreamLayout, RasterPipelineCreateDesc, RenderPassLayout,
    MATERIAL_BINDING_SLOT,
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

pub trait RenderMaterial: Debug + Send + Sync {
    fn create_instance(
        &self,
        desc: RenderMaterialInstanceDesc,
    ) -> Result<RenderMaterialInstance, Error>;
}

#[derive(Debug)]
pub struct RenderMaterialBase {
    renderer: Arc<Renderer>,
    pub depth: Option<MaterialShader>,
    pub main: MaterialShader,
    pub order: RenderMaterialOrder,
}

#[derive(Debug)]
pub struct MaterialShader {
    pub pipeline: PipelineHandle,
    uniform_layout: RenderMaterialUniformLayout<'static>,
    descriptor_layout: DescriptorSetLayoutDesc<'static>,
    unifroms: ConstUniformBuffer,
}

#[derive(Debug)]
pub struct MaterialShaderDesc<'a> {
    pub vertex_shader: &'a str,
    pub fragment_shader: &'a str,
    pub render_pass: &'static RenderPassLayout<'static>,
    pub input_layout: &'static [InputVertexStreamLayout<'static>],
    pub descriptor_layout: &'static [DescriptorSetLayoutDesc<'static>],
    pub uniform_layout: RenderMaterialUniformLayout<'static>,
    pub raster_desc: RasterPipelineCreateDesc,
}

impl MaterialShader {
    fn new(cache: &PipelineCache, desc: MaterialShaderDesc) -> Result<Self, Error> {
        Ok(Self {
            pipeline: cache.get_or_create_raster_pipeline(
                RasterPipelineDesc::new(
                    desc.vertex_shader,
                    desc.fragment_shader,
                    desc.render_pass,
                    desc.input_layout,
                    desc.descriptor_layout,
                )
                .pipeline_desc(desc.raster_desc),
            )?,
            unifroms: ConstUniformBuffer::new(
                &cache.renderer,
                (desc.uniform_layout.count() * mem::size_of::<f32>()) as u64,
            ),
            uniform_layout: desc.uniform_layout,
            descriptor_layout: desc.descriptor_layout[MATERIAL_BINDING_SLOT],
        })
    }

    fn create_shader_instance(
        &self,
        renderer: &Renderer,
        desc: &RenderMaterialInstanceDesc,
    ) -> Result<MaterialShaderInstance, Error> {
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
            if let Some(slot) = self.descriptor_layout.get_slot(name) {
                descriptor = descriptor.bind_image(slot, texture.image, vk::ImageAspectFlags::COLOR)
            }
        }
        Ok(MaterialShaderInstance {
            ds: renderer.create_descriptor_set(descriptor)?,
            uniform,
        })
    }
}

#[derive(Debug)]
struct MaterialShaderInstance {
    ds: DescriptorHandle,
    uniform: Option<BufferSlice>,
}

impl MaterialShaderInstance {
    pub fn free(&mut self, renderer: &Renderer, shader: &MaterialShader) {
        if let Some(uniform) = self.uniform.take() {
            shader.unifroms.free(uniform);
        }
        renderer.destroy_descriptor_set(self.ds);
    }
}

pub struct RenderMaterialBuilder<'a> {
    pub main: MaterialShaderDesc<'a>,
    pub depth: Option<MaterialShaderDesc<'a>>,
    pub order: RenderMaterialOrder,
}

impl<'a> RenderMaterialBuilder<'a> {
    pub fn new(main: MaterialShaderDesc<'a>) -> Self {
        Self {
            main,
            depth: None,
            order: RenderMaterialOrder::Opaque,
        }
    }

    pub fn depth(mut self, depth: MaterialShaderDesc<'a>) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn order(mut self, order: RenderMaterialOrder) -> Self {
        self.order = order;
        self
    }

    pub fn build(self, cache: &Arc<PipelineCache>) -> Result<RenderMaterialBase, Error> {
        let depth = if let Some(depth) = self.depth {
            Some(MaterialShader::new(cache, depth)?)
        } else {
            None
        };
        Ok(RenderMaterialBase {
            renderer: cache.renderer.clone(),
            main: MaterialShader::new(cache, self.main)?,
            depth,
            order: self.order,
        })
    }
}

#[derive(Debug)]
pub struct RenderMaterialInstance {
    material: Arc<RenderMaterialBase>,
    main: MaterialShaderInstance,
    depth: Option<MaterialShaderInstance>,
}

impl Drop for RenderMaterialInstance {
    fn drop(&mut self) {
        self.main.free(&self.material.renderer, &self.material.main);
        if let Some(depth) = &self.material.depth {
            self.depth
                .iter_mut()
                .for_each(|x| x.free(&self.material.renderer, depth));
        }
    }
}

impl RenderMaterial for Arc<RenderMaterialBase> {
    fn create_instance(
        &self,
        desc: RenderMaterialInstanceDesc,
    ) -> Result<RenderMaterialInstance, Error> {
        let depth = if let Some(depth) = &self.depth {
            Some(depth.create_shader_instance(&self.renderer, &desc)?)
        } else {
            None
        };
        Ok(RenderMaterialInstance {
            material: self.clone(),
            main: self.main.create_shader_instance(&self.renderer, &desc)?,
            depth,
        })
    }
}
