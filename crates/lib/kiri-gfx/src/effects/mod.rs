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
mod basic;
pub use basic::*;

use kiri_backend::{
    ash::vk::{self},
    DescriptorSetDesc, DescriptorSetLayoutDesc, InputVertexStreamLayout, RenderPassLayout,
};
use kiri_math::Vec4;

use std::{collections::HashMap, fmt::Debug, sync::Arc};

use crate::{BufferSlice, DescriptorHandle, DescriptorSetBuilder, Error, PipelineHandle, Texture};

#[derive(Debug)]
pub struct EffectInstance {
    pub pipelines: HashMap<&'static str, PipelineHandle>,
    pub ds: DescriptorHandle,
    pub uniform: Option<BufferSlice>,
}

impl EffectInstance {
    pub fn pipeline(&self, name: &str) -> Option<PipelineHandle> {
        self.pipelines.get(name).copied()
    }
}

#[derive(Debug)]
pub struct EffectInstanceDesc {
    pub textures: HashMap<String, Arc<Texture>>,
    pub scalars: HashMap<String, f32>,
    pub vectors: HashMap<String, Vec4>,
}

pub trait Effect: Debug + Send + Sync {
    fn create_instance(&self, desc: &EffectInstanceDesc) -> Result<EffectInstance, Error>;
    fn free_instance(&self, instance: &mut EffectInstance);
}

pub trait MeshEffectFactory: Debug + Send + Sync {
    fn get_or_create(
        &self,
        name: &str,
        pass_layout: &'static RenderPassLayout<'static>,
        input_layout: &'static [InputVertexStreamLayout<'static>],
    ) -> Result<Option<Arc<dyn Effect>>, Error>;
}

pub fn fill_descriptor_with_textures(
    mut builder: DescriptorSetBuilder,
    desc: &DescriptorSetLayoutDesc,
    textures: &HashMap<String, Arc<Texture>>,
) -> Result<DescriptorSetBuilder, Error> {
    for (slot, desc) in desc.layout {
        if desc.ty == vk::DescriptorType::SAMPLED_IMAGE
            || desc.ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        {
            let texture = textures
                .get(desc.name)
                .ok_or(Error::TextureSlotNotFound(desc.name.to_owned()))?;
            builder = builder.bind_image(*slot, texture.image, vk::ImageAspectFlags::COLOR);
        }
    }
    Ok(builder)
}

pub const RENDER_PASS_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[(
        0,
        DescriptorSetDesc {
            name: "per_pass",
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            count: 1,
        },
    )],
    update_after_bind: false,
};

pub const INSTANCE_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[(
        0,
        DescriptorSetDesc {
            name: "instance",
            ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            count: 1,
        },
    )],
    update_after_bind: false,
};

pub const EFFECT_PASS_GBUFFER: &str = "gbuffer";
pub const EFFECT_PASS_GBUFFER_MASKED: &str = "gbuffer_masked";
pub const EFFECT_PASS_TRANSPARENT: &str = "transparent";
pub const EFFECT_PASS_OPAQUE: &str = "opaque";
pub const EFFECT_PASS_OPAQUE_MASKED: &str = "opaque_masked";
pub const EFFECT_PASS_SHADOW: &str = "shadow";
pub const EFFECT_PASS_SHADOW_MASKED: &str = "shadow_masked";
