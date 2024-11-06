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

use std::{collections::HashMap, sync::Arc};

use kiri_backend::{
    ash::vk::{self},
    DescriptorSetDesc, DescriptorSetLayoutDesc, InputVertexStreamLayout, RasterPipelineCreateDesc,
    RenderPassLayout, EMPTY_DESCRIPTOR_LAYOUT,
};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use crate::{
    effects::{INSTANCE_DESCRIPTOR_LAYOUT, RENDER_PASS_DESCRIPTOR_LAYOUT},
    uniforms::ConstUniformBuffer,
    DescriptorSetBuilder, Error, PipelineCache, PipelineHandle, RasterPipelineDesc,
};

use super::{
    fill_descriptor_with_textures, Effect, EffectInstance, EffectInstanceDesc, MeshEffectFactory,
    EFFECT_PASS_DEPTH, EFFECT_PASS_DEPTH_MASKED, EFFECT_PASS_OPAQUE, EFFECT_PASS_OPAQUE_MASKED,
    EFFECT_PASS_TRANSPARENT,
};

const BASIC_MATERIAL_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[
        (
            0,
            DescriptorSetDesc {
                name: "material",
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
            },
        ),
        (
            1,
            DescriptorSetDesc {
                name: "base_color",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            2,
            DescriptorSetDesc {
                name: "normals",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            3,
            DescriptorSetDesc {
                name: "metallic_roughness",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            4,
            DescriptorSetDesc {
                name: "occlusion",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            5,
            DescriptorSetDesc {
                name: "emissive",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
    ],
    update_after_bind: false,
};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct BasicMaterialUniform {
    pub emissive_power: f32,
    pub alpha_cut: f32,
}

#[derive(Debug)]
pub struct BasicEffect {
    cache: Arc<PipelineCache>,
    uniforms: ConstUniformBuffer<BasicMaterialUniform>,
    pipelines: HashMap<(&'static str, &'static [InputVertexStreamLayout<'static>]), PipelineHandle>,
}

const BASIC_EFFECT_SPEC_DISCARD_INDEX: u32 = 0;

impl BasicEffect {
    pub fn create(
        cache: &Arc<PipelineCache>,
        pass: &'static RenderPassLayout<'static>,
        input_layout: &'static [InputVertexStreamLayout<'static>],
    ) -> Result<Arc<dyn Effect>, Error> {
        let transparent = Self::create_pipeline(
            cache,
            "shaders/basic.vert",
            "shaders/basic.frag",
            pass,
            RasterPipelineCreateDesc::default()
                .premultiplied()
                .depth_write(false),
            input_layout,
            false,
        )?;
        let opaque = Self::create_pipeline(
            cache,
            "shaders/basic.vert",
            "shaders/basic.frag",
            pass,
            RasterPipelineCreateDesc::default()
                .depth_write(false)
                .depth_test(vk::CompareOp::EQUAL),
            input_layout,
            false,
        )?;
        let opaque_masked = Self::create_pipeline(
            cache,
            "shaders/basic.vert",
            "shaders/basic.frag",
            pass,
            RasterPipelineCreateDesc::default()
                .depth_write(false)
                .depth_test(vk::CompareOp::EQUAL),
            input_layout,
            true,
        )?;
        let depth = Self::create_pipeline(
            cache,
            "shaders/depth.vert",
            "shaders/depth.frag",
            pass,
            RasterPipelineCreateDesc::default(),
            input_layout,
            false,
        )?;
        let depth_masked = Self::create_pipeline(
            cache,
            "shaders/depth.vert",
            "shaders/depth.frag",
            pass,
            RasterPipelineCreateDesc::default(),
            input_layout,
            true,
        )?;
        Ok(Arc::new(BasicEffect {
            cache: cache.clone(),
            uniforms: ConstUniformBuffer::new(&cache.renderer),
            pipelines: [
                ((EFFECT_PASS_TRANSPARENT, input_layout), transparent),
                ((EFFECT_PASS_OPAQUE, input_layout), opaque),
                ((EFFECT_PASS_OPAQUE_MASKED, input_layout), opaque_masked),
                ((EFFECT_PASS_DEPTH, input_layout), depth),
                ((EFFECT_PASS_DEPTH_MASKED, input_layout), depth_masked),
            ]
            .into(),
        }))
    }

    fn create_pipeline(
        cache: &PipelineCache,
        vertex_shader: &str,
        fragment_shader: &str,
        pass: &'static RenderPassLayout<'static>,
        desc: RasterPipelineCreateDesc,
        input_layout: &'static [InputVertexStreamLayout<'static>],
        use_discard: bool,
    ) -> Result<PipelineHandle, Error> {
        let pipeline = cache.get_or_create_raster_pipeline(
            RasterPipelineDesc::new(
                vertex_shader,
                fragment_shader,
                pass,
                input_layout,
                &[
                    RENDER_PASS_DESCRIPTOR_LAYOUT,
                    EMPTY_DESCRIPTOR_LAYOUT,
                    BASIC_MATERIAL_DESCRIPTOR_LAYOUT,
                    INSTANCE_DESCRIPTOR_LAYOUT,
                ],
            )
            .pipeline_desc(desc)
            .specialization(BASIC_EFFECT_SPEC_DISCARD_INDEX, use_discard.into()),
        )?;
        Ok(pipeline)
    }
}

impl Effect for BasicEffect {
    fn create_instance(&self, desc: &EffectInstanceDesc) -> Result<EffectInstance, Error> {
        let data = BasicMaterialUniform {
            emissive_power: desc
                .scalars
                .get("emissive_power")
                .copied()
                .unwrap_or_default(),
            alpha_cut: desc.scalars.get("alpha_cut").copied().unwrap_or_default(),
        };
        let uniform = self.uniforms.allocate(data)?;
        let builder = DescriptorSetBuilder::new(
            vk::ShaderStageFlags::ALL_GRAPHICS,
            BASIC_MATERIAL_DESCRIPTOR_LAYOUT,
        )
        .bind_uniform_buffer(0, uniform);
        let builder = fill_descriptor_with_textures(
            builder,
            &BASIC_MATERIAL_DESCRIPTOR_LAYOUT,
            &desc.textures,
        )?;

        Ok(EffectInstance {
            pipelines: self
                .pipelines
                .iter()
                .map(|((name, _), pipeline)| (*name, *pipeline))
                .collect(),
            ds: self.cache.renderer.create_descriptor_set(builder)?,
            uniform: Some(uniform),
        })
    }

    fn free_instance(&self, instance: &mut EffectInstance) {
        if let Some(uniform) = instance.uniform.take() {
            self.uniforms.free(uniform);
        }
        self.cache.renderer.destroy_descriptor_set(instance.ds);
    }
}

type EffectKey = (
    &'static RenderPassLayout<'static>,
    &'static [InputVertexStreamLayout<'static>],
);

#[derive(Debug)]
pub struct BasicEffectFactory {
    cache: Arc<PipelineCache>,
    effects: RwLock<HashMap<EffectKey, Arc<dyn Effect>>>,
}

impl BasicEffectFactory {
    pub fn new(cache: &Arc<PipelineCache>) -> Self {
        Self {
            cache: cache.clone(),
            effects: Default::default(),
        }
    }
}

impl MeshEffectFactory for BasicEffectFactory {
    fn get_or_create(
        &self,
        _name: &str,
        pass_layout: &'static RenderPassLayout<'static>,
        input_layout: &'static [InputVertexStreamLayout<'static>],
    ) -> Result<Option<Arc<dyn Effect>>, Error> {
        let effects = self.effects.upgradable_read();
        if let Some(effect) = effects.get(&(pass_layout, input_layout)) {
            Ok(Some(effect.clone()))
        } else {
            let mut effects = RwLockUpgradableReadGuard::upgrade(effects);
            if let Some(effect) = effects.get(&(pass_layout, input_layout)) {
                Ok(Some(effect.clone()))
            } else {
                let effect = BasicEffect::create(&self.cache, pass_layout, input_layout)?;
                effects.insert((pass_layout, input_layout), effect.clone());
                Ok(Some(effect))
            }
        }
    }
}
