// Copyright (C) 2025 gigablaster

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

use std::sync::Arc;

use kiri_assets::STATIC_MESH_VERTEX_LAYOUT;
use kiri_backend::{
    ash::vk,
    vulkan::{
        BufferSlice, DescriptorDesc, DescriptorHandle, DescriptorLayoutDesc,
        DescriptorSetCreateData, GraphicsDevice, ImageHandle, RasterPipelineCreateDesc,
        RasterPipelineHandle, EMPTY_DESCRIPTOR_LAYOUT,
    },
};

use crate::{pipeline_cache::PipelineCache, Error, ShaderUniforms};

use super::{
    Material, MaterialRenderData, RenderGroup, INSTANCE_DESCRIPTOR_LAYOUT, MAIN_RENDER_PASS_LAYOUT,
    SCENE_DESCRIPTOR_LAYOUT, ZPASS_RENDER_PASS_LAYOUT,
};

#[derive(Debug, Clone, Copy)]
pub enum BasicMaterialType {
    Opaque,
    Masked(f32),
    Transparent,
}

impl From<BasicMaterialType> for RenderGroup {
    fn from(value: BasicMaterialType) -> Self {
        match value {
            BasicMaterialType::Opaque => RenderGroup::Opaque,
            BasicMaterialType::Masked(_) => RenderGroup::Masked,
            BasicMaterialType::Transparent => RenderGroup::Transparent,
        }
    }
}

impl BasicMaterialType {
    pub fn alpha_cut(&self) -> f32 {
        match self {
            Self::Masked(alpha_cut) => *alpha_cut,
            _ => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BasicMaterialBuilder {
    pub ty: BasicMaterialType,
    pub emissive_power: f32,
    pub basic_color: ImageHandle,
    pub metallic_roughnbess: ImageHandle,
    pub normals: ImageHandle,
    pub occlusion: ImageHandle,
    pub emissive: ImageHandle,
}

/// Basic PBR material
#[derive(Debug)]
pub struct BasicMaterial {
    device: Arc<GraphicsDevice>,
    uniforms: Arc<ShaderUniforms>,
    group: RenderGroup,
    main: RasterPipelineHandle,
    depth: RasterPipelineHandle,
    descriptor: DescriptorHandle,
    uniform: BufferSlice,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BasicMaterialGpuData {
    pub alpha_cut: f32,
    pub emissive_power: f32,
}

impl BasicMaterialBuilder {
    pub fn build(
        self,
        device: Arc<GraphicsDevice>,
        uniforms: Arc<ShaderUniforms>,
        cache: &PipelineCache,
    ) -> Result<BasicMaterial, Error> {
        let uniform = uniforms.allocate(BasicMaterialGpuData {
            alpha_cut: self.ty.alpha_cut(),
            emissive_power: self.emissive_power,
        })?;
        let descriptor = device
            .descriptors()
            .create_descriptor(DescriptorSetCreateData {
                layout: BASIC_MATERIAL_DESCRIPTOR_LAYOUT,
                stages: vk::ShaderStageFlags::ALL_GRAPHICS,
                images: &[
                    self.basic_color,
                    self.metallic_roughnbess,
                    self.normals,
                    self.occlusion,
                    self.emissive,
                ],
                unifoms: &[uniform],
                ..Default::default()
            })?;
        let (main, depth) = match self.ty {
            BasicMaterialType::Opaque => (
                cache.get_or_create_raster_pipeline(
                    "shaders/basic",
                    "shaders/basic",
                    &MAIN_RENDER_PASS_LAYOUT,
                    &BASIC_MATERIAL_DESCRIPTOR_SET_LAYOUT,
                    &STATIC_MESH_VERTEX_LAYOUT,
                    &[],
                    RasterPipelineCreateDesc::default().depth_test(vk::CompareOp::EQUAL),
                )?,
                cache.get_or_create_raster_pipeline(
                    "shaders/depth",
                    "shaders/depth",
                    &ZPASS_RENDER_PASS_LAYOUT,
                    &BASIC_MATERIAL_DESCRIPTOR_SET_LAYOUT,
                    &STATIC_MESH_VERTEX_LAYOUT,
                    &[],
                    RasterPipelineCreateDesc::default(),
                )?,
            ),
            BasicMaterialType::Masked(_) => (
                cache.get_or_create_raster_pipeline(
                    "shaders/basic",
                    "shaders/basic",
                    &MAIN_RENDER_PASS_LAYOUT,
                    &BASIC_MATERIAL_DESCRIPTOR_SET_LAYOUT,
                    &STATIC_MESH_VERTEX_LAYOUT,
                    &[(0, 1)],
                    RasterPipelineCreateDesc::default().depth_test(vk::CompareOp::EQUAL),
                )?,
                cache.get_or_create_raster_pipeline(
                    "shaders/depth",
                    "shaders/depth",
                    &ZPASS_RENDER_PASS_LAYOUT,
                    &BASIC_MATERIAL_DESCRIPTOR_SET_LAYOUT,
                    &STATIC_MESH_VERTEX_LAYOUT,
                    &[(0, 1)],
                    RasterPipelineCreateDesc::default(),
                )?,
            ),
            BasicMaterialType::Transparent => (
                cache.get_or_create_raster_pipeline(
                    "shaders/basic",
                    "shaders/basic",
                    &MAIN_RENDER_PASS_LAYOUT,
                    &BASIC_MATERIAL_DESCRIPTOR_SET_LAYOUT,
                    &STATIC_MESH_VERTEX_LAYOUT,
                    &[],
                    RasterPipelineCreateDesc::default().alpha_blend(),
                )?,
                RasterPipelineHandle::default(),
            ),
        };
        Ok(BasicMaterial {
            device,
            uniforms,
            group: self.ty.into(),
            main,
            depth,
            descriptor,
            uniform,
        })
    }
}

impl Material for BasicMaterial {
    fn create_render_data(&self) -> MaterialRenderData {
        MaterialRenderData {
            group: self.group,
            priority: 0,
            depth: self.depth,
            main: self.main,
            descriptor: self.descriptor,
        }
    }
}

impl Drop for BasicMaterial {
    fn drop(&mut self) {
        self.device
            .descriptors()
            .destroy_descriptor(self.descriptor);
        self.uniforms.free(self.uniform);
    }
}

pub static BASIC_MATERIAL_DESCRIPTOR_LAYOUT: DescriptorLayoutDesc = DescriptorLayoutDesc {
    layout: &[
        (
            0,
            DescriptorDesc {
                name: "base_color",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            1,
            DescriptorDesc {
                name: "metallic_roughness",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            2,
            DescriptorDesc {
                name: "normals",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            3,
            DescriptorDesc {
                name: "occlusion",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            4,
            DescriptorDesc {
                name: "emissive",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            5,
            DescriptorDesc {
                name: "material",
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
            },
        ),
    ],
    compute_groups_size: None,
};

const BASIC_MATERIAL_DESCRIPTOR_SET_LAYOUT: [DescriptorLayoutDesc; 4] = [
    SCENE_DESCRIPTOR_LAYOUT,
    EMPTY_DESCRIPTOR_LAYOUT,
    BASIC_MATERIAL_DESCRIPTOR_LAYOUT,
    INSTANCE_DESCRIPTOR_LAYOUT,
];
