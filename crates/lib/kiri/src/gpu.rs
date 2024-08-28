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

use std::mem;

use ash::vk;
use kiri_assets::{MaterialData, MeshAssetMaterial, StaticMeshVertex};
use kiri_backend::{InputVertexAttrubute, InputVertexStreamLayout, PipelineVertex};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuMeshMaterial {
    pub base_color: glam::Vec4,
    pub emissive_color: glam::Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub alpha_cutoff: f32,
    pub emissive_power: f32,
}

impl GpuMeshMaterial {
    pub fn new(value: &MeshAssetMaterial) -> Self {
        let [_, roughness, metallic, _] = value
            .maps
            .get("metallic_roughness")
            .unwrap_or(&MaterialData::color([0.0, 1.0, 0.0, 1.0]))
            .get_color();
        let alpha_cutoff = value.blend.get_alpha_cut();
        let emissive_power = value.get_emissive_power();
        Self {
            base_color: value
                .maps
                .get("base_color")
                .unwrap_or(&MaterialData::color([0.5, 0.5, 0.5, 1.0]))
                .get_color()
                .into(),
            emissive_color: value
                .maps
                .get("emissive")
                .unwrap_or(&MaterialData::color([0.0, 0.0, 0.0, 0.0]))
                .get_color()
                .into(),
            metallic,
            roughness,
            alpha_cutoff,
            emissive_power,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuStaticVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv1: [f32; 2],
    // pub uv2: [f32; 2],
}

impl PipelineVertex for GpuStaticVertex {
    fn layout() -> &'static [InputVertexStreamLayout<'static>] {
        &[InputVertexStreamLayout {
            streams: &[
                InputVertexAttrubute {
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                InputVertexAttrubute {
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 12,
                },
                InputVertexAttrubute {
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 24,
                },
                InputVertexAttrubute {
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 40,
                },
            ],
            stride: mem::size_of::<GpuStaticVertex>() as u32,
        }]
    }
}

impl From<StaticMeshVertex> for GpuStaticVertex {
    fn from(value: StaticMeshVertex) -> Self {
        Self {
            position: value.position,
            normal: value.normal,
            tangent: value.tangent,
            uv1: value.uv1,
            // uv2: glam::U16Vec2::from_array(value.uv2),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RenderPassGpuData {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub view_projection: glam::Mat4,
    pub eye_position: glam::Vec3,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuInstanceData {
    pub model: glam::Mat4,
}
