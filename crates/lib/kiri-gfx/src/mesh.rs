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

use std::collections::HashMap;

use kiri_assets::BoneIndex;
use kiri_backend::{
    BindGroupDesc, BindGroupHandle, BindGroupSlotDesc, BindType, BufferSlice, ImageHandle,
    ShaderStage,
};

use crate::StaticMeshHandle;

#[derive(Debug, Clone, Copy)]
pub struct RenderMeshSurface {
    pub first_index: u32,
    pub index_count: u32,
    pub material_index: usize,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub(crate) struct RenderMeshShaderData {
    pub position_scale: f32,
    pub uv1_scale: f32,
    pub uv2_scale: f32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub(crate) struct PbrMaterialShaderData {
    pub emissive_power: f32,
    pub alpha_cutoff: f32,
}

pub(crate) const STATIC_MESH_BIND_GROUP: BindGroupDesc = BindGroupDesc {
    stage: ShaderStage::Graphics,
    set: &[BindGroupSlotDesc {
        name: "object",
        slot: 0,
        ty: BindType::Uniform,
    }],
};

pub(crate) const PBR_MATERIAL_BIND_GROUP: BindGroupDesc = BindGroupDesc {
    stage: ShaderStage::Graphics,
    set: &[
        BindGroupSlotDesc {
            name: "material",
            slot: 0,
            ty: BindType::Uniform,
        },
        BindGroupSlotDesc {
            name: "base_color",
            slot: 1,
            ty: BindType::CombinedSampledImage,
        },
        BindGroupSlotDesc {
            name: "normals",
            slot: 2,
            ty: BindType::CombinedSampledImage,
        },
        BindGroupSlotDesc {
            name: "metallic_roughness",
            slot: 3,
            ty: BindType::CombinedSampledImage,
        },
        BindGroupSlotDesc {
            name: "occlusion",
            slot: 4,
            ty: BindType::CombinedSampledImage,
        },
        BindGroupSlotDesc {
            name: "emissive",
            slot: 5,
            ty: BindType::CombinedSampledImage,
        },
    ],
};

#[derive(Debug)]
pub struct RenderMeshMaterial {
    pub bind_group: BindGroupHandle,
    pub images: Vec<ImageHandle>,
}

#[derive(Debug)]
pub struct StaticRenderMesh {
    pub vertices: BufferSlice,
    pub indices: BufferSlice,
    pub object_bind_group: BindGroupHandle,
    pub surfaces: Vec<RenderMeshSurface>,
    pub materials: Vec<RenderMeshMaterial>,
}

#[derive(Debug, Default)]
pub struct RenderScene {
    pub meshes: Vec<StaticMeshHandle>,
    pub names: HashMap<String, usize>,
    pub parents: Vec<BoneIndex>,
    pub local_transforms: Vec<glam::Mat4>,
    pub world_transforms: Vec<glam::Mat4>,
    pub node_to_mesh: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub struct RenderSceneGroup {
    pub scenes: HashMap<String, RenderScene>,
}
