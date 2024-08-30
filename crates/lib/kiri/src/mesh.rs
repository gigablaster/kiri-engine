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

use kiri_assets::{MeshMaterialBlend, NodeIndex};
use kiri_gfx::{BufferHandle, BufferPointer, DescriptorHandle, ImageHandle};

use crate::Bounds;

#[derive(Debug, Clone, Copy)]
pub struct RenderMeshSurface {
    pub first_index: u32,
    pub index_count: u32,
    pub material: RenderMaterial,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderMeshMaterialData {
    pub base_color: ImageHandle,
    pub normals: ImageHandle,
    pub metallic_roughness: ImageHandle,
    pub occlusion: ImageHandle,
    pub emissive: ImageHandle,
    pub emissive_power: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderMaterialType {
    Opaque,
    Masked,
    Transparent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderMaterial {
    pub ds: DescriptorHandle,
    pub ty: RenderMaterialType,
}

impl From<MeshMaterialBlend> for RenderMaterialType {
    fn from(value: MeshMaterialBlend) -> Self {
        match value {
            MeshMaterialBlend::Opaque => Self::Opaque,
            MeshMaterialBlend::AlphaBlend => Self::Transparent,
            MeshMaterialBlend::AlphaTest(_) => Self::Masked,
        }
    }
}

#[derive(Debug, Default)]
pub struct StaticRenderMesh {
    pub vertex_buffer: BufferPointer,
    pub index_buffer: BufferPointer,
    pub surfaces: Vec<RenderMeshSurface>,
    pub bounds: Bounds,
    // pub position_scale: f32,
    // pub uv_scale: [f32; 2],
}

#[derive(Debug, Default)]
pub struct RenderModel {
    pub vertices: BufferHandle,
    pub indices: BufferHandle,
    pub meshes: Vec<StaticRenderMesh>,
    pub bounds: Vec<Bounds>,
    pub names: HashMap<String, u32>,
    pub parents: Vec<NodeIndex>,
    pub local_transforms: Vec<glam::Affine3A>,
    pub world_transforms: Vec<glam::Affine3A>,
    pub node_to_mesh: Vec<(u32, u32)>,
    pub mesh_names: Vec<String>,
}

impl RenderModel {
    pub(super) fn update_world_transforms(&mut self) {
        for (index, local) in self.local_transforms.iter().enumerate() {
            let parent = self.parents[index]
                .index()
                .map(|index| self.world_transforms[index as usize])
                .unwrap_or(self.local_transforms[index]);
            self.world_transforms[index] = parent * *local;
        }
    }
}
