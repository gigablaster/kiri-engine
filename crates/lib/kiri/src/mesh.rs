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

use kiri_assets::NodeIndex;
use kiri_backend::{BindGroupHandle, BufferSlice, ImageHandle};

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
    pub bounds: Bounds,
}

#[derive(Debug, Default)]
pub struct RenderScene {
    pub meshes: Vec<StaticMeshHandle>,
    pub bounds: Vec<Bounds>,
    pub names: HashMap<String, usize>,
    pub parents: Vec<NodeIndex>,
    pub local_transforms: Vec<glam::Affine3A>,
    pub world_transforms: Vec<glam::Affine3A>,
    pub node_to_mesh: Vec<(usize, usize)>,
}

impl RenderScene {
    pub(crate) fn update_world_transforms(&mut self) {
        for (index, local) in self.local_transforms.iter().enumerate() {
            let parent = self.parents[index]
                .index()
                .map(|index| self.world_transforms[index as usize])
                .unwrap_or(self.local_transforms[index]);
            self.world_transforms[index] = parent * *local;
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Bounds {
    pub center: glam::Vec3,
    pub radius: f32,
}

impl Bounds {
    pub fn from_array_and_radius(center: [f32; 3], radius: f32) -> Self {
        Self {
            center: glam::Vec3::from_array(center),
            radius,
        }
    }

    pub fn transform(self, transform: glam::Affine3A) -> Self {
        let (scale, _, _) = transform.to_scale_rotation_translation();
        let scale = scale.max_element();
        Self {
            center: transform.transform_point3(self.center),
            radius: self.radius * scale,
        }
    }
}
