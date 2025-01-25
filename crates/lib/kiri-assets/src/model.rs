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

use std::collections::HashMap;

use speedy::{Readable, Writable};

use crate::{Asset, CompiledAssetPath};

#[derive(Debug, Clone, Copy, Readable, Writable)]
#[repr(C, align(8))]
pub struct RenderMeshVertex {
    pub position: [i16; 3],
    pub normal_packed: u32,
    pub tangent_packed: u32,
    pub uv: [i16; 2],
}

#[derive(Debug, Clone, Copy, Readable, Writable, PartialEq)]
pub enum MeshMaterialBlend {
    Opaque,
    AlphaBlend,
    AlphaTest(f32),
}

impl std::hash::Hash for MeshMaterialBlend {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            MeshMaterialBlend::Opaque => state.write_u8(1),
            MeshMaterialBlend::AlphaBlend => state.write_u8(2),
            MeshMaterialBlend::AlphaTest(value) => {
                state.write_u8(3);
                state.write_u32((value * 1000.0) as _)
            }
        }
    }
}

impl Eq for MeshMaterialBlend {}

impl MeshMaterialBlend {
    pub fn get_alpha_cut(&self) -> f32 {
        if let Self::AlphaTest(value) = self {
            *value
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Readable, Writable)]
pub enum ImageReference {
    External(CompiledAssetPath),
    Color([u8; 4]),
}

#[derive(Debug, Clone, Readable, Writable, PartialEq)]
pub struct MeshAssetMaterial {
    pub base_color: ImageReference,
    pub metallic_roughness: ImageReference,
    pub normals: ImageReference,
    pub occlusion: ImageReference,
    pub emissive: ImageReference,
    pub emissive_power: f32,
    pub blend: MeshMaterialBlend,
}

impl std::hash::Hash for MeshAssetMaterial {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.base_color.hash(state);
        self.metallic_roughness.hash(state);
        self.normals.hash(state);
        self.occlusion.hash(state);
        self.emissive.hash(state);
        ((self.emissive_power * 1000.0) as u64).hash(state);
        self.blend.hash(state);
    }
}

impl Eq for MeshAssetMaterial {}

impl MeshAssetMaterial {
    pub fn alpha_cutoff(&self) -> f32 {
        match self.blend {
            MeshMaterialBlend::AlphaTest(value) => value,
            _ => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable)]
pub struct MeshSurfaceAsset {
    pub first_index: u32,
    pub index_count: u32,
    pub material: u32,
}

#[derive(Debug, Readable, Writable)]
pub struct StaticMeshAsset {
    pub first_vertex: u64,
    pub first_index: u64,
    pub surfaces: Vec<MeshSurfaceAsset>,
    pub position_scale: f32,
    pub uv_scale: f32,
    pub bounds: ([f32; 3], [f32; 3]),
}

#[derive(Debug, Clone, Copy, Readable, Writable, Eq, PartialEq)]
pub struct NodeIndex(u32);

impl From<u32> for NodeIndex {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Default for NodeIndex {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

impl NodeIndex {
    pub fn none() -> NodeIndex {
        Self::default()
    }

    pub fn new(index: u32) -> NodeIndex {
        Self(index)
    }

    pub fn index(self) -> Option<u32> {
        if self.0 < u32::MAX {
            Some(self.0)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Readable, Writable)]
pub struct Node {
    pub name: String,
    pub parent: NodeIndex,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Debug, Readable, Writable)]
pub struct ModelAsset {
    pub vertices: Vec<RenderMeshVertex>,
    pub indices: Vec<u16>,
    pub meshes: Vec<StaticMeshAsset>,
    pub nodes: Vec<Node>,
    pub mesh_names: Vec<String>,
    pub name_to_mesh: HashMap<String, u32>,
    pub node_names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
    pub materials: Vec<MeshAssetMaterial>,
}

impl Asset for ModelAsset {
    const TYPE: uuid::Uuid = uuid::uuid!("3d731621-54b4-40b0-a089-37667f68fe35");
    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }

    fn serialize<W: std::io::Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }
}
