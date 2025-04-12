// Copyright (C) 2024-2025 gigablaster

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

use kiri_assets::{ModelAsset, NodeIndex, RenderMeshVertex};
use kiri_backend::vulkan::{
    BufferCreateDesc, BufferHandle, BufferPointer, GraphicsDevice, ImageHandle,
};
use kiri_math::{Affine3A, BoundingBox, Bounds, Vec3A};

use crate::{
    material::{Material, MaterialRenderData},
    Error,
};

#[derive(Debug, Clone, Copy)]
pub struct RenderMeshSurface {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub material: MaterialRenderData,
}

#[derive(Debug, Default)]
pub struct RenderMesh {
    pub vertex_buffer: BufferPointer,
    pub index_buffer: BufferPointer,
    pub surfaces: Vec<RenderMeshSurface>,
    pub bounds: BoundingBox,
    pub position_scale: f32,
    pub uv_scale: f32,
}

unsafe impl Send for RenderMesh {}
unsafe impl Sync for RenderMesh {}
#[derive(Debug)]
pub struct RenderModel {
    device: Arc<GraphicsDevice>,
    pub vertices: BufferHandle,
    pub indices: BufferHandle,
    pub meshes: Vec<Arc<RenderMesh>>,
    pub bounds_per_mesh: Vec<BoundingBox>,
    pub names: HashMap<String, u32>,
    pub parents: Vec<NodeIndex>,
    pub local_transforms: Vec<Affine3A>,
    pub world_transforms: Vec<Affine3A>,
    pub node_to_mesh: Vec<(u32, u32)>,
    pub bounds: BoundingBox,
}

unsafe impl Send for RenderModel {}
unsafe impl Sync for RenderModel {}

#[derive(Debug, Clone, Copy)]
pub struct RenderMeshPbrMaterialDesc {
    pub emissive_power: f32,
    pub alpha_cutoff: f32,
    pub base_color: ImageHandle,
    pub metallic_roughness: ImageHandle,
    pub normals: ImageHandle,
    pub occlusion: ImageHandle,
    pub emissive: ImageHandle,
}

#[derive(Debug)]
pub struct RenderMeshBuilder {
    pub first_vertex: usize,
    pub first_index: usize,
    pub surfaces: Vec<RenderMeshSurface>,
    pub bounds: BoundingBox,
    pub position_scale: f32,
    pub uv_scale: f32,
}

impl RenderMeshBuilder {
    pub fn new(vertex_offset: usize, index_offset: usize) -> Self {
        Self {
            first_vertex: vertex_offset,
            first_index: index_offset,
            bounds: BoundingBox::default(),
            position_scale: 1.0,
            uv_scale: 1.0,
            surfaces: Default::default(),
        }
    }

    pub fn surface<T: Material>(&mut self, first_index: u32, index_count: u32, material: &T) {
        self.surfaces.push(RenderMeshSurface {
            vertex_offset: self.first_vertex as u32,
            first_index,
            index_count,
            material: material.create_render_data(),
        });
    }

    pub fn bounds(mut self, value: BoundingBox) -> Self {
        self.bounds = value;
        self
    }

    pub fn position_scale(mut self, value: f32) -> Self {
        self.position_scale = value;
        self
    }

    pub fn uv_scale(mut self, value: f32) -> Self {
        self.uv_scale = value;
        self
    }

    pub fn build(self, vertices: BufferHandle, indices: BufferHandle) -> RenderMesh {
        RenderMesh {
            vertex_buffer: BufferPointer::new(vertices, 0),
            index_buffer: BufferPointer::new(indices, self.first_index * mem::size_of::<u16>()),
            bounds: self.bounds,
            surfaces: self.surfaces,
            position_scale: self.position_scale,
            uv_scale: self.uv_scale,
        }
    }
}

#[derive(Debug)]
struct NodeBuilder {
    pub parent: NodeIndex,
    pub name: String,
    pub local_transform: Affine3A,
    pub world_transform: Affine3A,
}

pub struct RenderModelBuilder<'a, T: Copy> {
    vertices: &'a [T],
    indices: &'a [u16],
    nodes: Vec<NodeBuilder>,
    meshes: Vec<RenderMeshBuilder>,
    attached_meshes: Vec<(u32, u32)>,
    name: Option<&'a str>,
}

impl<'a, T: Copy> RenderModelBuilder<'a, T> {
    pub fn new(vertices: &'a [T], indices: &'a [u16]) -> Self {
        Self {
            vertices,
            indices,
            nodes: Default::default(),
            meshes: Default::default(),
            attached_meshes: Default::default(),
            name: None,
        }
    }

    pub fn add_mesh(&mut self, mesh: RenderMeshBuilder) -> u32 {
        let index = self.meshes.len() as u32;
        self.meshes.push(mesh);
        index
    }

    pub fn add_node(
        &mut self,
        parent: NodeIndex,
        name: &str,
        local_transform: Affine3A,
    ) -> NodeIndex {
        let index = self.nodes.len() as u32;
        let parent_transform = parent
            .index()
            .map(|index| self.nodes[index as usize].world_transform)
            .unwrap_or_default();
        self.nodes.push(NodeBuilder {
            parent,
            name: name.to_owned(),
            local_transform,
            world_transform: parent_transform * local_transform,
        });
        NodeIndex::new(index)
    }

    pub fn attach_mesh(&mut self, node: u32, mesh: u32) {
        self.attached_meshes.push((node, mesh));
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    pub fn build(self, device: Arc<GraphicsDevice>) -> Result<RenderModel, Error> {
        assert!(!self.meshes.is_empty(), "Model must have at least one mesh");
        assert!(!self.nodes.is_empty(), "Model must have at least one node");
        let name = self.name.unwrap_or("Mesh");
        let vertices = device.create_buffer(
            BufferCreateDesc::gpu(mem::size_of_val(self.vertices) as _)
                .transfer_destination()
                .veretex_buffer()
                .name(&format!("{} positions", name)),
        )?;
        let indices = device.create_buffer(
            BufferCreateDesc::gpu(mem::size_of_val(self.indices) as _)
                .transfer_destination()
                .index_buffer()
                .name(&format!("{} indices", name)),
        )?;
        device.upload_buffer(BufferPointer::new(vertices, 0), self.vertices)?;
        device.upload_buffer(BufferPointer::new(indices, 0), self.indices)?;
        let bounds = self.calculate_bounds();
        Ok(RenderModel {
            device,
            vertices,
            indices,
            bounds_per_mesh: self.meshes.iter().map(|x| x.bounds).collect(),
            meshes: self
                .meshes
                .into_iter()
                .map(|x| x.build(vertices, indices))
                .map(Arc::new)
                .collect(),
            names: self
                .nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (node.name.clone(), index as u32))
                .collect(),
            parents: self.nodes.iter().map(|x| x.parent).collect(),
            local_transforms: self.nodes.iter().map(|x| x.local_transform).collect(),
            world_transforms: self.nodes.iter().map(|x| x.world_transform).collect(),
            node_to_mesh: self.attached_meshes,
            bounds,
        })
    }

    fn calculate_bounds(&self) -> BoundingBox {
        let mut min = Vec3A::MAX;
        let mut max = Vec3A::MIN;
        for (parent, mesh) in &self.attached_meshes {
            let bounds = self.meshes[*mesh as usize].bounds;
            let transform = self.nodes[*parent as usize].world_transform;
            let bounds = bounds.transform(transform);
            min = min.min(bounds.min);
            max = max.max(bounds.max);
        }
        BoundingBox::new(min.into(), max.into())
    }
}

impl Drop for RenderModel {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.vertices);
        self.device.destroy_buffer(self.indices);
    }
}
