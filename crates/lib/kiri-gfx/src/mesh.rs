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

use std::{collections::HashMap, mem, sync::Arc};

use crate::{
    effects::{Effect, EffectInstance},
    BufferHandle, BufferPointer, Error, PipelineHandle, Renderer,
};
use kiri_backend::BufferCreateDesc;
use kiri_common::NodeIndex;
use kiri_math::{Affine3A, BoundingBox, Bounds, Vec3A};

#[derive(Debug)]
pub struct RenderMeshMaterial {
    pub effect: Arc<dyn Effect>,
    pub instance: EffectInstance,
    pub main: PipelineHandle,
    pub transparent: PipelineHandle,
    pub depth: PipelineHandle,
}

impl Drop for RenderMeshMaterial {
    fn drop(&mut self) {
        self.effect.free_instance(&mut self.instance);
    }
}

#[derive(Debug)]
pub struct RenderMeshSurface {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub material: Arc<RenderMeshMaterial>,
}

#[derive(Debug, Default)]
pub struct RenderMesh {
    pub positions: BufferPointer,
    pub attributes: BufferPointer,
    pub index_buffer: BufferPointer,
    pub surfaces: Vec<RenderMeshSurface>,
    pub bounds: BoundingBox,
    pub position_scale: f32,
    pub uv_scale: f32,
}

#[derive(Debug)]
pub struct RenderModel {
    renderer: Arc<Renderer>,
    pub positions: BufferHandle,
    pub attributes: BufferHandle,
    pub indices: BufferHandle,
    pub meshes: Vec<RenderMesh>,
    pub bounds_per_mesh: Vec<BoundingBox>,
    pub names: HashMap<String, u32>,
    pub parents: Vec<NodeIndex>,
    pub local_transforms: Vec<Affine3A>,
    pub world_transforms: Vec<Affine3A>,
    pub node_to_mesh: Vec<(u32, u32)>,
    pub bounds: BoundingBox,
}

pub struct RenderMeshBuilder {
    pub vertex_offset: u64,
    pub index_offset: u64,
    pub surfaces: Vec<RenderMeshSurface>,
    pub bounds: BoundingBox,
    pub position_scale: f32,
    pub uv_scale: f32,
}

impl RenderMeshBuilder {
    pub fn new(vertex_offset: u64, index_offset: u64) -> Self {
        Self {
            vertex_offset,
            index_offset,
            bounds: BoundingBox::default(),
            position_scale: 1.0,
            uv_scale: 1.0,
            surfaces: Default::default(),
        }
    }

    pub fn surface(
        &mut self,
        first_index: u32,
        index_count: u32,
        material: &Arc<RenderMeshMaterial>,
    ) {
        self.surfaces.push(RenderMeshSurface {
            first_index,
            index_count,
            vertex_offset: self.vertex_offset as u32,
            material: material.clone(),
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

    pub fn build(
        self,
        positions: BufferHandle,
        attributes: BufferHandle,
        indices: BufferHandle,
    ) -> RenderMesh {
        RenderMesh {
            positions: BufferPointer::new(positions, self.vertex_offset),
            attributes: BufferPointer::new(attributes, self.vertex_offset),
            index_buffer: BufferPointer::new(indices, self.index_offset),
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

pub struct RenderModelBuilder<'a, T: Copy, U: Copy> {
    positions: &'a [T],
    attributes: &'a [U],
    indices: &'a [u16],
    nodes: Vec<NodeBuilder>,
    meshes: Vec<RenderMeshBuilder>,
    attached_meshes: Vec<(u32, u32)>,
    name: Option<&'a str>,
}

impl<'a, T: Copy, U: Copy> RenderModelBuilder<'a, T, U> {
    pub fn new(positions: &'a [T], attributes: &'a [U], indices: &'a [u16]) -> Self {
        Self {
            positions,
            attributes,
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

    pub fn build(self, renderer: &Arc<Renderer>) -> Result<RenderModel, Error> {
        assert!(!self.meshes.is_empty(), "Model must have at least one mesh");
        assert!(!self.nodes.is_empty(), "Model must have at least one node");
        let name = self.name.unwrap_or("Mesh");
        let positions = renderer.create_buffer(
            BufferCreateDesc::gpu(mem::size_of_val(self.positions) as _)
                .transfer_destination()
                .veretex_buffer()
                .device_address()
                .name(&format!("{} positions", name)),
        )?;
        let attributes = renderer.create_buffer(
            BufferCreateDesc::gpu(mem::size_of_val(self.attributes) as _)
                .transfer_destination()
                .veretex_buffer()
                .device_address()
                .name(&format!("{} attributes", name)),
        )?;
        let indices = renderer.create_buffer(
            BufferCreateDesc::gpu(mem::size_of_val(self.indices) as _)
                .transfer_destination()
                .index_buffer()
                .device_address()
                .name(&format!("{} indices", name)),
        )?;
        renderer.upload_buffer(BufferPointer::new(positions, 0), self.positions)?;
        renderer.upload_buffer(BufferPointer::new(attributes, 0), self.attributes)?;
        renderer.upload_buffer(BufferPointer::new(indices, 0), self.indices)?;
        let bounds = self.calculate_bounds();
        Ok(RenderModel {
            renderer: renderer.clone(),
            positions,
            attributes,
            indices,
            bounds_per_mesh: self.meshes.iter().map(|x| x.bounds).collect(),
            meshes: self
                .meshes
                .into_iter()
                .map(|x| x.build(positions, attributes, indices))
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
        self.renderer.destroy_buffer(self.positions);
        self.renderer.destroy_buffer(self.attributes);
        self.renderer.destroy_buffer(self.indices);
    }
}
