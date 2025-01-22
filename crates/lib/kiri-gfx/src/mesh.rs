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

use std::{collections::HashMap, hash::Hash, mem, sync::Arc};

use crate::{BufferHandle, BufferPointer, DescriptorHandle, Error, ImageHandle, Renderer};
use kiri_assets::{
    load_or_compile_asset, ImageAssetType, ImageData, ImageSource, MeshAssetMaterial,
    MeshMaterialBlend, RenderMeshVertex,
};
use kiri_backend::{
    ash::{
        ext::color_write_enable::Device,
        vk::{self, DescriptorSet},
    },
    BufferCreateDesc, DescriptorDesc, DescriptorSetLayoutDesc, Image, ImageCreateDesc,
    ImageUploadData, RenderDevice,
};
use kiri_common::NodeIndex;
use kiri_math::{Affine3A, BoundingBox, Bounds, Vec3A, Vec4};
use turbosloth::{async_trait, IntoLazy, LazyWorker, RunContext};

#[derive(Debug, Clone, Copy)]
pub enum RenderMeshMaterialType {
    PBR,
}

#[derive(Debug, Clone, Copy)]
pub enum RenderMeshMaterialOrder {
    Opaque,
    Masked,
    Transparent,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderMeshMaterial {
    pub ty: RenderMeshMaterialType,
    pub order: RenderMeshMaterialOrder,
    pub descriptor: DescriptorSet,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderMeshSurface {
    pub first_index: u32,
    pub index_count: u32,
    pub material: RenderMeshMaterial,
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

#[derive(Debug)]
pub struct RenderModel {
    renderer: Arc<Renderer>,
    pub vertices: BufferHandle,
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
    pub first_vertex: u64,
    pub first_index: u64,
    pub surfaces: Vec<RenderMeshSurface>,
    pub bounds: BoundingBox,
    pub position_scale: f32,
    pub uv_scale: f32,
}

impl RenderMeshBuilder {
    pub fn new(vertex_offset: u64, index_offset: u64) -> Self {
        Self {
            first_vertex: vertex_offset,
            first_index: index_offset,
            bounds: BoundingBox::default(),
            position_scale: 1.0,
            uv_scale: 1.0,
            surfaces: Default::default(),
        }
    }

    pub fn surface(&mut self, first_index: u32, index_count: u32, material: RenderMeshMaterial) {
        self.surfaces.push(RenderMeshSurface {
            first_index,
            index_count,
            material,
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
            vertex_buffer: BufferPointer::new(
                vertices,
                self.first_vertex * mem::size_of::<RenderMeshVertex>() as u64,
            ),
            index_buffer: BufferPointer::new(
                indices,
                self.first_index * mem::size_of::<u16>() as u64,
            ),
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

    pub fn build(self, renderer: &Arc<Renderer>) -> Result<RenderModel, Error> {
        assert!(!self.meshes.is_empty(), "Model must have at least one mesh");
        assert!(!self.nodes.is_empty(), "Model must have at least one node");
        let name = self.name.unwrap_or("Mesh");
        let vertices = renderer.create_buffer(
            BufferCreateDesc::gpu(mem::size_of_val(self.vertices) as _)
                .transfer_destination()
                .veretex_buffer()
                .device_address()
                .name(&format!("{} positions", name)),
        )?;
        let indices = renderer.create_buffer(
            BufferCreateDesc::gpu(mem::size_of_val(self.indices) as _)
                .transfer_destination()
                .index_buffer()
                .device_address()
                .name(&format!("{} indices", name)),
        )?;
        renderer.upload_buffer_data(BufferPointer::new(vertices, 0), self.vertices)?;
        renderer.upload_buffer_data(BufferPointer::new(indices, 0), self.indices)?;
        let bounds = self.calculate_bounds();
        Ok(RenderModel {
            renderer: renderer.clone(),
            vertices,
            indices,
            bounds_per_mesh: self.meshes.iter().map(|x| x.bounds).collect(),
            meshes: self
                .meshes
                .into_iter()
                .map(|x| x.build(vertices, indices))
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
        self.renderer.destroy_buffer(self.vertices);
        self.renderer.destroy_buffer(self.indices);
    }
}

#[derive(Debug)]
pub struct RenderTexture {
    renderer: Arc<Renderer>,
    handle: ImageHandle,
}

impl Drop for RenderTexture {
    fn drop(&mut self) {
        self.renderer.destroy_image(self.handle);
    }
}

#[derive(Debug, Clone)]
pub struct LoadTexture {
    renderer: Arc<Renderer>,
    source: ImageSource,
}

impl Hash for LoadTexture {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

impl LoadTexture {
    pub fn new(renderer: Arc<Renderer>, source: ImageSource) -> Self {
        Self { renderer, source }
    }
}

#[async_trait]
impl LazyWorker for LoadTexture {
    type Output = Result<RenderTexture, Error>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        let image = load_or_compile_asset(self.source).await?;
        let data = image
            .mips
            .iter()
            .map(|mip| ImageUploadData { data: &mip })
            .collect::<Vec<_>>();
        Ok(RenderTexture {
            handle: self.renderer.create_image(
                ImageCreateDesc::texture(image.format, image.dims),
                Some(&data),
            )?,
            renderer: self.renderer,
        })
    }
}

#[derive(Debug)]
pub struct MeshMaterial {
    remderer: Arc<Renderer>,
    ty: RenderMeshMaterialType,
    order: RenderMeshMaterialOrder,
    descriptor: DescriptorHandle,
}

#[derive(Debug, Clone)]
pub struct LoadMaterial {
    renderer: Arc<Renderer>,
    source: MeshAssetMaterial,
}

impl Hash for LoadMaterial {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

impl LoadMaterial {
    pub fn new(renderer: Arc<Renderer>, source: MeshAssetMaterial) -> Self {
        Self { renderer, source }
    }
}

pub static MESH_PBR_MATERIAL_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
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
                name: "occlusion",
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

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuMeshMaterial {
    alpha_cut: f32,
    emissive_power: f32,
}

#[async_trait]
impl LazyWorker for LoadMaterial {
    type Output = Result<Arc<MeshMaterial>, Error>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let images = [
            self.source.get_image(
                "base_color",
                ImageSource::color([127, 127, 127, 255]).srgb(true),
            ),
            self.source
                .get_image("metallic_roughness", ImageSource::color([0, 0, 192, 255])),
            self.source.get_image(
                "normals",
                ImageSource::color([127, 127, 255, 255]).ty(ImageAssetType::Rg),
            ),
            self.source
                .get_image("occlusion", ImageSource::color([0, 0, 0, 0])),
            self.source
                .get_image("emission", ImageSource::color([0, 0, 0, 0])),
        ]
        .map(|image| LoadTexture::new(self.renderer.clone(), image));
        let images = futures::future::try_join_all(
            images.into_iter().map(|image| image.into_lazy().eval(&ctx)),
        )
        .await?;
        let emissive_power = self
            .source
            .scalars
            .get("emissive_power")
            .copied()
            .unwrap_or(0.0);

        todo!()
    }
}
