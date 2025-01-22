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

use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    io,
    time::SystemTime,
};

use kiri_common::NodeIndex;
use speedy::{Readable, Writable};
use uuid::uuid;

use crate::ImportAsset;
use crate::{
    get_absolute_asset_path, get_relative_asset_path, is_asset_changed, Asset, AssetReference,
    AssetSource, ImageSource,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelSource(String);

impl AssetSource for ModelSource {
    fn reference(&self) -> AssetReference {
        AssetReference::new(self)
    }

    fn changed(&self, last_update: SystemTime) -> bool {
        is_asset_changed(&self.0, last_update)
    }
}

impl ModelSource {
    pub fn new(path: &str) -> Self {
        Self(path.replace("\\", "/"))
    }
}

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

impl Hash for MeshMaterialBlend {
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

#[derive(Debug, Clone, Readable, Writable, PartialEq)]
pub struct MeshAssetMaterial {
    pub name: String,
    pub images: HashMap<String, ImageSource>,
    pub scalars: HashMap<String, f32>,
    pub vectors: HashMap<String, [f32; 4]>,
    pub blend: MeshMaterialBlend,
}

impl Hash for MeshAssetMaterial {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.images.iter().for_each(|(name, texture)| {
            name.hash(state);
            texture.hash(state);
        });
        self.scalars.iter().for_each(|(name, value)| {
            name.hash(state);
            ((value * 100000.0) as u64).hash(state);
        });
        self.vectors.iter().for_each(|(name, [x, y, z, w])| {
            name.hash(state);
            ((x * 100000.0) as u64).hash(state);
            ((y * 100000.0) as u64).hash(state);
            ((z * 100000.0) as u64).hash(state);
            ((w * 100000.0) as u64).hash(state);
        });
        self.blend.hash(state);
    }
}

impl Eq for MeshAssetMaterial {}

impl MeshAssetMaterial {
    fn collect_images<'a>(&'a self, images: &mut HashSet<&'a ImageSource>) {
        self.images.iter().for_each(|(_, image)| {
            images.insert(image);
        });
    }

    pub fn get_image(&self, name: &str, default: ImageSource) -> ImageSource {
        self.images.get(name).cloned().unwrap_or(default)
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

impl ModelAsset {
    pub fn collect_dependencies(&self) -> HashSet<&ImageSource> {
        let mut result = HashSet::new();
        for material in &self.materials {
            material.collect_images(&mut result);
        }
        result
    }
}

impl Asset for ModelAsset {
    const TYPE: uuid::Uuid = uuid!("3d731621-54b4-40b0-a089-37667f68fe35");
    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }

    fn serialize<W: std::io::Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }
}

mod import {
    use std::{collections::HashMap, io};

    use gltf::mesh::Mode;

    use crate::{ImageAssetType, ImageSource, MeshAssetBuilder, MeshSurfaceBuilder};

    use super::{
        MeshAssetMaterial, MeshMaterialBlend, ModelAsset, Node, NodeIndex, RenderMeshVertex,
        StaticMeshAsset,
    };

    pub struct GltfProcessingContext<'a> {
        pub base_path: &'a str,
        pub buffers: Vec<gltf::buffer::Data>,
        pub vertices: Vec<RenderMeshVertex>,
        pub indices: Vec<u16>,
        pub materials: Vec<MeshAssetMaterial>,
    }

    pub struct NodeProcessingContext<'a> {
        context: &'a mut GltfProcessingContext<'a>,
        bone_to_mesh: HashMap<u32, u32>,
        bones: Vec<Node>,
        bone_names: HashMap<String, u32>,
        meshes: Vec<StaticMeshAsset>,
        mesh_names: Vec<String>,
        name_to_mesh: HashMap<String, u32>,
        processed_meshes: HashMap<u32, u32>,
    }

    fn process_texture(
        context: &GltfProcessingContext,
        texture: &gltf::texture::Texture,
        ty: ImageAssetType,
        srgb: bool,
    ) -> ImageSource {
        match texture.source().source() {
            gltf::image::Source::Uri { uri, .. } => {
                ImageSource::path(format!("{}/{}", context.base_path, uri))
                    .ty(ty)
                    .srgb(srgb)
            }
            _ => panic!(),
        }
    }

    fn process_blend(material: &gltf::Material) -> MeshMaterialBlend {
        match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => MeshMaterialBlend::Opaque,
            gltf::material::AlphaMode::Mask => {
                MeshMaterialBlend::AlphaTest(material.alpha_cutoff().unwrap_or(0.0).clamp(0.0, 1.0))
            }
            gltf::material::AlphaMode::Blend => MeshMaterialBlend::AlphaBlend,
        }
    }

    fn color(color: [f32; 4]) -> [u8; 4] {
        [
            (color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (color[2].clamp(0.0, 1.0) * 255.0) as u8,
            (color[3].clamp(0.0, 1.0) * 255.0) as u8,
        ]
    }

    fn process_material(
        context: &GltfProcessingContext,
        material: gltf::Material,
    ) -> MeshAssetMaterial {
        let base_color =
            if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
                process_texture(context, &texture.texture(), ImageAssetType::Rgba, true)
            } else {
                ImageSource::color(color(material.pbr_metallic_roughness().base_color_factor()))
            };
        let metallic_roughness = if let Some(texture) = material
            .pbr_metallic_roughness()
            .metallic_roughness_texture()
        {
            process_texture(context, &texture.texture(), ImageAssetType::Rgba, false)
        } else {
            ImageSource::color(color([
                0.0,
                material.pbr_metallic_roughness().roughness_factor(),
                material.pbr_metallic_roughness().metallic_factor(),
                1.0,
            ]))
        };
        let normals = if let Some(texture) = material.normal_texture() {
            process_texture(context, &texture.texture(), ImageAssetType::Rg, false)
        } else {
            ImageSource::color([127, 127, 255, 255])
        };
        let occlusion = if let Some(texture) = material.occlusion_texture() {
            process_texture(context, &texture.texture(), ImageAssetType::Rgba, false)
        } else {
            ImageSource::color([0, 0, 0, 0])
        };
        let emissive_color = material.emissive_factor();
        let emissive = if let Some(texture) = material.emissive_texture() {
            process_texture(context, &texture.texture(), ImageAssetType::Rgba, false)
        } else {
            ImageSource::color(color([
                emissive_color[0],
                emissive_color[1],
                emissive_color[2],
                1.0,
            ]))
        };
        MeshAssetMaterial {
            name: material.name().unwrap_or("default").to_string(),
            images: [
                ("base_color".to_owned(), base_color),
                ("normals".to_owned(), normals),
                ("metallic_roughness".to_owned(), metallic_roughness),
                ("occlusion".to_owned(), occlusion),
                ("emissive".to_owned(), emissive),
            ]
            .into(),
            scalars: [(
                "emissive_power".to_owned(),
                material.emissive_strength().unwrap_or(0.0),
            )]
            .into(),
            blend: process_blend(&material),
            vectors: Default::default(),
        }
    }

    fn process_mesh(
        context: &mut GltfProcessingContext,
        mesh: gltf::Mesh,
    ) -> io::Result<StaticMeshAsset> {
        let mut builder = MeshAssetBuilder::default();
        for prim in mesh.primitives() {
            let mut surface = MeshSurfaceBuilder::new(process_material(context, prim.material()));
            if prim.mode() != Mode::Triangles {
                return Err(io::Error::other(
                    "Only processing triangle meshes".to_string(),
                ));
            }
            let reader = prim.reader(|buffer| Some(&context.buffers[buffer.index()]));
            if let Some(positions) = reader.read_positions() {
                surface.push_position(&positions.collect::<Vec<_>>());
            } else {
                return Err(io::Error::other("Mesh has no positions".to_string()));
            };
            if let Some(indices) = reader.read_indices() {
                surface.push_indices(&indices.into_u32().collect::<Vec<_>>());
            } else {
                return Err(io::Error::other(
                    "Only processing indexed meshes".to_string(),
                ));
            }
            if let Some(normals) = reader.read_normals() {
                surface.push_normals(&normals.collect::<Vec<_>>());
            }
            if let Some(tangents) = reader.read_tangents() {
                surface.push_tangents(&tangents.collect::<Vec<_>>());
            }
            if let Some(uvs) = reader.read_tex_coords(0) {
                surface.push_uv1(&uvs.into_f32().collect::<Vec<_>>());
            }
            if let Some(uvs) = reader.read_tex_coords(1) {
                surface.push_uv2(&uvs.into_f32().collect::<Vec<_>>());
            }
            builder.push(surface);
        }
        Ok(builder.build(
            &mut context.vertices,
            &mut context.indices,
            &mut context.materials,
        ))
    }

    fn process_node(
        context: &mut NodeProcessingContext,
        parent_index: NodeIndex,
        name: &str,
        node: gltf::Node,
    ) -> io::Result<()> {
        let bone_index = context.bones.len() as u32;
        let (translation, rotation, scale) = node.transform().decomposed();
        context.bones.push(Node {
            name: name.to_owned(),
            parent: parent_index,
            translation,
            rotation,
            scale,
        });
        context.bone_names.insert(name.to_owned(), bone_index);
        if let Some(mesh) = node.mesh() {
            if let Some(mesh_index) = context.processed_meshes.get(&(mesh.index() as u32)) {
                context.bone_to_mesh.insert(bone_index, *mesh_index);
            } else {
                let name = mesh.name().unwrap_or(name);

                let mesh = process_mesh(context.context, mesh)?;
                let mesh_index = context.meshes.len() as u32;
                context.meshes.push(mesh);
                context.mesh_names.push(name.to_owned());
                context.name_to_mesh.insert(name.to_owned(), mesh_index);
                context.bone_to_mesh.insert(bone_index, mesh_index);
            }
        }
        for (index, child) in node.children().enumerate() {
            process_node(
                context,
                NodeIndex::new(bone_index),
                &format!("{}/{}", name, child.name().unwrap_or(&format!("{}", index))),
                child,
            )?;
        }
        Ok(())
    }

    pub fn import_scene<'a>(
        context: &'a mut GltfProcessingContext<'a>,
        scene: gltf::Scene,
    ) -> io::Result<ModelAsset> {
        let mut context = NodeProcessingContext {
            context,
            bone_to_mesh: Default::default(),
            meshes: Default::default(),
            bones: Default::default(),
            bone_names: Default::default(),
            mesh_names: Default::default(),
            name_to_mesh: Default::default(),
            processed_meshes: Default::default(),
        };
        for (index, node) in scene.nodes().enumerate() {
            process_node(
                &mut context,
                NodeIndex::default(),
                node.name().unwrap_or(&format!("{}", index)),
                node,
            )?;
        }
        Ok({
            ModelAsset {
                vertices: context.context.vertices.clone(),
                indices: context.context.indices.clone(),
                meshes: context.meshes,
                nodes: context.bones,
                mesh_names: context.mesh_names,
                name_to_mesh: context.name_to_mesh,
                node_names: context.bone_names,
                node_to_mesh: context.bone_to_mesh.into_iter().collect::<Vec<_>>(),
                materials: context.context.materials.clone(),
            }
        })
    }

    pub(crate) fn import_scenes<'a>(
        context: &'a mut GltfProcessingContext<'a>,
        document: gltf::Document,
    ) -> io::Result<ModelAsset> {
        let scene = document
            .default_scene()
            .ok_or(io::Error::other("Default scene not found".to_string()))?;
        import_scene(context, scene)
    }
}

impl ImportAsset<ModelAsset> for ModelSource {
    fn import(&self) -> io::Result<ModelAsset> {
        let (document, buffers, _) = gltf::import(get_absolute_asset_path(&self.0)?)
            .map_err(|err| io::Error::other(err.to_string()))?;
        let base_path = get_relative_asset_path(&self.0)?
            .parent()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        import::import_scenes(
            &mut import::GltfProcessingContext {
                vertices: Default::default(),
                indices: Default::default(),
                materials: Default::default(),
                base_path: &base_path,
                buffers,
            },
            document,
        )
    }
}
