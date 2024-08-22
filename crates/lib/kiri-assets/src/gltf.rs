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
    path::Path,
    time::SystemTime,
};

use gltf::mesh::Mode;
use normalize_path::NormalizePath;
use speedy::{Readable, Writable};
use uuid::uuid;

use crate::{
    get_absolute_asset_path, get_relative_asset_path, is_asset_changed, Asset, AssetReference,
    AssetSource, Error, ImageAssetType, ImportAsset, MeshAssetBuilder, MeshSurfaceBuilder,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GltfSceneSource(String);

impl AssetSource for GltfSceneSource {
    fn reference(&self) -> AssetReference {
        AssetReference::new(&self.0)
    }

    fn changed(&self, last_update: SystemTime) -> bool {
        is_asset_changed(&self.0, last_update)
    }
}

impl GltfSceneSource {
    pub fn new(path: &str) -> Self {
        Self(path.replace("\\", "/"))
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable)]
pub struct StaticMeshVertex {
    pub position: [u16; 3],
    pub normal: [u16; 2],
    pub tangent: [u16; 2],
    pub uv1: [u16; 2],
    pub uv2: [u16; 2],
}

#[derive(Debug, Clone, Copy, Readable, Writable, PartialEq, Eq, Hash)]
pub enum MeshMaterialBlend {
    Opaque,
    AlphaBlend,
    AlphaTest(u16), // Normalized
}

#[derive(Debug, Clone, Readable, Writable, PartialEq, Eq, Hash)]
pub enum MaterialColorSource {
    Image(AssetReference, ImageAssetType),
    Color([u8; 4]),
    None,
}

impl MaterialColorSource {
    fn collect_images(&self, images: &mut HashSet<(AssetReference, ImageAssetType)>) {
        if let Self::Image(reference, ty) = self {
            images.insert((reference.clone(), *ty));
        }
    }

    fn compiled(&mut self) {
        if let Self::Image(reference, ty) = self {
            *self = Self::Image(reference.compiled(), *ty);
        }
    }
}

impl MaterialColorSource {
    pub fn color(color: [f32; 4]) -> Self {
        Self::Color([
            (color[0].clamp(0.0, 1.0) * u8::MAX as f32) as u8,
            (color[1].clamp(0.0, 1.0) * u8::MAX as f32) as u8,
            (color[2].clamp(0.0, 1.0) * u8::MAX as f32) as u8,
            (color[3].clamp(0.0, 1.0) * u8::MAX as f32) as u8,
        ])
    }
}

#[derive(Debug, Clone, Readable, Writable, PartialEq, Eq, Hash)]
pub struct MeshAssetMaterial {
    pub base_color: MaterialColorSource,
    pub normals: MaterialColorSource,
    pub metallic_roughness: MaterialColorSource,
    pub occlusion: MaterialColorSource,
    pub emissive: MaterialColorSource,
    pub emissive_power: u32, // Value * 1000
    pub blend: MeshMaterialBlend,
}

impl MeshAssetMaterial {
    fn collect_images(&self, images: &mut HashSet<(AssetReference, ImageAssetType)>) {
        self.base_color.collect_images(images);
        self.normals.collect_images(images);
        self.metallic_roughness.collect_images(images);
        self.occlusion.collect_images(images);
        self.emissive.collect_images(images);
    }

    fn compiled(&mut self) {
        self.base_color.compiled();
        self.normals.compiled();
        self.metallic_roughness.compiled();
        self.occlusion.compiled();
        self.emissive.compiled();
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
    pub vertex_offset: u32,
    pub surfaces: Vec<MeshSurfaceAsset>,
    pub positon_scale: f32,
    pub uv_scale: [f32; 2],
    pub bounds: ([f32; 3], f32),
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
pub struct SceneAsset {
    pub vertices: Vec<StaticMeshVertex>,
    pub indices: Vec<u16>,
    pub meshes: Vec<StaticMeshAsset>,
    pub nodes: Vec<Node>,
    pub mesh_names: HashMap<String, u32>,
    pub node_names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
    pub materials: Vec<MeshAssetMaterial>,
}

impl SceneAsset {
    pub fn collect_dependencies(&self) -> HashSet<(AssetReference, ImageAssetType)> {
        let mut result = HashSet::new();
        for material in &self.materials {
            material.collect_images(&mut result);
        }
        result
    }

    pub fn compiled(&mut self) {
        self.materials.iter_mut().for_each(|x| x.compiled())
    }
}

impl Asset for SceneAsset {
    const TYPE: uuid::Uuid = uuid!("3d731621-54b4-40b0-a089-37667f68fe35");
    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }

    fn serialize<W: std::io::Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }
}

struct GltfProcessingContext<'a> {
    pub base_path: &'a str,
    pub buffers: Vec<gltf::buffer::Data>,
    pub vertices: Vec<StaticMeshVertex>,
    pub indices: Vec<u16>,
    pub materials: Vec<MeshAssetMaterial>,
}

struct NodeProcessingContext<'a> {
    context: &'a mut GltfProcessingContext<'a>,
    bone_to_mesh: HashMap<u32, u32>,
    bones: Vec<Node>,
    bone_names: HashMap<String, u32>,
    meshes: Vec<StaticMeshAsset>,
    mesh_names: HashMap<String, u32>,
    processed_meshes: HashMap<u32, u32>,
}

fn process_texture(
    context: &GltfProcessingContext,
    texture: &gltf::texture::Texture,
    ty: ImageAssetType,
) -> MaterialColorSource {
    match texture.source().source() {
        gltf::image::Source::Uri { uri, .. } => MaterialColorSource::Image(
            AssetReference::new(
                Path::new(&context.base_path)
                    .join(uri)
                    .normalize()
                    .to_str()
                    .unwrap(),
            ),
            ty,
        ),
        _ => panic!(),
    }
}

fn process_blend(material: &gltf::Material) -> MeshMaterialBlend {
    match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => MeshMaterialBlend::Opaque,
        gltf::material::AlphaMode::Mask => MeshMaterialBlend::AlphaTest(
            (material.alpha_cutoff().unwrap_or(0.0).clamp(0.0, 1.0) * u16::MAX as f32) as u16,
        ),
        gltf::material::AlphaMode::Blend => MeshMaterialBlend::AlphaBlend,
    }
}

fn process_material(
    context: &GltfProcessingContext,
    material: gltf::Material,
) -> MeshAssetMaterial {
    let base_color = if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Srgba)
    } else {
        MaterialColorSource::color(material.pbr_metallic_roughness().base_color_factor())
    };
    let metallic_roughness = if let Some(texture) = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
    {
        process_texture(context, &texture.texture(), ImageAssetType::Rgba)
    } else {
        MaterialColorSource::color([
            0.0,
            material.pbr_metallic_roughness().roughness_factor(),
            material.pbr_metallic_roughness().metallic_factor(),
            1.0,
        ])
    };
    let normals = if let Some(texture) = material.normal_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Rg)
    } else {
        MaterialColorSource::None
    };
    let occlusion = if let Some(texture) = material.occlusion_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Rgba)
    } else {
        MaterialColorSource::None
    };
    let emissive = if let Some(texture) = material.emissive_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Rgba)
    } else {
        let emissive_color = material.emissive_factor();
        MaterialColorSource::color([emissive_color[0], emissive_color[1], emissive_color[2], 1.0])
    };
    MeshAssetMaterial {
        emissive_power: (material.emissive_strength().unwrap_or(0.0) * 1000.0) as u32,
        blend: process_blend(&material),
        base_color,
        normals,
        metallic_roughness,
        occlusion,
        emissive,
    }
}

fn process_mesh(
    context: &mut GltfProcessingContext,
    mesh: gltf::Mesh,
) -> Result<StaticMeshAsset, Error> {
    let mut builder = MeshAssetBuilder::default();
    for prim in mesh.primitives() {
        let mut surface = MeshSurfaceBuilder::new(process_material(context, prim.material()));
        if prim.mode() != Mode::Triangles {
            return Err(Error::ProcessingFailed(
                "Only processing triangle meshes".into(),
            ));
        }
        let reader = prim.reader(|buffer| Some(&context.buffers[buffer.index()]));
        if let Some(positions) = reader.read_positions() {
            surface.push_position(&positions.collect::<Vec<_>>());
        } else {
            return Err(Error::ProcessingFailed("Mesh has no positions".into()));
        };
        if let Some(indices) = reader.read_indices() {
            surface.push_indices(&indices.into_u32().collect::<Vec<_>>());
        } else {
            return Err(Error::ProcessingFailed(
                "Only processing indexed meshes".into(),
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
) -> Result<(), Error> {
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
            context.mesh_names.insert(name.to_owned(), mesh_index);
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

fn import_scene<'a>(
    context: &'a mut GltfProcessingContext<'a>,
    scene: gltf::Scene,
) -> Result<SceneAsset, Error> {
    let mut context = NodeProcessingContext {
        context,
        bone_to_mesh: Default::default(),
        meshes: Default::default(),
        bones: Default::default(),
        bone_names: Default::default(),
        mesh_names: Default::default(),
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
        SceneAsset {
            vertices: context.context.vertices.clone(),
            indices: context.context.indices.clone(),
            meshes: context.meshes,
            nodes: context.bones,
            mesh_names: context.mesh_names,
            node_names: context.bone_names,
            node_to_mesh: context.bone_to_mesh.into_iter().collect::<Vec<_>>(),
            materials: context.context.materials.clone(),
        }
    })
}

fn import_scenes<'a>(
    context: &'a mut GltfProcessingContext<'a>,
    document: gltf::Document,
) -> Result<SceneAsset, Error> {
    let scene = document
        .default_scene()
        .ok_or(Error::ImportFailed("Default scene not found".to_owned()))?;
    import_scene(context, scene)
}

impl ImportAsset<SceneAsset> for GltfSceneSource {
    fn import(&self) -> Result<SceneAsset, Error> {
        let (document, buffers, _) = gltf::import(get_absolute_asset_path(&self.0)?)
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;
        let base_path = get_relative_asset_path(&self.0)?
            .parent()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        import_scenes(
            &mut GltfProcessingContext {
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
