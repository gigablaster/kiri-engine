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
    collections::HashMap,
    hash::{Hash, Hasher},
    path::Path,
    time::SystemTime,
};

use gltf::mesh::Mode;
use kiri_backend::{Format, InputVertexAttrubute, InputVertexStreamLayout, PipelineVertex};
use normalize_path::NormalizePath;
use siphasher::sip::SipHasher;
use speedy::{Readable, Writable};

use crate::{
    get_absolute_asset_path, get_relative_asset_path, is_asset_changed, Asset, AssetImportContext,
    AssetReference, AssetSource, Error, ImageAssetSource, ImageAssetType, ImportAsset,
    MeshAssetBuilder, MeshSurfaceBuilder,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GltfSceneSource(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GltfMeshSource {
    pub gltf: String,
    pub mesh: String,
}

impl AssetSource for GltfSceneSource {
    fn reference(&self) -> AssetReference {
        let mut hasher = SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish().into()
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

impl AssetSource for GltfMeshSource {
    fn reference(&self) -> AssetReference {
        let mut hasher = SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish().into()
    }

    fn changed(&self, last_update: SystemTime) -> bool {
        is_asset_changed(&self.gltf, last_update)
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable)]
#[repr(C, align(8))]
pub struct StaticMeshVertex {
    pub position: [u16; 3],
    pub _pad: u16,
    pub normal: [u16; 2],
    pub tangent: [u16; 2],
    pub uv1: [u16; 2],
    pub uv2: [u16; 2],
}

const STATIC_MESH_INPUT_LAYOUT: &[InputVertexStreamLayout] = &[InputVertexStreamLayout {
    streams: &[
        InputVertexAttrubute {
            format: Format::RGB16_UNORM,
            offset: 0,
        },
        InputVertexAttrubute {
            format: Format::RG16_UNORM,
            offset: 8,
        },
        InputVertexAttrubute {
            format: Format::RG16_UNORM,
            offset: 12,
        },
        InputVertexAttrubute {
            format: Format::RG16_UNORM,
            offset: 16,
        },
        InputVertexAttrubute {
            format: Format::RG16_UNORM,
            offset: 20,
        },
    ],
}];

impl PipelineVertex for StaticMeshVertex {
    fn layout() -> &'static [InputVertexStreamLayout<'static>] {
        STATIC_MESH_INPUT_LAYOUT
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable)]
pub enum MeshMaterialBlend {
    Opaque,
    AlphaBlend,
    AlphaTest(f32),
}

#[derive(Debug, Clone, Readable, Writable)]
pub struct MeshMaterialAsset {
    pub images: HashMap<String, (AssetReference, ImageAssetType)>,
    pub emissive_power: f32,
    pub blend: MeshMaterialBlend,
}

#[derive(Debug, Clone, Copy, Readable, Writable)]
pub struct MeshSurfaceAsset {
    pub first_index: u32,
    pub index_count: u32,
    pub material: u32,
}

#[derive(Debug, Readable, Writable)]
pub struct StaticMeshAsset {
    pub vertices: Vec<StaticMeshVertex>,
    pub indices: Vec<u16>,
    pub materials: Vec<MeshMaterialAsset>,
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
    pub meshes: Vec<AssetReference>,
    pub nodes: Vec<Node>,
    pub mesh_names: HashMap<String, u32>,
    pub node_names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
}

impl Asset for StaticMeshAsset {
    fn load(data: bytes::Bytes) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(&data)?)
    }

    fn save(&self) -> std::io::Result<bytes::Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

impl Asset for SceneAsset {
    fn load(data: bytes::Bytes) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(&data)?)
    }

    fn save(&self) -> std::io::Result<bytes::Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

struct GltfProcessingContext<'a> {
    pub asset_importer: &'a dyn AssetImportContext,
    pub base_path: String,
    pub gltf_path: String,
    pub buffers: Vec<gltf::buffer::Data>,
}

struct NodeProcessingContext<'a> {
    context: &'a GltfProcessingContext<'a>,
    bone_to_mesh: HashMap<u32, u32>,
    bones: Vec<Node>,
    bone_names: HashMap<String, u32>,
    meshes: Vec<AssetReference>,
    mesh_names: HashMap<String, u32>,
    processed_meshes: HashMap<u32, u32>,
}

fn process_texture(
    context: &GltfProcessingContext,
    texture: &gltf::texture::Texture,
    ty: ImageAssetType,
) -> AssetReference {
    match texture.source().source() {
        gltf::image::Source::Uri { uri, .. } => {
            let image_path = Path::new(&context.base_path)
                .join(uri)
                .normalize()
                .to_str()
                .unwrap()
                .to_owned();
            context
                .asset_importer
                .import_image(ImageAssetSource::from_file(&image_path).ty(ty))
        }
        _ => panic!(),
    }
}

fn process_placeholder(
    context: &GltfProcessingContext,
    color: [f32; 4],
    ty: ImageAssetType,
) -> AssetReference {
    context.asset_importer.import_image(
        ImageAssetSource::from_color(color.map(|x| (x.clamp(0.0, 1.0) * 255.0) as u8)).ty(ty),
    )
}

fn process_blend(material: &gltf::Material) -> MeshMaterialBlend {
    match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => MeshMaterialBlend::Opaque,
        gltf::material::AlphaMode::Mask => {
            MeshMaterialBlend::AlphaTest(material.alpha_cutoff().unwrap_or(0.0))
        }
        gltf::material::AlphaMode::Blend => MeshMaterialBlend::AlphaBlend,
    }
}

fn process_material(
    context: &GltfProcessingContext,
    material: gltf::Material,
) -> MeshMaterialAsset {
    let base_color = if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Color)
    } else {
        process_placeholder(
            context,
            material.pbr_metallic_roughness().base_color_factor(),
            ImageAssetType::Color,
        )
    };
    let metallic_roughness = if let Some(texture) = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
    {
        process_texture(
            context,
            &texture.texture(),
            ImageAssetType::MetallicRoughness,
        )
    } else {
        process_placeholder(
            context,
            [
                0.0,
                material.pbr_metallic_roughness().roughness_factor(),
                material.pbr_metallic_roughness().metallic_factor(),
                1.0,
            ],
            ImageAssetType::MetallicRoughness,
        )
    };
    let normals = if let Some(texture) = material.normal_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Normal)
    } else {
        process_placeholder(context, [0.0, 0.0, 1.0, 1.0], ImageAssetType::Normal)
    };
    let occlusion = if let Some(texture) = material.occlusion_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Occlusion)
    } else {
        process_placeholder(context, [1.0, 0.0, 0.0, 1.0], ImageAssetType::Occlusion)
    };
    let emissive = if let Some(texture) = material.emissive_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Emissive)
    } else {
        let emissive_color = material.emissive_factor();
        process_placeholder(
            context,
            [emissive_color[0], emissive_color[1], emissive_color[2], 1.0],
            ImageAssetType::Emissive,
        )
    };
    MeshMaterialAsset {
        images: [
            ("base_color".into(), (base_color, ImageAssetType::Color)),
            ("normals".into(), (normals, ImageAssetType::Normal)),
            (
                "metallic_roughness".into(),
                (metallic_roughness, ImageAssetType::MetallicRoughness),
            ),
            ("occlusion".into(), (occlusion, ImageAssetType::Occlusion)),
            ("emissive".into(), (emissive, ImageAssetType::Emissive)),
        ]
        .into(),
        emissive_power: material.emissive_strength().unwrap_or(1.0),
        blend: process_blend(&material),
    }
}

fn process_mesh(
    context: &GltfProcessingContext,
    mesh_name: &str,
    mesh: gltf::Mesh,
) -> Result<Option<AssetReference>, Error> {
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
            return Ok(None);
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
    Ok(Some(context.asset_importer.import_static_mesh(
        GltfMeshSource {
            gltf: context.gltf_path.clone(),
            mesh: mesh_name.into(),
        },
        builder,
    )))
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

            if let Some(mesh) = process_mesh(context.context, name, mesh)? {
                let mesh_index = context.meshes.len() as u32;
                context.meshes.push(mesh);
                context.mesh_names.insert(name.to_owned(), mesh_index);
                context.bone_to_mesh.insert(bone_index, mesh_index);
            }
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

fn import_scene(context: &GltfProcessingContext, scene: gltf::Scene) -> Result<SceneAsset, Error> {
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
            meshes: context.meshes,
            nodes: context.bones,
            mesh_names: context.mesh_names,
            node_names: context.bone_names,
            node_to_mesh: context.bone_to_mesh.into_iter().collect::<Vec<_>>(),
        }
    })
}

fn import_scenes(
    context: GltfProcessingContext,
    document: gltf::Document,
) -> Result<SceneAsset, Error> {
    let scene = document
        .default_scene()
        .ok_or(Error::ImportFailed("Default scene not found".to_owned()))?;
    import_scene(&context, scene)
}

impl ImportAsset<GltfSceneSource> for SceneAsset {
    fn import(
        source: GltfSceneSource,
        asset_importer: &dyn AssetImportContext,
    ) -> Result<Self, Error> {
        let (document, buffers, _) = gltf::import(get_absolute_asset_path(&source.0)?)
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;
        let base_path = get_relative_asset_path(&source.0)?
            .parent()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        import_scenes(
            GltfProcessingContext {
                gltf_path: source.0,
                asset_importer,
                base_path,
                buffers,
            },
            document,
        )
    }
}
