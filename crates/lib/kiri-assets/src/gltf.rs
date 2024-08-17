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
    path::{Path, PathBuf},
    time::SystemTime,
};

use gltf::mesh::Mode;
use normalize_path::NormalizePath;
use siphasher::sip::SipHasher;
use speedy::{Readable, Writable};

use crate::{
    get_absolute_asset_path, get_relative_asset_path, is_asset_changed, Asset, AssetImportContext,
    AssetReference, AssetSource, Error, ImageAssetSource, ImageAssetType, ImportAsset,
    MeshAssetBuilder, MeshSurfaceBuilder,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GltfSceneSource(PathBuf);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GltfMeshSource {
    pub gltf: PathBuf,
    pub scene: String,
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
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self(p.as_ref().to_owned())
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

#[derive(Debug, Clone, Copy, Readable, Writable)]
pub struct BoneIndex(u32);

impl From<u32> for BoneIndex {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Default for BoneIndex {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

impl BoneIndex {
    pub fn new(index: Option<u32>) -> BoneIndex {
        if let Some(index) = index {
            Self(index)
        } else {
            Self(u32::MAX)
        }
    }

    pub fn index(self) -> Option<u32> {
        if self.0 < u32::MAX {
            Some(self.0)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable)]
pub struct GltfBone {
    pub parent: BoneIndex,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Debug, Readable, Writable)]
pub struct GltfSceneAsset {
    pub meshes: Vec<AssetReference>,
    pub bones: Vec<GltfBone>,
    pub mesh_names: HashMap<String, u32>,
    pub bone_names: HashMap<String, u32>,
    pub bone_to_mesh: Vec<(u32, u32)>,
}

#[derive(Debug, Readable, Writable)]
pub struct GltfAsset {
    pub scenes: HashMap<String, GltfSceneAsset>,
}

impl Asset for StaticMeshAsset {
    fn load(data: bytes::Bytes) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(&data)?)
    }

    fn save(&self) -> std::io::Result<bytes::Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

impl Asset for GltfAsset {
    fn load(data: bytes::Bytes) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(&data)?)
    }

    fn save(&self) -> std::io::Result<bytes::Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

struct GltfProcessingContext<'a> {
    pub asset_importer: &'a dyn AssetImportContext,
    pub base_path: PathBuf,
    pub gltf_path: PathBuf,
    pub buffers: Vec<gltf::buffer::Data>,
}

struct NodeProcessingContext<'a> {
    context: &'a GltfProcessingContext<'a>,
    scene_name: &'a str,
    bone_to_mesh: HashMap<u32, u32>,
    bones: Vec<GltfBone>,
    bone_names: HashMap<String, u32>,
    meshes: Vec<AssetReference>,
    mesh_names: HashMap<String, u32>,
    processed_meshes: HashMap<u32, u32>,
}

fn process_texture(
    context: &GltfProcessingContext,
    texture: &gltf::texture::Texture,
    srgb: bool,
    ty: ImageAssetType,
) -> AssetReference {
    match texture.source().source() {
        gltf::image::Source::Uri { uri, .. } => {
            let image_path = context.base_path.join(uri).normalize();
            context
                .asset_importer
                .import_image(ImageAssetSource::from_file(image_path).srgb(srgb).ty(ty))
        }
        _ => panic!(),
    }
}

fn process_placeholder(
    context: &GltfProcessingContext,
    color: [f32; 4],
    srgb: bool,
    ty: ImageAssetType,
) -> AssetReference {
    context.asset_importer.import_image(
        ImageAssetSource::from_color(color.map(|x| (x.clamp(0.0, 1.0) * 255.0) as u8))
            .srgb(srgb)
            .ty(ty),
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
        process_texture(context, &texture.texture(), true, ImageAssetType::Rgba)
    } else {
        process_placeholder(
            context,
            material.pbr_metallic_roughness().base_color_factor(),
            true,
            ImageAssetType::Rgba,
        )
    };
    let metallic_roughness = if let Some(texture) = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
    {
        process_texture(context, &texture.texture(), false, ImageAssetType::Rgba)
    } else {
        process_placeholder(
            context,
            [
                0.0,
                material.pbr_metallic_roughness().roughness_factor(),
                material.pbr_metallic_roughness().metallic_factor(),
                1.0,
            ],
            false,
            ImageAssetType::Rgba,
        )
    };
    let normals = if let Some(texture) = material.normal_texture() {
        process_texture(context, &texture.texture(), false, ImageAssetType::Rg)
    } else {
        process_placeholder(context, [0.0, 0.0, 1.0, 1.0], false, ImageAssetType::Rg)
    };
    let occlusion = if let Some(texture) = material.occlusion_texture() {
        process_texture(context, &texture.texture(), false, ImageAssetType::Rgba)
    } else {
        process_placeholder(context, [1.0, 0.0, 0.0, 1.0], false, ImageAssetType::Rgba)
    };
    let emissive = if let Some(texture) = material.emissive_texture() {
        process_texture(context, &texture.texture(), false, ImageAssetType::Rgba)
    } else {
        let emissive_color = material.emissive_factor();
        process_placeholder(
            context,
            [emissive_color[0], emissive_color[1], emissive_color[2], 1.0],
            false,
            ImageAssetType::Rgba,
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
    scene_name: &str,
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
            scene: scene_name.into(),
            mesh: mesh_name.into(),
        },
        builder,
    )))
}

fn process_node(
    context: &mut NodeProcessingContext,
    parent_index: BoneIndex,
    name: &str,
    node: gltf::Node,
) -> Result<(), Error> {
    let bone_index = context.bones.len() as u32;
    let (translation, rotation, scale) = node.transform().decomposed();
    context.bones.push(GltfBone {
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

            if let Some(mesh) = process_mesh(context.context, context.scene_name, name, mesh)? {
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
            BoneIndex::new(Some(bone_index)),
            &format!("{}/{}", name, child.name().unwrap_or(&format!("{}", index))),
            child,
        )?;
    }
    Ok(())
}

fn import_scene(
    context: &GltfProcessingContext,
    name: &str,
    scene: gltf::Scene,
) -> Result<GltfSceneAsset, Error> {
    let mut context = NodeProcessingContext {
        context,
        scene_name: name,
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
            BoneIndex::new(None),
            node.name().unwrap_or(&format!("{}", index)),
            node,
        )?;
    }
    Ok({
        GltfSceneAsset {
            meshes: context.meshes,
            bones: context.bones,
            mesh_names: context.mesh_names,
            bone_names: context.bone_names,
            bone_to_mesh: context.bone_to_mesh.into_iter().collect::<Vec<_>>(),
        }
    })
}

fn import_scenes(
    context: GltfProcessingContext,
    document: gltf::Document,
) -> Result<GltfAsset, Error> {
    let mut scenes = HashMap::new();
    for scene in document.scenes() {
        let name = scene.name().unwrap_or("main");
        if !name.starts_with("_") && !name.starts_with("!") {
            scenes.insert(name.to_owned(), import_scene(&context, name, scene)?);
        }
    }
    Ok(GltfAsset { scenes })
}

impl ImportAsset<GltfSceneSource> for GltfAsset {
    fn import(
        source: GltfSceneSource,
        asset_importer: &dyn AssetImportContext,
    ) -> Result<Self, Error> {
        let (document, buffers, _) = gltf::import(get_absolute_asset_path(&source.0)?)
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;
        let base_path = get_relative_asset_path(&source.0)?
            .parent()
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
