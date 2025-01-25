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

use std::{collections::HashMap, io, time::SystemTime};

use kiri_assets::{EmbeddedImage, ModelAsset, SourceAssetPath};
use kiri_backend::ash::vk;

use crate::{AssetPipelineContext, AssetSource};
use crate::{ImageAssetType, ImportAsset};

use gltf::mesh::Mode;
use kiri_assets::{
    ImageReference, MeshAssetMaterial, MeshMaterialBlend, Node, NodeIndex, RenderMeshVertex,
    StaticMeshAsset,
};

use crate::{
    mesh_builder::{MeshAssetBuilder, MeshSurfaceBuilder},
    ImageSource,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelSource(SourceAssetPath);

impl AssetSource for ModelSource {
    fn changed(&self, timestamp: SystemTime) -> bool {
        self.0.changed(timestamp)
    }

    fn source(&self) -> &SourceAssetPath {
        &self.0
    }
}

pub struct GltfProcessingContext<'a, T: AssetPipelineContext> {
    pub pipeline: &'a T,
    pub base_path: &'a str,
    pub buffers: Vec<gltf::buffer::Data>,
    pub vertices: Vec<RenderMeshVertex>,
    pub indices: Vec<u16>,
    pub materials: Vec<MeshAssetMaterial>,
}

pub struct NodeProcessingContext<'a, T: AssetPipelineContext> {
    context: &'a mut GltfProcessingContext<'a, T>,
    bone_to_mesh: HashMap<u32, u32>,
    bones: Vec<Node>,
    bone_names: HashMap<String, u32>,
    meshes: Vec<StaticMeshAsset>,
    mesh_names: Vec<String>,
    name_to_mesh: HashMap<String, u32>,
    processed_meshes: HashMap<u32, u32>,
}

fn process_texture<T: AssetPipelineContext>(
    context: &GltfProcessingContext<T>,
    texture: &gltf::texture::Texture,
    ty: ImageAssetType,
    srgb: bool,
) -> ImageReference {
    match texture.source().source() {
        gltf::image::Source::Uri { uri, .. } => {
            ImageReference::External(context.pipeline.import_image(ImageSource {
                source: SourceAssetPath::new(format!("{}/{}", context.base_path, uri)),
                ty,
                srgb,
            }))
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

fn color(color: [f32; 4], format: vk::Format) -> EmbeddedImage {
    EmbeddedImage::color(
        [
            (color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (color[2].clamp(0.0, 1.0) * 255.0) as u8,
            (color[3].clamp(0.0, 1.0) * 255.0) as u8,
        ],
        format,
    )
}

fn process_material<T: AssetPipelineContext>(
    context: &GltfProcessingContext<T>,
    material: gltf::Material,
) -> MeshAssetMaterial {
    let base_color = if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Rgba, true)
    } else {
        ImageReference::Embedded(color(
            material.pbr_metallic_roughness().base_color_factor(),
            vk::Format::A8B8G8R8_SRGB_PACK32,
        ))
    };
    let metallic_roughness = if let Some(texture) = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
    {
        process_texture(context, &texture.texture(), ImageAssetType::Rgba, false)
    } else {
        ImageReference::Embedded(color(
            [
                0.0,
                material.pbr_metallic_roughness().roughness_factor(),
                material.pbr_metallic_roughness().metallic_factor(),
                1.0,
            ],
            vk::Format::A8B8G8R8_UNORM_PACK32,
        ))
    };
    let normals = if let Some(texture) = material.normal_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Rg, false)
    } else {
        ImageReference::Embedded(EmbeddedImage::color(
            [127, 127, 255, 255],
            vk::Format::A8B8G8R8_UNORM_PACK32,
        ))
    };
    let occlusion = if let Some(texture) = material.occlusion_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Rgba, false)
    } else {
        ImageReference::Embedded(EmbeddedImage::color(
            [0, 0, 0, 0],
            vk::Format::A8B8G8R8_UNORM_PACK32,
        ))
    };
    let emissive_color = material.emissive_factor();
    let emissive = if let Some(texture) = material.emissive_texture() {
        process_texture(context, &texture.texture(), ImageAssetType::Rgba, false)
    } else {
        ImageReference::Embedded(color(
            [emissive_color[0], emissive_color[1], emissive_color[2], 1.0],
            vk::Format::A8B8G8R8_UNORM_PACK32,
        ))
    };
    MeshAssetMaterial {
        base_color,
        metallic_roughness,
        normals,
        occlusion,
        emissive,
        emissive_power: material.emissive_strength().unwrap_or_default(),
        blend: process_blend(&material),
    }
}

fn process_mesh<T: AssetPipelineContext>(
    context: &mut GltfProcessingContext<T>,
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

fn process_node<T: AssetPipelineContext>(
    context: &mut NodeProcessingContext<T>,
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

pub fn import_scene<'a, T: AssetPipelineContext>(
    context: &'a mut GltfProcessingContext<'a, T>,
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

fn import_scenes<'a, T: AssetPipelineContext>(
    context: &'a mut GltfProcessingContext<'a, T>,
    document: gltf::Document,
) -> io::Result<ModelAsset> {
    let scene = document
        .default_scene()
        .ok_or(io::Error::other("Default scene not found".to_string()))?;
    import_scene(context, scene)
}

impl ImportAsset<ModelAsset> for ModelSource {
    fn import(self, context: &impl AssetPipelineContext) -> io::Result<ModelAsset> {
        let (document, buffers, _) = gltf::import(self.0.full_source_path())
            .map_err(|err| io::Error::other(err.to_string()))?;
        let base_path = self.0.parent();
        import_scenes(
            &mut GltfProcessingContext {
                pipeline: context,
                vertices: Default::default(),
                indices: Default::default(),
                materials: Default::default(),
                base_path: base_path.to_str().unwrap(),
                buffers,
            },
            document,
        )
    }
}
