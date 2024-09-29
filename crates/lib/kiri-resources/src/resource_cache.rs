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

use std::{collections::HashMap, fmt::Debug, hash::Hash, mem, sync::Arc};

use bevy_tasks::{block_on, IoTaskPool, Task};
#[cfg(feature = "devel")]
use kiri_assets::{
    get_compiled_asset_change_time, get_compiled_asset_path, save_asset, ImportAsset,
};
use kiri_assets::{
    load_asset, Asset, AssetSource, ImageAsset, ImageSource, MeshAssetMaterial,
    MeshVertexAttributes, MeshVertexPositions, ModelAsset, ModelSource,
};
use kiri_backend::{
    ash::vk, BufferCreateDesc, DescriptorSetDesc, DescriptorSetLayoutDesc, ImageCreateDesc,
};
use kiri_common::{Handle, Pool};
use kiri_gfx::{
    BufferPointer, DescriptorHandle, DescriptorSetBuilder, ImageHandle, ImageUploadData, Renderer,
};
use kiri_math::{Affine3A, BoundingBox, Quat, Vec3};
use kiri_vfs::{vfs_load, AssetReference};
use log::debug;
#[cfg(feature = "devel")]
use log::warn;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard};
#[cfg(feature = "devel")]
use std::{fs::File, io};

use crate::{
    ConstUniformBuffer, Error, RenderMaterial, RenderMeshSurface, RenderModel, StaticRenderMesh,
};

pub type ModelHandle = Handle<RenderModel>;

type ModelPool = Pool<RenderModel>;

type ImageLoadingTask = Task<Result<(ImageHandle, ImageAsset, ImageSource), Error>>;
type SceneLoadingTask = Task<Result<(ModelHandle, ModelAsset, ModelSource), Error>>;

pub const MATERIAL_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[
        (
            0,
            DescriptorSetDesc {
                name: "data",
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
            },
        ),
        (
            1,
            DescriptorSetDesc {
                name: "base_color",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            2,
            DescriptorSetDesc {
                name: "normals",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            3,
            DescriptorSetDesc {
                name: "metallic_roughness",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            4,
            DescriptorSetDesc {
                name: "occlusion",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            5,
            DescriptorSetDesc {
                name: "emissive",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
    ],
    update_after_bind: false,
};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct MeshMaterialData {
    pub alpha_cutoff: f32,
    pub emissive_power: f32,
}

/// Interaface to resource access
pub trait ResourceResolver {
    fn resolve_static_mesh(&self, handle: ModelHandle, index: u32) -> Option<&StaticRenderMesh>;
    fn resolve_model(&self, handle: ModelHandle) -> Option<&RenderModel>;
}

/// Keeps normalized asset name -> asset + ref count.
///
/// T must be a handle
#[derive(Debug, Default)]
struct AssetTracker<T: Copy + Hash + Eq> {
    assets: RwLock<HashMap<AssetReference, T>>,
}

impl<T: Copy + Hash + Eq> AssetTracker<T> {
    /// Return asset if exists and increase ref count
    fn get_or_load<U: AssetSource, LOAD: FnOnce(&U) -> Result<T, Error>>(
        &self,
        source: &U,
        load: LOAD,
    ) -> Result<T, Error> {
        let assets = self.assets.upgradable_read();
        let key = source.reference();
        if let Some(asset) = assets.get(&key) {
            Ok(*asset)
        } else {
            let mut assets = RwLockUpgradableReadGuard::upgrade(assets);
            if let Some(asset) = assets.get(&key) {
                Ok(*asset)
            } else {
                let asset = load(source)?;
                assets.insert(key, asset);
                Ok(asset)
            }
        }
    }
}

#[cfg(not(feature = "devel"))]
pub(crate) fn load_or_compile_asset<T: AssetSource + Debug, U: Asset>(
    source: &T,
) -> Result<U, Error> {
    let reference = source.reference();
    let reader = vfs_load(reference)?;
    Ok(load_asset(reader)?)
}

#[cfg(feature = "devel")]
pub(crate) fn load_or_compile_asset<T: AssetSource + ImportAsset<U> + Debug, U: Asset>(
    source: &T,
) -> Result<U, Error> {
    let reference = source.reference();
    let newer = get_compiled_asset_change_time(reference)
        .map(|x| source.changed(x))
        .unwrap_or(false);
    if !newer {
        if let Ok(reader) = vfs_load(reference) {
            debug!("Loading asset: {:?}", source);
            return Ok(load_asset(reader)?);
        }
    }
    // There's no compiled asset, so compile it in runtime
    warn!("Compile asset: {:?}", source);
    let asset = source.import()?;
    if let Err(err) = try_save_asset(reference, &asset) {
        warn!("Failed to save compiled asset to cache: {}", err);
    }
    Ok(asset)
}

#[cfg(feature = "devel")]
fn try_save_asset<T: Asset>(reference: AssetReference, asset: &T) -> io::Result<()> {
    save_asset(File::create(get_compiled_asset_path(reference)?)?, asset)
}

const MATERIAL_BUFFER_SIZE: u64 = 2 * 1024 * 1024;
const MAX_RESOURCES: usize = 0xffff;

#[derive(Debug)]
pub struct ResourceCache {
    pub renderer: Arc<Renderer>,
    // loading_tasks: Mutex<Vec<LoadingTask>>,
    image_loading_tasks: Mutex<Vec<ImageLoadingTask>>,
    scene_loading_tasks: Mutex<Vec<SceneLoadingTask>>,
    images: AssetTracker<ImageHandle>,
    scenes: AssetTracker<ModelHandle>,
    scene_assets: RwLock<ModelPool>,
    material_uniforms: ConstUniformBuffer,
    materials: RwLock<HashMap<MeshAssetMaterial, DescriptorHandle>>,
}

impl ResourceCache {
    pub fn new(renderer: &Arc<Renderer>) -> Result<Arc<Self>, Error> {
        debug!("Create resource manager");
        Ok(Arc::new(Self {
            renderer: renderer.clone(),
            // loading_tasks: Default::default(),
            image_loading_tasks: Default::default(),
            scene_loading_tasks: Default::default(),
            images: Default::default(),
            materials: Default::default(),
            material_uniforms: ConstUniformBuffer::new(renderer, MATERIAL_BUFFER_SIZE)?,
            scenes: Default::default(),
            scene_assets: RwLock::new(ModelPool::new(MAX_RESOURCES)),
        }))
    }

    pub fn tick(&self) -> Result<(), Error> {
        {
            let mut loading = self.scene_loading_tasks.lock();
            let mut i = 0;
            while i < loading.len() {
                if loading[i].is_finished() {
                    let task = loading.remove(i);
                    let (handle, asset, source) = block_on(task)?;
                    self.process_scene(handle, asset, source)?;
                } else {
                    i += 1;
                }
            }
        }
        {
            let mut loading = self.image_loading_tasks.lock();
            let mut i = 0;
            while i < loading.len() {
                if loading[i].is_finished() {
                    let task = loading.remove(i);
                    let (handle, asset, source) = block_on(task)?;
                    self.process_image(handle, asset, source)?;
                } else {
                    i += 1;
                }
            }
        }

        Ok(())
    }

    pub fn resolve(&self) -> ResourceCacheMeshResolver {
        ResourceCacheMeshResolver {
            scens: self.scene_assets.read(),
        }
    }

    pub fn get_or_load_image(
        &self,
        source: &ImageSource,
        default_color: [u8; 4],
    ) -> Result<ImageHandle, Error> {
        let source = source.clone();
        self.images.get_or_load(&source, |source| {
            self.load_image_impl(source, default_color)
        })
    }

    fn load_image_impl(
        &self,
        source: &ImageSource,
        default_color: [u8; 4],
    ) -> Result<ImageHandle, Error> {
        let handle = match &source.data {
            kiri_assets::ImageData::Path(_) => {
                let handle = self.renderer.create_image(
                    ImageCreateDesc::texture(source.uncompressed_format(), [1, 1])
                        .name(&format!("{:?} - DUMMY", source)),
                    Some(&[ImageUploadData {
                        data: &default_color,
                    }]),
                )?;
                self.image_loading_tasks
                    .lock()
                    .push(IoTaskPool::get().spawn(Self::load_image(handle, source.clone())));
                handle
            }
            kiri_assets::ImageData::Color(color) => self.renderer.create_image(
                ImageCreateDesc::texture(source.uncompressed_format(), [1, 1])
                    .name(&format!("{:?}", source)),
                Some(&[ImageUploadData { data: color }]),
            )?,
        };

        Ok(handle)
    }

    fn process_image(
        &self,
        handle: ImageHandle,
        asset: ImageAsset,
        source: ImageSource,
    ) -> Result<(), Error> {
        let upload = asset
            .mips
            .iter()
            .map(|x| ImageUploadData { data: x })
            .collect::<Vec<_>>();
        self.renderer.update_image(
            handle,
            ImageCreateDesc::texture(asset.format, asset.dims)
                .mip_levels(asset.mips.len() as _)
                .name(&format!("{:?}", source)),
            Some(&upload),
        )?;
        Ok(())
    }

    pub fn get_or_load_material(
        &self,
        material: &MeshAssetMaterial,
    ) -> Result<DescriptorHandle, Error> {
        let materials = self.materials.upgradable_read();
        if let Some(material) = materials.get(material) {
            Ok(*material)
        } else {
            let mut materials = RwLockUpgradableReadGuard::upgrade(materials);
            if let Some(material) = materials.get(material) {
                Ok(*material)
            } else {
                // Get all images

                // Allocate and copy uniform data
                let builder = DescriptorSetBuilder::new(
                    vk::ShaderStageFlags::ALL_GRAPHICS,
                    MATERIAL_DESCRIPTOR_LAYOUT,
                )
                .bind_uniform_buffer(
                    0,
                    self.material_uniforms.push(MeshMaterialData {
                        alpha_cutoff: material.blend.get_alpha_cut(),
                        emissive_power: material.emissive_power,
                    })?,
                )
                .bind_image(
                    1,
                    self.get_or_load_image(&material.base_color, [127, 127, 127, 255])?,
                    vk::ImageAspectFlags::COLOR,
                )
                .bind_image(
                    2,
                    self.get_or_load_image(&material.normals, [127, 127, 255, 255])?,
                    vk::ImageAspectFlags::COLOR,
                )
                .bind_image(
                    3,
                    self.get_or_load_image(&material.metallic_roughness, [0, 255, 0, 255])?,
                    vk::ImageAspectFlags::COLOR,
                )
                .bind_image(
                    4,
                    self.get_or_load_image(&material.occlusion, [0, 0, 0, 0])?,
                    vk::ImageAspectFlags::COLOR,
                )
                .bind_image(
                    5,
                    self.get_or_load_image(&material.emissive, [0, 0, 0, 0])?,
                    vk::ImageAspectFlags::COLOR,
                );
                let descriptor_set = self.renderer.create_descriptor_set(builder)?;
                materials.insert(material.clone(), descriptor_set);
                Ok(descriptor_set)
            }
        }
    }

    pub fn get_or_load_scene(&self, name: &str) -> Result<ModelHandle, Error> {
        let source = ModelSource::new(name);
        self.scenes
            .get_or_load(&source, |source| self.load_scene_impl(source))
    }

    fn load_scene_impl(&self, source: &ModelSource) -> Result<ModelHandle, Error> {
        let handle = self.scene_assets.write().push(RenderModel::default());
        self.scene_loading_tasks
            .lock()
            .push(IoTaskPool::get().spawn(Self::load_scene(handle, source.clone())));
        Ok(handle)
    }

    fn process_scene(
        &self,
        handle: ModelHandle,
        asset: ModelAsset,
        source: ModelSource,
    ) -> Result<(), Error> {
        let mut materials = Vec::new();
        for material in &asset.materials {
            materials.push(RenderMaterial {
                ds: self.get_or_load_material(material)?,
                ty: material.blend.into(),
            });
        }

        // Create buffers
        let vertex_positions = self.renderer.create_buffer(
            BufferCreateDesc::gpu(
                (mem::size_of::<MeshVertexPositions>() * asset.vertex_positions.len()) as _,
            )
            .veretex_buffer()
            .transfer_destination()
            .name(&format!("{:?} - VB Pos", source)),
        )?;
        let vertex_attributes = self.renderer.create_buffer(
            BufferCreateDesc::gpu(
                (mem::size_of::<MeshVertexAttributes>() * asset.vertex_positions.len()) as _,
            )
            .veretex_buffer()
            .transfer_destination()
            .name(&format!("{:?} - VB Attr", source)),
        )?;
        let indices = self.renderer.create_buffer(
            BufferCreateDesc::gpu((mem::size_of::<u16>() * asset.indices.len()) as _)
                .index_buffer()
                .transfer_destination()
                .name(&format!("{:?} - IB", source)),
        )?;

        // Upload buffers
        self.renderer.upload_buffer(
            BufferPointer::new(vertex_positions, 0),
            &asset.vertex_positions,
        )?;
        self.renderer.upload_buffer(
            BufferPointer::new(vertex_attributes, 0),
            &asset.vertex_attributes,
        )?;
        self.renderer
            .upload_buffer(BufferPointer::new(indices, 0), &asset.indices)?;

        // Create meshes
        let mut meshes = Vec::new();
        let mut bounds = Vec::new();
        for mesh in asset.meshes {
            let surfaces = mesh
                .surfaces
                .into_iter()
                .map(|x| RenderMeshSurface {
                    first_index: x.first_index + mesh.first_index as u32,
                    index_count: x.index_count,
                    vertex_offset: mesh.first_vertex as u32,
                    material: materials[x.material as usize],
                })
                .collect::<Vec<_>>();
            let mesh = StaticRenderMesh {
                vertex_positions: BufferPointer::new(vertex_positions, 0),
                vertex_attributes: BufferPointer::new(vertex_attributes, 0),
                index_buffer: BufferPointer::new(indices, 0),
                surfaces,
                bounds: BoundingBox::from_extent_array(mesh.bounds.0, mesh.bounds.1),
                position_scale: mesh.position_scale,
                uv_scale: mesh.uv_scale,
            };
            bounds.push(mesh.bounds);
            meshes.push(mesh);
        }
        let mut scene = RenderModel {
            vertex_positions,
            vertex_attributes,
            indices,
            meshes,
            bounds_per_mesh: bounds,
            names: asset.name_to_mesh,
            parents: asset.nodes.iter().map(|x| x.parent).collect(),
            local_transforms: asset
                .nodes
                .iter()
                .map(|x| {
                    Affine3A::from_scale_rotation_translation(
                        Vec3::from_array(x.scale),
                        Quat::from_array(x.rotation),
                        Vec3::from_array(x.translation),
                    )
                })
                .collect(),
            world_transforms: asset.nodes.iter().map(|_| Affine3A::IDENTITY).collect(),
            node_to_mesh: asset.node_to_mesh,
            mesh_names: asset.mesh_names,
            bounds: Default::default(),
        };
        scene.update_world_transforms();
        self.scene_assets.write().replace(handle, scene);
        Ok(())
    }

    async fn load_image(
        handle: ImageHandle,
        source: ImageSource,
    ) -> Result<(ImageHandle, ImageAsset, ImageSource), Error> {
        Ok((handle, load_or_compile_asset(&source)?, source))
    }

    async fn load_scene(
        handle: ModelHandle,
        source: ModelSource,
    ) -> Result<(ModelHandle, ModelAsset, ModelSource), Error> {
        Ok((handle, load_or_compile_asset(&source)?, source))
    }
}

impl Drop for ResourceCache {
    fn drop(&mut self) {
        self.images
            .assets
            .write()
            .drain()
            .for_each(|(_, handle)| self.renderer.destroy_image(handle));
        self.scene_assets.write().drain().for_each(|scene| {
            self.renderer.destroy_buffer(scene.vertex_positions);
            self.renderer.destroy_buffer(scene.vertex_positions);
        });
        self.materials
            .write()
            .drain()
            .for_each(|(_, ds)| self.renderer.destroy_descriptor_set(ds));
    }
}

pub struct ResourceCacheMeshResolver<'a> {
    scens: RwLockReadGuard<'a, ModelPool>,
}

impl<'a> ResourceResolver for ResourceCacheMeshResolver<'a> {
    fn resolve_static_mesh(&self, handle: ModelHandle, index: u32) -> Option<&StaticRenderMesh> {
        let scene = self.scens.get(handle)?;
        Some(&scene.meshes[index as usize])
    }

    fn resolve_model(&self, handle: ModelHandle) -> Option<&RenderModel> {
        self.scens.get(handle)
    }
}
