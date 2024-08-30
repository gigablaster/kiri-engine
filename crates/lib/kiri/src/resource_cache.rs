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

use std::{collections::HashMap, fmt::Debug, fs::File, hash::Hash, io, mem, sync::Arc};

use ash::vk;
use bevy_tasks::{block_on, IoTaskPool, Task};
use kiri_assets::{
    get_compiled_asset_change_time, get_compiled_asset_path, load_asset, save_asset, Asset,
    AssetSource, ImageAsset, ImageSource, ImportAsset, MeshAssetMaterial, ModelAsset, ModelSource,
};
use kiri_backend::{BufferCreateDesc, DescriptorSetDesc, DescriptorSetLayoutDesc, ImageCreateDesc};
use kiri_common::{Handle, Pool};
use kiri_gfx::{
    BufferPointer, DescriptorHandle, DescriptorSetBuilder, ImageHandle, ImageUploadData, Renderer,
};
use kiri_vfs::{vfs_load, AssetReference};
use log::{debug, warn};
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard};

use crate::{
    gpu::{GpuMeshMaterial, GpuStaticVertex},
    Bounds, ConstUniformBuffer, Error, MeshResolver, RenderMaterial, RenderMeshSurface,
    RenderModel, StaticRenderMesh,
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

pub(super) fn load_or_compile_asset<T: AssetSource + ImportAsset<U> + Debug, U: Asset>(
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
                    self.material_uniforms.push(GpuMeshMaterial {
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
        // let reference = source.reference();
        let vertex_data: Vec<GpuStaticVertex> =
            asset.vertices.into_iter().map(|x| x.into()).collect();
        let vertices = self.renderer.create_buffer(
            BufferCreateDesc::gpu((mem::size_of::<GpuStaticVertex>() * vertex_data.len()) as _)
                .veretex_buffer()
                .transfer_destination()
                .name(&format!("{:?} - VB", source)),
        )?;
        let indices = self.renderer.create_buffer(
            BufferCreateDesc::gpu((mem::size_of::<u16>() * asset.indices.len()) as _)
                .index_buffer()
                .transfer_destination()
                .name(&format!("{:?} - IB", source)),
        )?;
        self.renderer
            .upload_buffer(BufferPointer::new(vertices, 0), &vertex_data)?;
        self.renderer
            .upload_buffer(BufferPointer::new(indices, 0), &asset.indices)?;
        let mut meshes = Vec::new();
        let mut bounds = Vec::new();
        for mesh in asset.meshes {
            let surfaces = mesh
                .surfaces
                .into_iter()
                .map(|x| RenderMeshSurface {
                    first_index: x.first_index,
                    index_count: x.index_count,
                    material: materials[x.material as usize],
                })
                .collect::<Vec<_>>();
            let mesh = StaticRenderMesh {
                vertex_buffer: BufferPointer::new(
                    vertices,
                    mesh.first_vertex * mem::size_of::<GpuStaticVertex>() as u64,
                ),
                index_buffer: BufferPointer::new(
                    indices,
                    mesh.first_index * mem::size_of::<u16>() as u64,
                ),
                surfaces,
                bounds: Bounds::from_array_and_radius(mesh.bounds.0, mesh.bounds.1),
                // position_scale: mesh.positon_scale,
                // uv_scale: mesh.uv_scale,
            };
            bounds.push(mesh.bounds);
            meshes.push(mesh);
        }
        let mut scene = RenderModel {
            vertices,
            indices,
            meshes,
            bounds,
            names: asset.name_to_mesh,
            parents: asset.nodes.iter().map(|x| x.parent).collect(),
            local_transforms: asset
                .nodes
                .iter()
                .map(|x| {
                    glam::Affine3A::from_scale_rotation_translation(
                        glam::Vec3::from_array(x.scale),
                        glam::Quat::from_array(x.rotation),
                        glam::Vec3::from_array(x.translation),
                    )
                })
                .collect(),
            world_transforms: asset
                .nodes
                .iter()
                .map(|_| glam::Affine3A::IDENTITY)
                .collect(),
            node_to_mesh: asset.node_to_mesh,
            mesh_names: asset.mesh_names,
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
            self.renderer.destroy_buffer(scene.vertices);
            self.renderer.destroy_buffer(scene.vertices);
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

impl<'a> MeshResolver for ResourceCacheMeshResolver<'a> {
    fn resolve_static_mesh(&self, handle: ModelHandle, index: u32) -> Option<&StaticRenderMesh> {
        let scene = self.scens.get(handle)?;
        Some(&scene.meshes[index as usize])
    }

    fn resolve_model(&self, handle: ModelHandle) -> Option<&RenderModel> {
        self.scens.get(handle)
    }
}
