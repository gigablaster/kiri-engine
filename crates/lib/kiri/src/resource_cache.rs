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
    get_compiled_asset_path, load_asset, save_asset, Asset, AssetSource, GltfSceneSource,
    ImageAsset, ImageAssetType, ImageSource, ImportAsset, MeshAssetMaterial, SceneAsset,
};
use kiri_backend::{BufferCreateDesc, DescriptorSetLayoutDesc, ImageCreateDesc};
use kiri_common::{Handle, Pool};
use kiri_gfx::{
    BufferPointer, DescriptorHandle, DescriptorSetBuilder, ImageHandle, ImageUploadData, Renderer,
};
use kiri_vfs::{vfs_load, AssetReference};
use lazy_static::lazy_static;
use log::{debug, warn};
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard};

use crate::{
    gpu::{GpuMeshMaterial, GpuStaticVertex},
    Bounds, ConstUniformBuffer, Error, MeshResolver, RenderMaterial, RenderMeshSurface,
    RenderScene, StaticRenderMesh,
};

pub type SceneHandle = Handle<RenderScene>;
pub type StaticMeshHandle = Handle<(SceneHandle, usize)>;

type ScenePool = Pool<RenderScene>;
type StaticMeshPool = Pool<(SceneHandle, usize)>;

type ImageLoadingTask = Task<Result<(ImageHandle, ImageAsset), Error>>;
type SceneLoadingTask = Task<Result<(SceneHandle, SceneAsset), Error>>;

lazy_static! {
    static ref MATERIAL_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc =
        DescriptorSetLayoutDesc::default()
            .slot(0, "material", vk::DescriptorType::UNIFORM_BUFFER, 1)
            .slot(
                1,
                "base_color",
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1
            )
            .slot(2, "normal", vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1)
            .slot(
                3,
                "metallic_roughness",
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1
            )
            .slot(
                4,
                "occlusion",
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1
            )
            .slot(5, "emissive", vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1);
}

// pub trait ResourceLoader {
//     fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> Result<ImageHandle, Error>;
//     fn get_or_load_material(&self, material: &MeshAssetMaterial)
//         -> Result<DescriptorHandle, Error>;
//     fn get_or_load_scene(&self, name: &str) -> Result<SceneHandle, Error>;
// }

// impl ResourceLoader for Arc<ResourceCache> {
//     fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> Result<ImageHandle, Error> {
//         let source = ImageAssetSource::new(name).ty(ty);
//         self.images
//             .get_or_load(source, |source| load_image_impl(self, source))
//     }

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
    // First, attempt to load compiled asset
    let reference = source.reference();
    if let Ok(reader) = vfs_load(reference) {
        debug!("Loading asset: {:?}", reference);
        Ok(load_asset(reader)?)
    } else {
        // There's no compiled asset, so compile it in runtime
        warn!("Compile asset: {:?}", source);
        let asset = source.import()?;
        if let Err(err) = try_save_asset(reference, &asset) {
            warn!("Failed to save compiled asset to cache: {}", err);
        }
        Ok(asset)
    }
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
    scenes: AssetTracker<SceneHandle>,
    scene_assets: RwLock<ScenePool>,
    meshes: RwLock<HashMap<String, StaticMeshHandle>>,
    mesh_assets: RwLock<StaticMeshPool>,
    material_uniforms: ConstUniformBuffer,
    materials: RwLock<HashMap<MeshAssetMaterial, DescriptorHandle>>,
    dummy_color_image: ImageHandle,
    dummy_emissive_image: ImageHandle,
    dummy_normal_image: ImageHandle,
    dummy_occlusion_metallic_roughness: ImageHandle,
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
            dummy_color_image: renderer.create_image(
                ImageCreateDesc::texture(vk::Format::R8G8B8A8_SRGB, [1, 1]).name("Dummy color"),
                Some(&[ImageUploadData {
                    data: &[127, 127, 127, 255],
                }]),
            )?,
            dummy_emissive_image: renderer.create_image(
                ImageCreateDesc::texture(vk::Format::R8G8B8A8_UNORM, [1, 1]).name("Dummy emissive"),
                Some(&[ImageUploadData {
                    data: &[0, 0, 0, 255],
                }]),
            )?,
            dummy_occlusion_metallic_roughness: renderer.create_image(
                ImageCreateDesc::texture(vk::Format::R8G8B8A8_UNORM, [1, 1]).name("Dummy ORM"),
                Some(&[ImageUploadData {
                    data: &[0, 255, 0, 255],
                }]),
            )?,
            dummy_normal_image: renderer.create_image(
                ImageCreateDesc::texture(vk::Format::R8G8B8A8_UNORM, [1, 1]).name("Dummy emissive"),
                Some(&[ImageUploadData {
                    data: &[0, 0, 255, 255],
                }]),
            )?,
            scenes: Default::default(),
            scene_assets: RwLock::new(ScenePool::new(MAX_RESOURCES)),
            mesh_assets: RwLock::new(StaticMeshPool::new(MAX_RESOURCES)),
            meshes: Default::default(),
        }))
    }

    pub fn tick(&self) -> Result<(), Error> {
        // let mut finished_images = Vec::new();
        // let mut finished_scenes = Vec::new();
        {
            let mut loading = self.scene_loading_tasks.lock();
            let mut i = 0;
            while i < loading.len() {
                if loading[i].is_finished() {
                    let task = loading.remove(i);
                    let (handle, asset) = block_on(task)?;
                    self.process_scene(handle, asset)?;
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
                    let (handle, asset) = block_on(task)?;
                    self.process_image(handle, asset)?;
                } else {
                    i += 1;
                }
            }
        }

        Ok(())
    }

    pub fn resolve(&self) -> ResourceCacheMeshResolver {
        ResourceCacheMeshResolver {
            static_meshes: self.mesh_assets.read(),
            scens: self.scene_assets.read(),
        }
    }

    pub fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> Result<ImageHandle, Error> {
        let source = ImageSource::new(name).ty(ty);
        self.images
            .get_or_load(&source, |source| self.load_image_impl(source))
    }

    fn load_image_impl(&self, source: &ImageSource) -> Result<ImageHandle, Error> {
        let handle = self.renderer.create_image(
            ImageCreateDesc::texture(source.ty.uncompressed_format(), [1, 1])
                .name(&format!("{} - DUMMY", source.reference())),
            Some(&[ImageUploadData {
                data: &[128, 128, 128, 255],
            }]),
        )?;
        self.image_loading_tasks
            .lock()
            .push(IoTaskPool::get().spawn(Self::load_image(handle, source.clone())));
        Ok(handle)
    }

    fn process_image(&self, handle: ImageHandle, asset: ImageAsset) -> Result<(), Error> {
        let upload = asset
            .mips
            .iter()
            .map(|x| ImageUploadData { data: x })
            .collect::<Vec<_>>();
        self.renderer.update_image(
            handle,
            ImageCreateDesc::texture(asset.format, asset.dims).mip_levels(asset.mips.len() as _),
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

                let base_color = if let Some(source) = material
                    .get_map("base_color")
                    .cloned()
                    .unwrap_or_default()
                    .get_image()
                {
                    self.get_or_load_image(&source.path, source.ty)?
                } else {
                    self.dummy_color_image
                };
                let normals = if let Some(source) = material
                    .get_map("normal")
                    .cloned()
                    .unwrap_or_default()
                    .get_image()
                {
                    self.get_or_load_image(&source.path, source.ty)?
                } else {
                    self.dummy_normal_image
                };
                let metallic_roughness = if let Some(source) = material
                    .get_map("metallic_roughness")
                    .cloned()
                    .unwrap_or_default()
                    .get_image()
                {
                    self.get_or_load_image(&source.path, source.ty)?
                } else {
                    self.dummy_occlusion_metallic_roughness
                };
                let occlusion = if let Some(source) = material
                    .get_map("occlusion")
                    .cloned()
                    .unwrap_or_default()
                    .get_image()
                {
                    self.get_or_load_image(&source.path, source.ty)?
                } else {
                    self.dummy_occlusion_metallic_roughness
                };
                let emissive = if let Some(source) = material
                    .get_map("emissive")
                    .cloned()
                    .unwrap_or_default()
                    .get_image()
                {
                    self.get_or_load_image(&source.path, source.ty)?
                } else {
                    self.dummy_emissive_image
                };

                // Allocate and copy uniform data
                let builder = DescriptorSetBuilder::new(
                    vk::ShaderStageFlags::ALL_GRAPHICS,
                    &MATERIAL_DESCRIPTOR_LAYOUT,
                )
                .bind_uniform_buffer(
                    0,
                    self.material_uniforms
                        .push(GpuMeshMaterial::new(material))?,
                )
                .bind_image(1, base_color, vk::ImageAspectFlags::COLOR)
                .bind_image(2, normals, vk::ImageAspectFlags::COLOR)
                .bind_image(3, metallic_roughness, vk::ImageAspectFlags::COLOR)
                .bind_image(4, occlusion, vk::ImageAspectFlags::COLOR)
                .bind_image(5, emissive, vk::ImageAspectFlags::COLOR);
                let descriptor_set = self.renderer.create_descriptor_set(builder)?;
                materials.insert(material.clone(), descriptor_set);
                Ok(descriptor_set)
            }
        }
    }

    pub fn get_or_load_scene(&self, name: &str) -> Result<SceneHandle, Error> {
        let source = GltfSceneSource::new(name);
        self.scenes
            .get_or_load(&source, |source| self.load_scene_impl(source))
    }

    fn load_scene_impl(&self, source: &GltfSceneSource) -> Result<SceneHandle, Error> {
        let handle = self.scene_assets.write().push(RenderScene::default());
        self.scene_loading_tasks
            .lock()
            .push(IoTaskPool::get().spawn(Self::load_scene(handle, source.clone())));
        Ok(handle)
    }

    fn process_scene(&self, handle: SceneHandle, asset: SceneAsset) -> Result<(), Error> {
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
                .transfer_destination(),
        )?;
        let indices = self.renderer.create_buffer(
            BufferCreateDesc::gpu((mem::size_of::<u16>() * asset.indices.len()) as _)
                .index_buffer()
                .transfer_destination(),
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
        let mesh_handles = meshes
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let name = format!("{}#{}", "AAA!", asset.mesh_names[index]);
                self.meshes
                    .read()
                    .get(&name)
                    .copied()
                    .unwrap_or_else(|| self.mesh_assets.write().push((handle, index)))
            })
            .collect::<Vec<_>>();
        let mut scene = RenderScene {
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
            mesh_handles: mesh_handles.clone(),
            mesh_names: asset.mesh_names,
        };
        // debug!("Scene loaded: {:?}", source);
        scene.update_world_transforms();
        let mut named_meshes = self.meshes.write();
        for i in 0..scene.meshes.len() {
            let name = format!("{}#{}", "AAAA", scene.mesh_names[i]);
            named_meshes.insert(name, scene.mesh_handles[i]);
        }
        drop(named_meshes);
        self.scene_assets.write().replace(handle, scene);
        Ok(())
    }

    async fn load_image(
        handle: ImageHandle,
        source: ImageSource,
    ) -> Result<(ImageHandle, ImageAsset), Error> {
        Ok((handle, load_or_compile_asset(&source)?))
    }

    async fn load_scene(
        handle: SceneHandle,
        source: GltfSceneSource,
    ) -> Result<(SceneHandle, SceneAsset), Error> {
        Ok((handle, load_or_compile_asset(&source)?))
    }
}

impl Drop for ResourceCache {
    fn drop(&mut self) {
        self.renderer.destroy_image(self.dummy_color_image);
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
    static_meshes: RwLockReadGuard<'a, StaticMeshPool>,
    scens: RwLockReadGuard<'a, ScenePool>,
}

impl<'a> MeshResolver for ResourceCacheMeshResolver<'a> {
    fn resolve_static_mesh(&self, handle: StaticMeshHandle) -> Option<&StaticRenderMesh> {
        let (mesh, index) = self.static_meshes.get(handle).copied()?;
        let scene = self.scens.get(mesh)?;
        Some(&scene.meshes[index])
    }

    fn resolve_scene(&self, handle: SceneHandle) -> Option<&RenderScene> {
        self.scens.get(handle)
    }
}
