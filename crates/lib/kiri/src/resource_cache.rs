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

use ash::vk;
use bevy_tasks::{block_on, IoTaskPool, Task};
use kiri_assets::{
    load_asset, Asset, AssetSource, GltfSceneSource, ImageAssetSource, ImageAssetType, ImportAsset,
    ImportMode, MeshAssetMaterial,
};
use kiri_backend::{BufferCreateDesc, DescriptorSetLayoutDesc, GpuAllocator, ImageCreateDesc};
use kiri_common::{Handle, Pool};
use kiri_gfx::{DescriptorHandle, DescriptorSetBuilder, ImageHandle, ImageUploadData, Renderer};
use kiri_vfs::{vfs_load, AssetReference};
use lazy_static::lazy_static;
use log::{debug, error, warn};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{
    gpu::{GpuMeshMaterial, GpuStaticVertex},
    Bounds, ConstUniformBuffer, Error, RenderMaterialDesc, RenderMeshSurface, RenderScene,
    StaticRenderMesh,
};

pub type SceneHandle = Handle<RenderScene>;
pub type StaticMeshHandle = Handle<(SceneHandle, usize)>;

type ScenePool = Pool<RenderScene>;
type StaticMeshPool = Pool<(SceneHandle, usize)>;

type LoadingTask = Task<()>;

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

pub trait ResourceLoader {
    fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> Result<ImageHandle, Error>;
    fn get_or_load_material(&self, material: &MeshAssetMaterial)
        -> Result<DescriptorHandle, Error>;
    fn get_or_load_scene(&self, name: &str) -> Result<SceneHandle, Error>;
}

impl ResourceLoader for Arc<ResourceCache> {
    fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> Result<ImageHandle, Error> {
        let source = ImageAssetSource::new(name).ty(ty);
        self.images
            .get_or_load(source, |source| load_image_impl(self, source))
    }

    fn get_or_load_material(
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

                let base_color = if let Some((name, ty)) = material.base_color.get_image() {
                    self.get_or_load_image(name.as_str(), ty)?
                } else {
                    self.dummy_image
                };
                let normals = if let Some((name, ty)) = material.normals.get_image() {
                    self.get_or_load_image(name.as_str(), ty)?
                } else {
                    self.dummy_image
                };
                let metallic_roughness =
                    if let Some((name, ty)) = material.metallic_roughness.get_image() {
                        self.get_or_load_image(name.as_str(), ty)?
                    } else {
                        self.dummy_image
                    };
                let occlusion = if let Some((name, ty)) = material.occlusion.get_image() {
                    self.get_or_load_image(name.as_str(), ty)?
                } else {
                    self.dummy_image
                };
                let emissive = if let Some((name, ty)) = material.emissive.get_image() {
                    self.get_or_load_image(name.as_str(), ty)?
                } else {
                    self.dummy_image
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

    fn get_or_load_scene(&self, name: &str) -> Result<SceneHandle, Error> {
        let source = GltfSceneSource::new(name);
        self.scenes
            .get_or_load(source, |source| load_scene_impl(self, source))
    }
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
    fn get_or_load<U: AssetSource, LOAD: FnOnce(U) -> Result<T, Error>>(
        &self,
        source: U,
        load: LOAD,
    ) -> Result<T, Error> {
        let assets = self.assets.upgradable_read();
        let key = source.reference().normalized();
        if let Some(asset) = assets.get(&key) {
            Ok(*asset)
        } else {
            let mut assets = RwLockUpgradableReadGuard::upgrade(assets);
            if let Some(asset) = assets.get(&key) {
                Ok(*asset)
            } else {
                let asset = load(source)?;
                assets.insert(key.clone(), asset);
                Ok(asset)
            }
        }
    }
}

fn load_scene_impl(
    manager: &Arc<ResourceCache>,
    source: GltfSceneSource,
) -> Result<SceneHandle, Error> {
    let handle = manager.scene_assets.write().push(RenderScene::default());
    manager
        .loading_tasks
        .lock()
        .push(IoTaskPool::get().spawn(load_scene(manager.clone(), handle, source)));
    Ok(handle)
}

async fn load_scene(manager: Arc<ResourceCache>, handle: SceneHandle, source: GltfSceneSource) {
    if let Err(err) = do_load_scene(&manager, handle, &source) {
        error!("Failed to load scene {:?}: {}", source, err);
    }
}

fn do_load_scene(
    manager: &Arc<ResourceCache>,
    handle: SceneHandle,
    source: &GltfSceneSource,
) -> Result<(), Error> {
    let asset = load_or_compile_asset(source)?;
    let mut named_meshes = manager.meshes.write();
    let mut mesh_assets = manager.meshe_assets.write();
    let mut materials = Vec::new();
    for material in &asset.materials {
        materials.push(RenderMaterialDesc {
            ds: manager.get_or_load_material(material)?,
            ty: material.blend.into(),
        });
    }
    let reference = source.reference();
    let vertices: Vec<GpuStaticVertex> = asset.vertices.into_iter().map(|x| x.into()).collect();
    let vertices = manager.renderer.create_buffer(
        BufferCreateDesc::gpu((mem::size_of::<GpuStaticVertex>() * vertices.len()) as _)
            .veretex_buffer()
            .transfer_destination()
            .name(&format!("{} - VB", reference))
            .allocator(&manager.allocatpr),
    )?;
    let indices = manager.renderer.create_buffer(
        BufferCreateDesc::gpu((mem::size_of::<u16>() * asset.indices.len()) as _)
            .index_buffer()
            .transfer_destination()
            .name(&format!("{} - IB", reference))
            .allocator(&manager.allocatpr),
    )?;
    let mut meshes = Vec::new();
    let mut bounds = Vec::new();
    for mesh in asset.meshes {
        let surfaces = mesh
            .surfaces
            .into_iter()
            .map(|x| RenderMeshSurface {
                first_index: x.first_index,
                index_count: x.index_count,
                material_index: x.material,
            })
            .collect::<Vec<_>>();
        let mesh = StaticRenderMesh {
            vertex_offset: mesh.vertex_offset, //FIXME: is it in bytes? I assume not
            surfaces,
            bounds: Bounds::from_array_and_radius(mesh.bounds.0, mesh.bounds.1),
            position_scale: mesh.positon_scale,
            uv_scale: mesh.uv_scale,
        };
        bounds.push(mesh.bounds);
        meshes.push(mesh);
    }
    let scene_name = source.reference().as_str().to_owned();
    let mesh_handles = meshes
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let name = format!("{}#{}", scene_name, asset.mesh_names[index]);
            named_meshes
                .get(&name)
                .copied()
                .unwrap_or_else(|| mesh_assets.push((handle, index)))
        })
        .collect::<Vec<_>>();
    let mut scene = RenderScene {
        vertices,
        indices,
        meshes,
        materials,
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
    debug!("Scene loaded: {:?}", source);
    scene.update_world_transforms();
    for i in 0..scene.meshes.len() {
        let name = format!("{}#{}", scene_name, scene.mesh_names[i]);
        named_meshes.insert(name, scene.mesh_handles[i]);
    }
    Ok(())
}

pub(super) fn load_or_compile_asset<T: AssetSource + ImportAsset<U> + Debug, U: Asset>(
    source: &T,
) -> Result<U, Error> {
    // First, attempt to load compiled asset
    let reference = source.reference();
    if let Ok(reader) = vfs_load(&reference.compiled()) {
        debug!("Loading asset: {:?}", reference);
        Ok(load_asset(reader)?)
    } else {
        // There's no compiled asset, so compile it in runtime
        warn!("Compile asset: {:?}", source);
        Ok(source.import(ImportMode::Runtime)?)
    }
}

fn load_image_impl(
    manager: &Arc<ResourceCache>,
    source: ImageAssetSource,
) -> Result<ImageHandle, Error> {
    let handle = manager.renderer.create_image(
        &manager.allocatpr,
        ImageCreateDesc::texture(source.ty.uncompressed_format(), [1, 1]),
        Some(&[ImageUploadData {
            data: &[128, 128, 128, 255],
        }]),
    )?;
    manager
        .loading_tasks
        .lock()
        .push(IoTaskPool::get().spawn(load_image(manager.clone(), handle, source)));
    Ok(handle)
}

async fn load_image(manager: Arc<ResourceCache>, handle: ImageHandle, source: ImageAssetSource) {
    if let Err(err) = do_load_image(&manager, handle, &source) {
        error!("Failed to load image {:?}: {}", source, err);
    }
}

fn do_load_image(
    manager: &Arc<ResourceCache>,
    handle: ImageHandle,
    source: &ImageAssetSource,
) -> Result<(), Error> {
    let asset = load_or_compile_asset(source)?;
    let upload = asset
        .mips
        .iter()
        .map(|x| ImageUploadData { data: x })
        .collect::<Vec<_>>();
    manager.renderer.update_image(
        handle,
        &manager.allocatpr,
        ImageCreateDesc::texture(asset.format, asset.dims)
            .mip_levels(asset.mips.len() as _)
            .name(&format!("{}", source.reference())),
        Some(&upload),
    )?;
    debug!(
        "Image loaded: {:?} ({:?} {:?})",
        source, asset.format, asset.dims
    );
    Ok(())
}

const MATERIAL_BUFFER_SIZE: u64 = 2 * 1024 * 1024;
const MAX_RESOURCES: usize = 0xffff;

#[derive(Debug)]
pub struct ResourceCache {
    renderer: Arc<Renderer>,
    loading_tasks: Mutex<Vec<LoadingTask>>,
    images: AssetTracker<ImageHandle>,
    scenes: AssetTracker<SceneHandle>,
    scene_assets: RwLock<ScenePool>,
    meshes: RwLock<HashMap<String, StaticMeshHandle>>,
    meshe_assets: RwLock<StaticMeshPool>,
    material_uniforms: ConstUniformBuffer,
    materials: RwLock<HashMap<MeshAssetMaterial, DescriptorHandle>>,
    dummy_image: ImageHandle,
    allocatpr: GpuAllocator,
}

impl ResourceCache {
    pub fn new(renderer: &Arc<Renderer>) -> Result<Arc<Self>, Error> {
        debug!("Create resource manager");
        let allocator = GpuAllocator::new(&renderer.device);
        Ok(Arc::new(Self {
            renderer: renderer.clone(),
            loading_tasks: Default::default(),
            images: Default::default(),
            materials: Default::default(),
            material_uniforms: ConstUniformBuffer::new(renderer, &allocator, MATERIAL_BUFFER_SIZE)?,
            dummy_image: renderer.create_image(
                &allocator,
                ImageCreateDesc::texture(vk::Format::R8G8B8A8_UNORM, [1, 1]).name("Dummy image"),
                Some(&[ImageUploadData {
                    data: &[127, 127, 127, 255],
                }]),
            )?,
            scenes: Default::default(),
            scene_assets: RwLock::new(ScenePool::new(MAX_RESOURCES)),
            meshe_assets: RwLock::new(StaticMeshPool::new(MAX_RESOURCES)),
            meshes: Default::default(),
            allocatpr: allocator,
        }))
    }

    pub fn tick(&self) {
        let mut loading = self.loading_tasks.lock();
        let mut i = 0;
        while i < loading.len() {
            if loading[i].is_finished() {
                let task = loading.remove(i);
                block_on(task);
            } else {
                i += 1;
            }
        }
    }
}

impl Drop for ResourceCache {
    fn drop(&mut self) {
        self.renderer.destroy_image(self.dummy_image);
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
