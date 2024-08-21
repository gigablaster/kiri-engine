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

use std::{collections::HashMap, mem, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use bevy_tasks::{block_on, IoTaskPool, Task};
use kiri_assets::{
    Asset, AssetSource, GltfMeshSource, ImageAsset, ImageAssetSource, ImageAssetType,
    StaticMeshAsset,
};
use kiri_backend::{Image, ImageCreateDesc, RenderDevice};
use kiri_common::{DynamicAllocator, Handle, Pool};
use kiri_gfx::{
    BindlessManager, BufferHandle, BufferManager, BufferSlice, ImageHandle, ImageUploadData,
    Staging,
};
use kiri_vfs::{vfs_load, AssetReference};
use log::{debug, error};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{Bounds, Error, RenderMeshMaterial, RenderMeshSurface, StaticRenderMesh};

pub type StaticMeshHandle = Handle<Arc<StaticRenderMesh>>;
// pub type SceneHandle = Handle<RenderScene>;

type StaticMeshPool = Pool<Arc<StaticRenderMesh>>;
// type ScenePool = Pool<RenderScene>;

type LoadingTask = Task<()>;
// type StaticMeshLoadingTask = Task<Result<Arc<StaticMeshAsset>, Error>>;

const GEOMETRY_PAGE_SIZE: usize = 64 * 1024 * 1024;

pub trait ResourceLoader {
    fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> ImageHandle;
    fn get_or_load_static_mesh(&self, name: &str) -> StaticMeshHandle;
}

impl ResourceLoader for Arc<ResourceManager> {
    fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> ImageHandle {
        let reference = ImageAssetSource::from_file(name).reference();
        load_image_impl(self, reference, ty)
    }

    fn get_or_load_static_mesh(&self, name: &str) -> StaticMeshHandle {
        let parts = name.split("#").collect::<ArrayVec<_, 2>>();
        let source = GltfMeshSource {
            gltf: parts[0].to_owned(),
            mesh: parts[1].to_owned(),
        };
        load_static_mesh_impl(self, source.reference())
    }
}

fn load_image_impl(
    manager: &Arc<ResourceManager>,
    reference: AssetReference,
    ty: ImageAssetType,
) -> ImageHandle {
    let images = manager.image_assets.upgradable_read();
    if let Some(handle) = images.get(&reference) {
        let image = manager
            .bindless
            .read()
            .resolve_image(*handle)
            .unwrap()
            .clone();
        unsafe { Arc::increment_strong_count(Arc::into_raw(image)) };
        *handle
    } else {
        let mut images = RwLockUpgradableReadGuard::upgrade(images);
        if let Some(handle) = images.get(&reference) {
            let image = manager
                .bindless
                .read()
                .resolve_image(*handle)
                .unwrap()
                .clone();
            unsafe { Arc::increment_strong_count(Arc::into_raw(image)) };
            *handle
        } else {
            let handle = manager.bindless.write().import_image(
                manager.dummy_images.get(&ty).cloned().unwrap(),
                vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            images.insert(reference, handle);
            manager
                .loading
                .lock()
                .push(IoTaskPool::get().spawn(load_image(manager.clone(), handle, reference)));
            manager.image_tracking.lock().insert(handle, reference);
            handle
        }
    }
}

async fn load_image(manager: Arc<ResourceManager>, handle: ImageHandle, reference: AssetReference) {
    if let Err(err) = do_load_image(&manager, handle, reference) {
        error!("Failed to load image {}: {}", reference, err);
    }
}

fn do_load_image(
    manager: &ResourceManager,
    handle: ImageHandle,
    reference: AssetReference,
) -> Result<(), Error> {
    let asset = ImageAsset::load(vfs_load(reference)?)?;
    let image = Image::new(
        &manager.device,
        ImageCreateDesc::texture(asset.format, asset.dims),
    )?;
    let data = asset
        .mips
        .iter()
        .map(|data| ImageUploadData { data })
        .collect::<Vec<_>>();
    manager.staging.lock().upload_image(&image, &data)?;
    manager.bindless.write().update_image(
        handle,
        image.into(),
        vk::ImageAspectFlags::COLOR,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
    Ok(())
}

fn load_static_mesh_impl(
    manager: &Arc<ResourceManager>,
    reference: AssetReference,
) -> StaticMeshHandle {
    let meshes = manager.static_mesh_assets.upgradable_read();
    if let Some(handle) = meshes.get(&reference) {
        let mesh = manager.static_meshes.read().get(*handle).cloned().unwrap();
        unsafe { Arc::increment_strong_count(Arc::into_raw(mesh)) }
        *handle
    } else {
        let mut meshes = RwLockUpgradableReadGuard::upgrade(meshes);
        if let Some(handle) = meshes.get(&reference) {
            let mesh = manager.static_meshes.read().get(*handle).cloned().unwrap();
            unsafe { Arc::increment_strong_count(Arc::into_raw(mesh)) }
            *handle
        } else {
            let handle = manager.static_meshes.write().push(Default::default());
            meshes.insert(reference, handle);
            manager
                .loading
                .lock()
                .push(IoTaskPool::get().spawn(load_static_mesh(
                    manager.clone(),
                    handle,
                    reference,
                )));
            handle
        }
    }
}

async fn load_static_mesh(
    manager: Arc<ResourceManager>,
    handle: StaticMeshHandle,
    reference: AssetReference,
) {
    if let Err(err) = do_load_static_mesh(&manager, handle, reference) {
        error!("Failed to load mesh {}: {}", reference, err)
    }
}

fn do_load_static_mesh(
    manager: &Arc<ResourceManager>,
    handle: StaticMeshHandle,
    reference: AssetReference,
) -> Result<(), Error> {
    let asset = StaticMeshAsset::load(vfs_load(reference)?)?;
    let vertices = manager.allocate_geometry(mem::size_of_val(&asset.vertices))?;
    let indices = manager.allocate_geometry(mem::size_of_val(&asset.indices))?;
    manager.upload_buffer(vertices, &asset.vertices)?;
    manager.upload_buffer(indices, &asset.indices)?;
    let mesh = StaticRenderMesh {
        vertices,
        indices,
        surfaces: asset
            .surfaces
            .into_iter()
            .map(|x| RenderMeshSurface {
                first_index: x.first_index,
                index_count: x.index_count,
                material_index: x.material,
            })
            .collect(),
        materials: asset
            .materials
            .into_iter()
            .map(|material| RenderMeshMaterial {
                base_color: load_image_impl(manager, material.base_color, ImageAssetType::Color),
                normals: load_image_impl(manager, material.normals, ImageAssetType::Normal),
                metallic_roughness: load_image_impl(
                    manager,
                    material.metallic_roughness,
                    ImageAssetType::MetallicRoughness,
                ),
                occlusion: load_image_impl(manager, material.occlusion, ImageAssetType::Occlusion),
                emissive: load_image_impl(manager, material.emissive, ImageAssetType::Emissive),
                emissive_power: material.emissive_power,
                blend: material.blend,
            })
            .collect(),
        bounds: Bounds::from_array_and_radius(asset.bounds.0, asset.bounds.1),
    };
    manager.static_meshes.write().replace(handle, mesh.into());
    Ok(())
}

#[derive(Debug)]
pub struct ResourceManager {
    device: Arc<RenderDevice>,
    bindless: RwLock<BindlessManager>,
    buffers: RwLock<BufferManager>,
    staging: Mutex<Staging>,
    loading: Mutex<Vec<LoadingTask>>,
    image_assets: RwLock<HashMap<AssetReference, ImageHandle>>,
    image_tracking: Mutex<HashMap<ImageHandle, AssetReference>>,
    static_meshes: RwLock<StaticMeshPool>,
    static_mesh_assets: RwLock<HashMap<AssetReference, StaticMeshHandle>>,
    geometry: Mutex<HashMap<BufferHandle, DynamicAllocator>>,
    // loading_static_meshes: Mutex<Vec<StaticMeshLoadingTask>>,
    dummy_images: HashMap<ImageAssetType, Arc<Image>>,
}

// const MESH_POOL_SIZE: usize = 128 * 1024 * 1024;
// const MATERIAL_BUFFER_SIZE: usize = 512 * 1024;
// const MAX_MATERIALS_COUNT: usize = MATERIAL_BUFFER_SIZE / mem::size_of::<GpuMeshMaterial>();

// struct MaterialPoolLimits {}

// impl PoolLimits for MaterialPoolLimits {
//     const DEFAULT_SPACE: usize = MAX_MATERIALS_COUNT;

//     fn max_index() -> u32 {
//         MAX_MATERIALS_COUNT as u32
//     }
// }

// pub type MaterialHandle = Handle<GpuMeshMaterial, MaterialPoolLimits>;
// pub type MaterialPool =
//     Pool<GpuMeshMaterial, MaybeUninitVauleWrapper<GpuMeshMaterial>, MaterialPoolLimits>;

impl ResourceManager {
    pub fn new(device: &Arc<RenderDevice>) -> Result<Arc<Self>, Error> {
        debug!("Create resource manager");
        let mut staging = Staging::new(device)?;
        let mut dummy_images = HashMap::new();
        for ty in [
            ImageAssetType::Color,
            ImageAssetType::Emissive,
            ImageAssetType::MetallicRoughness,
            ImageAssetType::NonColor,
            ImageAssetType::Normal,
            ImageAssetType::Occlusion,
        ] {
            let image = Image::new(
                device,
                ImageCreateDesc::texture(ty.uncompressed_format(), [1, 1]),
            )?;
            staging.upload_image(
                &image,
                &[ImageUploadData {
                    data: &ty.default_values(),
                }],
            )?;
            dummy_images.insert(ty, image.into());
        }
        Ok(Arc::new(Self {
            device: device.clone(),
            bindless: RwLock::new(BindlessManager::new(device)?),
            buffers: RwLock::new(BufferManager::new(device)),
            staging: Mutex::new(staging),
            image_assets: Default::default(),
            image_tracking: Default::default(),
            loading: Default::default(),
            static_meshes: Default::default(),
            static_mesh_assets: Default::default(),
            geometry: Default::default(),
            dummy_images,
        }))
    }

    pub fn tick(&self) {
        let mut loading = self.loading.lock();
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

    pub fn allocate_geometry(&self, size: usize) -> Result<BufferSlice, Error> {
        assert!(size <= GEOMETRY_PAGE_SIZE);
        let mut geometry = self.geometry.lock();
        if let Some(slice) = geometry.iter_mut().find_map(|(handle, allocator)| {
            if let Some(offset) = allocator.allocate(size) {
                Some(BufferSlice::new(*handle, offset))
            } else {
                None
            }
        }) {
            Ok(slice)
        } else {
            let buffer = self.buffers.write().create(size)?;
            let mut allocator = DynamicAllocator::new(
                GEOMETRY_PAGE_SIZE,
                self.device
                    .physical_device
                    .properties
                    .limits
                    .min_storage_buffer_offset_alignment as _,
            );
            let offset = allocator.allocate(size).unwrap();
            geometry.insert(buffer, allocator);
            Ok(BufferSlice::new(buffer, offset))
        }
    }

    pub fn upload_buffer<T: Copy>(&self, buffer: BufferSlice, data: &[T]) -> Result<(), Error> {
        let gpu_buffer = self.buffers.read().resolve(buffer.buffer)?;
        self.staging
            .lock()
            .upload_buffer(&gpu_buffer, buffer.offset as _, data)?;
        Ok(())
    }

    pub fn unload_image(&self, handle: ImageHandle) {
        let mut bindless = self.bindless.write();
        if let Ok(image) = bindless.resolve_image(handle) {
            // Check if last instance.
            // It might create problems if we have Arcs somewhere outside of ResourceManager, so
            // in perfect world we sould need a separated asset tracking. But we won't let images
            // to leak outside.
            if Arc::strong_count(image) == 1 {
                bindless.remove_image(handle);
                let mut tracking = self.image_tracking.lock();
                let mut images = self.image_assets.write();
                if let Some(reference) = tracking.remove(&handle) {
                    images.remove(&reference);
                }
            } else {
                unsafe { Arc::decrement_strong_count(Arc::into_raw(image.clone())) };
            }
        }
    }
}
