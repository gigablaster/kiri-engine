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
    hash::Hash,
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use bevy_tasks::{block_on, IoTaskPool, Task};
use kiri_assets::{
    load_asset, AssetSource, ImageAsset, ImageAssetSource, ImageAssetType, ImportAsset, ImportMode,
};
use kiri_backend::ImageCreateDesc;
use kiri_common::{Handle, Pool};
use kiri_gfx::{ImageHandle, ImageUploadData, Renderer};
use kiri_vfs::{vfs_load, AssetReference};
use log::{debug, error, warn};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{Error, StaticRenderMesh};

pub type StaticMeshHandle = Handle<Arc<StaticRenderMesh>>;
// pub type SceneHandle = Handle<RenderScene>;

type StaticMeshPool = Pool<Arc<StaticRenderMesh>>;
// type ScenePool = Pool<RenderScene>;

type LoadingTask = Task<()>;
// type StaticMeshLoadingTask = Task<Result<Arc<StaticMeshAsset>, Error>>;

pub trait ResourceLoader {
    fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> Result<ImageHandle, Error>;
}

impl ResourceLoader for Arc<ResourceManager> {
    fn get_or_load_image(&self, name: &str, ty: ImageAssetType) -> Result<ImageHandle, Error> {
        let source = ImageAssetSource::new(name).ty(ty);
        self.images
            .get_or_load(source, |source| load_image_impl(self, source))
    }
}

/// Keeps normalized asset name -> asset + ref count.
///
/// T must be a handle
#[derive(Debug, Default)]
struct AssetTracker<T: Copy + Hash + Eq> {
    assets: RwLock<HashMap<AssetReference, (T, AtomicUsize)>>,
    references: Mutex<HashMap<T, AssetReference>>,
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
        if let Some((asset, count)) = assets.get(&key) {
            count.fetch_add(1, Ordering::AcqRel);
            Ok(*asset)
        } else {
            let mut assets = RwLockUpgradableReadGuard::upgrade(assets);
            if let Some((asset, count)) = assets.get(&key) {
                count.fetch_add(1, Ordering::AcqRel);
                Ok(*asset)
            } else {
                let asset = load(source)?;
                assets.insert(key.clone(), (asset, AtomicUsize::new(1)));
                self.references.lock().insert(asset, key);
                Ok(asset)
            }
        }
    }

    /// Decreases ref count and release reasources if tehre's no references left
    fn release<UNLOAD: FnOnce(T)>(&self, handle: T, unload: UNLOAD) {
        let mut references = self.references.lock();
        if let Some(reference) = references.get(&handle) {
            let assets = self.assets.upgradable_read();
            if let Some((_, count)) = assets.get(reference) {
                if count.fetch_sub(1, Ordering::AcqRel) == 0 {
                    let mut assets = RwLockUpgradableReadGuard::upgrade(assets);
                    assets.remove(reference);
                    references.remove(&handle);
                    unload(handle);
                }
            }
        }
    }
}

fn load_image_impl(
    manager: &Arc<ResourceManager>,
    source: ImageAssetSource,
) -> Result<ImageHandle, Error> {
    let handle = manager.renderer.create_image(
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

async fn load_image(manager: Arc<ResourceManager>, handle: ImageHandle, source: ImageAssetSource) {
    if let Err(err) = do_load_image(&manager, handle, &source) {
        error!("Failed to load image {:?}: {}", source, err);
    }
}

fn do_load_image(
    manager: &Arc<ResourceManager>,
    handle: ImageHandle,
    source: &ImageAssetSource,
) -> Result<(), Error> {
    let asset = load_image_asset(source)?;
    let upload = asset
        .mips
        .iter()
        .map(|x| ImageUploadData { data: x })
        .collect::<Vec<_>>();
    manager.renderer.update_image(
        handle,
        ImageCreateDesc::texture(asset.format, asset.dims).mip_levels(asset.mips.len()),
        Some(&upload),
    )?;
    debug!(
        "Image loaded: {:?} ({:?} {:?})",
        source, asset.format, asset.dims
    );
    Ok(())
}

fn load_image_asset(source: &ImageAssetSource) -> Result<ImageAsset, Error> {
    // First, attempt to load compiled asset
    let reference = source.reference();
    if let Ok(reader) = vfs_load(&reference.compiled()) {
        debug!("Loading image {:?}", reference);
        Ok(load_asset(reader)?)
    } else {
        // There's no imported asset, so import it in runtime (slow)
        warn!("Compile asset {:?} at runtime", source);
        Ok(source.import(ImportMode::Runtime)?)
    }
}

#[derive(Debug)]
pub struct ResourceManager {
    renderer: Arc<Renderer>,
    loading_tasks: Mutex<Vec<LoadingTask>>,
    images: AssetTracker<ImageHandle>,
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
    pub fn new(renderer: &Arc<Renderer>) -> Result<Arc<Self>, Error> {
        debug!("Create resource manager");
        Ok(Arc::new(Self {
            renderer: renderer.clone(),
            loading_tasks: Default::default(),
            images: Default::default(),
        }))
    }

    pub fn unload_image(&self, handle: ImageHandle) {
        self.images
            .release(handle, |handle| self.renderer.destroy_image(handle));
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
