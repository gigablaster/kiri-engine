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

use std::{collections::HashMap, fmt::Debug, hash::Hash, sync::Arc};

#[cfg(feature = "devel")]
use kiri_assets::{
    get_compiled_asset_change_time, get_compiled_asset_path, save_asset, ImportAsset,
};
use kiri_assets::{
    Asset, AssetSource, ImageAsset, ImageSource, MeshAssetMaterial, ModelAsset, ModelSource,
    STATIC_MESH_INPUT_LAYOUT,
};
use kiri_common::{block_on, spawn, spawn_io, yield_now, Task};
use kiri_common::{Handle, Pool};
use kiri_gfx::{
    effects::{
        BasicEffectFactory, EffectInstanceDesc, MeshEffectFactory, DEPTH_RENDER_PASS_LAYOUT,
        EFFECT_PASS_DEPTH, EFFECT_PASS_DEPTH_MASKED, EFFECT_PASS_OPAQUE, EFFECT_PASS_OPAQUE_MASKED,
        EFFECT_PASS_TRANSPARENT, MAIN_RENDER_PASS_LAYOUT,
    },
    ImageUploadData, PipelineCache, PipelineHandle, RenderMeshBuilder, RenderMeshMaterial,
    RenderModel, RenderModelBuilder, Renderer, Texture, TextureBuilder,
};
use kiri_math::{Affine3A, BoundingBox, Quat, Vec3, Vec4};
#[cfg(feature = "devel")]
use log::warn;
use log::{debug, error};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};
#[cfg(feature = "devel")]
use std::{fs::File, io};

use crate::Error;

pub type ModelHandle = Handle<Resource<RenderModel>>;
pub type TextureHandle = Handle<Resource<Texture>>;
pub type MaterialHandle = Handle<Resource<RenderMeshMaterial>>;

const MAX_RESOURCES: usize = 0xffff;

#[derive(Debug)]
pub enum Resource<T: Debug + Send + Sync> {
    Loading,
    Failed,
    Loaded(Arc<T>),
}

impl<T: Debug + Send + Sync> Clone for Resource<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Loading => Self::Loading,
            Self::Failed => Self::Failed,
            Self::Loaded(arg0) => Self::Loaded(arg0.clone()),
        }
    }
}

pub type LoadingTask<T> = (Handle<Resource<T>>, Task<Result<T, Error>>);

#[derive(Debug)]
pub struct ResourceType<K: Hash + Eq, T: Debug + Send + Sync> {
    pool: Mutex<Pool<Resource<T>>>,
    names: RwLock<HashMap<K, Handle<Resource<T>>>>,
    loading: Mutex<Vec<LoadingTask<T>>>,
}

impl<K: Hash + Eq, T: Debug + Send + Sync> Default for ResourceType<K, T> {
    fn default() -> Self {
        Self {
            pool: Mutex::new(Pool::new(MAX_RESOURCES)),
            names: Default::default(),
            loading: Default::default(),
        }
    }
}

impl<K: Hash + Eq, T: Debug + Send + Sync> ResourceType<K, T> {
    fn get_or_load<LOAD: FnOnce() -> Task<Result<T, Error>>>(
        &self,
        key: K,
        load: LOAD,
    ) -> Handle<Resource<T>> {
        let names = self.names.upgradable_read();
        if let Some(handle) = names.get(&key) {
            *handle
        } else {
            let mut names = RwLockUpgradableReadGuard::upgrade(names);
            if let Some(handle) = names.get(&key) {
                *handle
            } else {
                let task = load();
                let handle = self.pool.lock().push(Resource::Loading);
                names.insert(key, handle);
                self.loading.lock().push((handle, task));
                handle
            }
        }
    }

    fn resolve(&self, handle: Handle<Resource<T>>) -> Option<Resource<T>> {
        self.pool.lock().get(handle).cloned()
    }

    async fn wait(&self, handle: Handle<Resource<T>>) -> Result<Arc<T>, Error> {
        loop {
            if let Some(resource) = self.resolve(handle) {
                match resource {
                    Resource::Loading => yield_now().await,
                    Resource::Failed => return Err(Error::ResourceLoadingFailed),
                    Resource::Loaded(resource) => return Ok(resource),
                }
            }
        }
    }

    fn purge(&self) {
        let mut pool = self.pool.lock();
        let mut to_delete = Vec::default();
        for (handle, resource) in pool.enumerate() {
            if let Resource::Loaded(resource) = resource {
                if Arc::strong_count(resource) == 1 {
                    to_delete.push(handle);
                }
            }
        }
        for handle in to_delete {
            pool.remove(handle);
        }
        self.names
            .write()
            .retain(|_, handle| pool.is_handle_valid(*handle));
    }

    fn tick(&self) {
        let mut pool = self.pool.lock();
        let mut loading = self.loading.lock();
        let mut i = 0;
        while i < loading.len() {
            if loading[i].1.is_finished() {
                let (handle, task) = loading.remove(i);
                match block_on(task) {
                    Ok(resource) => {
                        pool.replace(handle, Resource::Loaded(Arc::new(resource)));
                    }
                    Err(err) => {
                        error!("Failed to load asset: {}", err);
                        pool.replace(handle, Resource::Failed);
                    }
                }
            } else {
                i += 1;
            }
        }
    }
}

pub trait ResourceLoader {
    fn get_or_load_texture(&self, source: &ImageSource) -> Handle<Resource<Texture>>;
    fn get_or_load_model(&self, name: &str) -> Handle<Resource<RenderModel>>;
}

#[derive(Debug)]
pub struct ResourceManager {
    pub renderer: Arc<Renderer>,
    pub pipeline_cache: Arc<PipelineCache>,
    textures: ResourceType<ImageSource, Texture>,
    materials: ResourceType<MeshAssetMaterial, RenderMeshMaterial>,
    models: ResourceType<String, RenderModel>,
    effect_factory: RwLock<Vec<Box<dyn MeshEffectFactory>>>,
}

impl ResourceLoader for Arc<ResourceManager> {
    fn get_or_load_texture(&self, source: &ImageSource) -> Handle<Resource<Texture>> {
        let source = source.clone();
        self.textures.get_or_load(source.clone(), || {
            spawn(ResourceManager::load_texture(self.clone(), source))
        })
    }

    fn get_or_load_model(&self, name: &str) -> Handle<Resource<RenderModel>> {
        self.models.get_or_load(name.to_owned(), || {
            spawn(ResourceManager::load_model(self.clone(), name.to_owned()))
        })
    }
}

pub struct ResourceResolveContext<'a> {
    models: &'a Pool<Resource<RenderModel>>,
}

impl<'a> ResourceResolveContext<'a> {
    pub fn resolve_model(&self, handle: ModelHandle) -> Result<Option<Arc<RenderModel>>, Error> {
        let resource = self
            .models
            .get(handle)
            .ok_or(Error::InvalidModelHandle(handle))?;
        match resource {
            Resource::Loading => Ok(None),
            Resource::Failed => Err(Error::ResourceLoadingFailed),
            Resource::Loaded(model) => Ok(Some(model.clone())),
        }
    }
}

impl ResourceManager {
    pub fn new(renderer: &Arc<Renderer>) -> Result<Arc<Self>, Error> {
        debug!("Create resource manager");
        let pipeline_cache = PipelineCache::new(renderer);
        Ok(Arc::new(Self {
            effect_factory: RwLock::new(vec![Box::new(BasicEffectFactory::new(&pipeline_cache))]),
            renderer: renderer.clone(),
            pipeline_cache,
            textures: Default::default(),
            materials: Default::default(),
            models: Default::default(),
        }))
    }

    pub fn tick(&self) {
        self.textures.tick();
        self.materials.tick();
        self.models.tick();
    }

    pub fn purge(&self) {
        self.models.purge();
        self.materials.purge();
        self.textures.purge();
    }

    pub fn get_model(&self, handle: ModelHandle) -> Option<Resource<RenderModel>> {
        self.models.resolve(handle)
    }

    pub fn get_texture(&self, handle: TextureHandle) -> Option<Resource<Texture>> {
        self.textures.resolve(handle)
    }

    pub fn resolve_model(&self, handle: ModelHandle) -> Result<Arc<RenderModel>, Error> {
        match self
            .get_model(handle)
            .ok_or(Error::InvalidModelHandle(handle))?
        {
            Resource::Loading => block_on(self.models.wait(handle)),
            Resource::Failed => Err(Error::ResourceLoadingFailed),
            Resource::Loaded(model) => Ok(model),
        }
    }

    pub fn register_mesh_effect(&self, effect: Box<dyn MeshEffectFactory>) {
        self.effect_factory.write().push(effect);
    }

    #[cfg(feature = "devel")]
    async fn load_or_compile_asset<T: AssetSource + ImportAsset<U> + std::fmt::Debug, U: Asset>(
        source: T,
    ) -> Result<U, Error> {
        use kiri_assets::load_or_compile_asset;

        Ok(load_or_compile_asset(&source)?)
    }

    #[cfg(not(feature = "devel"))]
    async fn load_or_compile_asset<T: AssetSource + Debug, U: Asset>(
        source: T,
    ) -> Result<U, Error> {
        use kiri_assets::load_asset;
        use kiri_vfs::vfs_load;

        let reference = source.reference();
        let reader = vfs_load(reference)?;
        Ok(load_asset(reader)?)
    }

    async fn load_texture(
        manager: Arc<ResourceManager>,
        source: ImageSource,
    ) -> Result<Texture, Error> {
        match &source.data {
            kiri_assets::ImageData::Path(path) => {
                let asset: ImageAsset =
                    spawn_io(Self::load_or_compile_asset(source.clone())).await?;
                let mips = asset
                    .mips
                    .iter()
                    .map(|x| ImageUploadData::new(x))
                    .collect::<Vec<_>>();
                debug!("Create texture {:?}", source);
                Ok(TextureBuilder::new(asset.format, asset.dims)
                    .name(path)
                    .data(&mips)
                    .build(&manager.renderer)?)
            }
            kiri_assets::ImageData::Color(color) => {
                Ok(TextureBuilder::new(source.uncompressed_format(), [1, 1])
                    .data(&[ImageUploadData::new(color)])
                    .build(&manager.renderer)?)
            }
        }
    }

    pub fn resolve<CB: FnOnce(ResourceResolveContext)>(&self, cb: CB) {
        let models = self.models.pool.lock();
        cb(ResourceResolveContext { models: &models });
    }

    async fn load_material(
        manager: Arc<ResourceManager>,
        source: MeshAssetMaterial,
    ) -> Result<RenderMeshMaterial, Error> {
        let loading_textures = source
            .images
            .into_iter()
            .map(|(slot, source)| (slot, manager.get_or_load_texture(&source)))
            .collect::<HashMap<_, _>>();
        let mut textures = HashMap::new();
        for (name, handle) in loading_textures {
            textures.insert(name, manager.textures.wait(handle).await?);
        }
        let desc = EffectInstanceDesc {
            textures,
            scalars: source.scalars,
            vectors: source
                .vectors
                .into_iter()
                .map(|(name, value)| (name, Vec4::from_array(value)))
                .collect(),
        };
        let mut material = None;
        for factory in manager.effect_factory.read().iter() {
            if let Some(effect) = factory.get_or_create(
                &source.name,
                &DEPTH_RENDER_PASS_LAYOUT,
                &MAIN_RENDER_PASS_LAYOUT,
                &STATIC_MESH_INPUT_LAYOUT,
            )? {
                let instance = effect.create_instance(&desc)?;
                let main = match source.blend {
                    kiri_assets::MeshMaterialBlend::Opaque => instance
                        .pipeline(EFFECT_PASS_OPAQUE)
                        .ok_or(Error::EffectPipelineNotFound(EFFECT_PASS_OPAQUE.to_owned()))?,
                    kiri_assets::MeshMaterialBlend::AlphaBlend => PipelineHandle::invalid(),
                    kiri_assets::MeshMaterialBlend::AlphaTest(_) => {
                        instance.pipeline(EFFECT_PASS_OPAQUE_MASKED).ok_or(
                            Error::EffectPipelineNotFound(EFFECT_PASS_OPAQUE_MASKED.to_owned()),
                        )?
                    }
                };
                let transparent = match source.blend {
                    kiri_assets::MeshMaterialBlend::Opaque => PipelineHandle::invalid(),
                    kiri_assets::MeshMaterialBlend::AlphaBlend => {
                        instance.pipeline(EFFECT_PASS_TRANSPARENT).ok_or(
                            Error::EffectPipelineNotFound(EFFECT_PASS_TRANSPARENT.to_owned()),
                        )?
                    }
                    kiri_assets::MeshMaterialBlend::AlphaTest(_) => PipelineHandle::invalid(),
                };
                let shadow = match source.blend {
                    kiri_assets::MeshMaterialBlend::Opaque => {
                        instance.pipeline(EFFECT_PASS_DEPTH).unwrap_or_default()
                    }
                    kiri_assets::MeshMaterialBlend::AlphaBlend => PipelineHandle::invalid(),
                    kiri_assets::MeshMaterialBlend::AlphaTest(_) => instance
                        .pipeline(EFFECT_PASS_DEPTH_MASKED)
                        .unwrap_or_default(),
                };
                material = Some(RenderMeshMaterial {
                    instance,
                    main,
                    transparent,
                    depth: shadow,
                    effect,
                });
                break;
            }
        }
        material.ok_or(Error::MaterialNotFound)
    }

    async fn load_model(manager: Arc<ResourceManager>, name: String) -> Result<RenderModel, Error> {
        let asset: ModelAsset =
            spawn_io(Self::load_or_compile_asset(ModelSource::new(&name))).await?;
        let mut builder = RenderModelBuilder::new(
            &asset.vertex_positions,
            &asset.vertex_attributes,
            &asset.indices,
        )
        .name(&name);
        let materials = asset
            .materials
            .into_iter()
            .map(|x| {
                manager
                    .materials
                    .get_or_load(x.clone(), || spawn(Self::load_material(manager.clone(), x)))
            })
            .collect::<Vec<_>>();
        let mut loaded_materials = Vec::default();
        for material in materials {
            loaded_materials.push(Arc::new(manager.materials.wait(material).await?));
        }
        for mesh in asset.meshes {
            let mut mesh_builder = RenderMeshBuilder::new(mesh.first_vertex, mesh.first_index)
                .bounds(BoundingBox::from_arrays(mesh.bounds.0, mesh.bounds.1))
                .position_scale(mesh.position_scale)
                .uv_scale(mesh.uv_scale);
            for surface in mesh.surfaces {
                mesh_builder.surface(
                    surface.first_index,
                    surface.index_count,
                    &loaded_materials[surface.material as usize],
                );
            }
            builder.add_mesh(mesh_builder);
        }
        asset.nodes.into_iter().for_each(|x| {
            builder.add_node(
                x.parent,
                &x.name,
                Affine3A::from_scale_rotation_translation(
                    Vec3::from_array(x.scale),
                    Quat::from_array(x.rotation),
                    Vec3::from_array(x.translation),
                ),
            );
        });
        asset
            .node_to_mesh
            .into_iter()
            .for_each(|(node, mesh)| builder.attach_mesh(node, mesh));
        debug!("Create model {}", name);
        Ok(builder.build(&manager.renderer)?)
    }
}
