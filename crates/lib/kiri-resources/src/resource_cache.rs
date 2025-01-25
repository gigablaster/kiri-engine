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

use kiri_assets::load_or_compile_asset;
use kiri_assets::{ImageAsset, ImageSource, MeshAssetMaterial, ModelAsset, ModelSource};
use kiri_backend::ash::vk;
use kiri_backend::{ImageCreateDesc, ImageUploadData};
use kiri_common::{block_on, spawn, yield_now, Task};
use kiri_gfx::{
    DescriptorSetCreateDesc, GpuPbrMeshMaterialData, ImageHandle, RenderMeshBuilder,
    RenderMeshMaterial, RenderMeshMaterialOrder, RenderModel, RenderModelBuilder, Renderer,
    MESH_PBR_MATERIAL_DESCRIPTOR_LAYOUT,
};
use kiri_math::{Affine3A, BoundingBox, Quat, Vec3};
use log::{debug, error};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::Error;

#[derive(Debug, Clone, Copy)]
pub struct ModelHandle(u32);
#[derive(Debug, Clone, Copy)]
pub struct TextureHandle(u32);
#[derive(Debug, Clone, Copy)]
pub struct MaterialHandle(u32);

pub trait ResourceHandle: Copy {
    fn from_index(index: usize) -> Self;
    fn to_index(self) -> usize;
}

impl ResourceHandle for ModelHandle {
    fn from_index(index: usize) -> Self {
        Self(index as u32)
    }

    fn to_index(self) -> usize {
        self.0 as usize
    }
}

impl ResourceHandle for TextureHandle {
    fn from_index(index: usize) -> Self {
        Self(index as u32)
    }

    fn to_index(self) -> usize {
        self.0 as usize
    }
}

impl ResourceHandle for MaterialHandle {
    fn from_index(index: usize) -> Self {
        Self(index as u32)
    }

    fn to_index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug)]
pub enum Resource<T: Debug + Send + Sync + Clone> {
    Loading,
    Failed,
    Loaded(T),
}

impl<T: Debug + Send + Sync + Clone> Clone for Resource<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Loading => Self::Loading,
            Self::Failed => Self::Failed,
            Self::Loaded(arg0) => Self::Loaded(arg0.clone()),
        }
    }
}

pub type LoadingTask<T, H> = (H, Task<Result<T, Error>>);

#[derive(Debug)]
pub struct ResourceType<K: Hash + Eq, T: Debug + Send + Sync + Clone, H: ResourceHandle> {
    pool: Mutex<Vec<Resource<T>>>,
    names: RwLock<HashMap<K, H>>,
    loading: Mutex<Vec<LoadingTask<T, H>>>,
}

impl<K: Hash + Eq, T: Debug + Send + Sync + Clone, H: ResourceHandle> Default
    for ResourceType<K, T, H>
{
    fn default() -> Self {
        Self {
            pool: Default::default(),
            names: Default::default(),
            loading: Default::default(),
        }
    }
}

impl<K: Hash + Eq, T: Debug + Send + Sync + Clone, H: ResourceHandle> ResourceType<K, T, H> {
    fn get_or_load<LOAD: FnOnce() -> Task<Result<T, Error>>>(&self, key: K, load: LOAD) -> H {
        let names = self.names.upgradable_read();
        if let Some(handle) = names.get(&key) {
            *handle
        } else {
            let mut names = RwLockUpgradableReadGuard::upgrade(names);
            if let Some(handle) = names.get(&key) {
                *handle
            } else {
                let task = load();
                let mut pool = self.pool.lock();
                let handle = H::from_index(pool.len());
                pool.push(Resource::Loading);

                names.insert(key, handle);
                self.loading.lock().push((handle, task));
                handle
            }
        }
    }

    fn resolve(&self, handle: H) -> Option<Resource<T>> {
        self.pool.lock().get(handle.to_index()).cloned()
    }

    async fn wait(&self, handle: H) -> Result<T, Error> {
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

    fn tick(&self) {
        let mut pool = self.pool.lock();
        let mut loading = self.loading.lock();
        let mut i = 0;
        while i < loading.len() {
            if loading[i].1.is_finished() {
                let (handle, task) = loading.remove(i);
                match block_on(task) {
                    Ok(resource) => {
                        pool[handle.to_index()] = Resource::Loaded(resource);
                    }
                    Err(err) => {
                        error!("Failed to load asset: {}", err);
                        pool[handle.to_index()] = Resource::Failed;
                    }
                }
            } else {
                i += 1;
            }
        }
    }
}

pub trait ResourceLoader {
    fn get_or_load_texture(&self, source: &ImageSource) -> TextureHandle;
    fn get_or_load_model(&self, name: &str) -> ModelHandle;
}

#[derive(Debug)]
pub struct ResourceCache {
    pub renderer: Arc<Renderer>,
    textures: ResourceType<ImageSource, ImageHandle, TextureHandle>,
    materials: ResourceType<MeshAssetMaterial, RenderMeshMaterial, MaterialHandle>,
    models: ResourceType<String, Arc<RenderModel>, ModelHandle>,
}

impl ResourceLoader for Arc<ResourceCache> {
    fn get_or_load_texture(&self, source: &ImageSource) -> TextureHandle {
        let source = source.clone();
        self.textures.get_or_load(source.clone(), || {
            spawn(ResourceCache::load_texture(self.clone(), source))
        })
    }

    fn get_or_load_model(&self, name: &str) -> ModelHandle {
        self.models.get_or_load(name.to_owned(), || {
            spawn(ResourceCache::load_model(self.clone(), name.to_owned()))
        })
    }
}

pub struct ResourceResolveContext<'a> {
    models: &'a Vec<Resource<Arc<RenderModel>>>,
}

impl ResourceResolveContext<'_> {
    pub fn resolve_model(&self, handle: ModelHandle) -> Result<Option<Arc<RenderModel>>, Error> {
        let resource = self
            .models
            .get(handle.to_index())
            .ok_or(Error::InvalidModelHandle(handle))?;
        match resource {
            Resource::Loading => Ok(None),
            Resource::Failed => Err(Error::ResourceLoadingFailed),
            Resource::Loaded(model) => Ok(Some(model.clone())),
        }
    }
}

impl ResourceCache {
    pub fn new(renderer: &Arc<Renderer>) -> Result<Arc<Self>, Error> {
        debug!("Create resource cache");
        Ok(Arc::new(Self {
            renderer: renderer.clone(),
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

    pub fn get_model(&self, handle: ModelHandle) -> Option<Resource<Arc<RenderModel>>> {
        self.models.resolve(handle)
    }

    pub fn get_texture(&self, handle: TextureHandle) -> Option<Resource<ImageHandle>> {
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

    async fn load_texture(
        manager: Arc<ResourceCache>,
        source: ImageSource,
    ) -> Result<ImageHandle, Error> {
        match &source.data {
            kiri_assets::ImageData::Path(_) => {
                let asset: ImageAsset = load_or_compile_asset(source.clone()).await?;
                let mips = asset
                    .mips
                    .iter()
                    .map(|x| ImageUploadData::new(x))
                    .collect::<Vec<_>>();
                debug!("Create texture {:?}", source);
                let dims = [asset.dims[0] as usize, asset.dims[1] as usize];
                Ok(manager
                    .renderer
                    .create_image(ImageCreateDesc::texture(asset.format, dims), Some(&mips))?)
            }
            kiri_assets::ImageData::Color(color) => Ok(manager.renderer.create_image(
                ImageCreateDesc::new(source.uncompressed_format(), [1, 1]),
                Some(&[ImageUploadData::new(color)]),
            )?),
        }
    }

    pub fn resolve<CB: FnOnce(ResourceResolveContext)>(&self, cb: CB) {
        let models = self.models.pool.lock();
        cb(ResourceResolveContext { models: &models });
    }

    async fn load_material(
        manager: Arc<ResourceCache>,
        source: MeshAssetMaterial,
    ) -> Result<RenderMeshMaterial, Error> {
        let images = futures::future::try_join_all(
            [
                manager.get_or_load_texture(&source.base_color),
                manager.get_or_load_texture(&source.metallic_roughness),
                manager.get_or_load_texture(&source.normals),
                manager.get_or_load_texture(&source.occlusion),
                manager.get_or_load_texture(&source.emissive),
            ]
            .into_iter()
            .map(|handle| manager.textures.wait(handle)),
        )
        .await?;
        let uniform = manager.renderer.allocate_uniform(GpuPbrMeshMaterialData {
            emissive_power: source.emissive_power,
            alpha_cutoff: source.alpha_cutoff(),
        })?;
        let descriptor =
            manager
                .renderer
                .with_descriptors()
                .create_descriptor(DescriptorSetCreateDesc {
                    layout: MESH_PBR_MATERIAL_DESCRIPTOR_LAYOUT,
                    stages: vk::ShaderStageFlags::ALL_GRAPHICS,
                    images: &images,
                    unifoms: &[uniform],
                    ..Default::default()
                })?;
        let order = match source.blend {
            kiri_assets::MeshMaterialBlend::Opaque => RenderMeshMaterialOrder::Opaque,
            kiri_assets::MeshMaterialBlend::AlphaBlend => RenderMeshMaterialOrder::Transparent,
            kiri_assets::MeshMaterialBlend::AlphaTest(_) => RenderMeshMaterialOrder::Masked,
        };
        Ok(RenderMeshMaterial {
            ty: kiri_gfx::RenderMeshMaterialType::PBR,
            order,
            descriptor,
            uniform,
        })
    }

    async fn load_model(
        manager: Arc<ResourceCache>,
        name: String,
    ) -> Result<Arc<RenderModel>, Error> {
        let asset: ModelAsset = load_or_compile_asset(ModelSource::new(&name)).await?;
        let mut builder = RenderModelBuilder::new(&asset.vertices, &asset.indices).name(&name);
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
            loaded_materials.push(manager.materials.wait(material).await?);
        }
        for mesh in asset.meshes {
            let mut mesh_builder =
                RenderMeshBuilder::new(mesh.first_vertex as _, mesh.first_index as _)
                    .bounds(BoundingBox::from_arrays(mesh.bounds.0, mesh.bounds.1))
                    .position_scale(mesh.position_scale)
                    .uv_scale(mesh.uv_scale);
            for surface in mesh.surfaces {
                mesh_builder.surface(
                    surface.first_index,
                    surface.index_count,
                    loaded_materials[surface.material as usize],
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
        Ok(Arc::new(builder.build(&manager.renderer)?))
    }
}
