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

use kiri_assets::{ImageAsset, ImageReference, MeshAssetMaterial, ModelAsset, load_asset_from_vfs};
use kiri_backend::vulkan::{GraphicsDevice, ImageCreateDesc, ImageHandle, ImageUploadData};
use kiri_common::{Handle, Pool, Task, block_on, spawn, yield_now};
use kiri_gfx::{
    PipelineCache, RenderMeshBuilder, RenderModel, RenderModelBuilder, ShaderUniforms,
    material::{BasicMaterial, BasicMaterialBuilder, BasicMaterialType},
};
use kiri_math::{Affine3A, BoundingBox, Quat, Vec3};
use kiri_vfs::AssetReference;
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::Error;

#[derive(Debug)]
pub enum Resource<T: Send + Sync> {
    Loading(Task<Result<T, Error>>),
    Failed(Box<Error>),
    Loaded(T),
}

impl<T: Send + Sync> Resource<T> {
    pub fn resolve(&mut self) {
        if let Resource::Loading(task) = self {
            match block_on(task) {
                Ok(resource) => *self = Self::Loaded(resource),
                Err(error) => *self = Self::Failed(Box::new(error)),
            }
        }
    }

    pub fn is_ready(&self) -> bool {
        match self {
            Resource::Loading(task) => task.is_finished(),
            _ => true,
        }
    }
}

#[derive(Debug)]
pub struct ResourceType<K: Debug + Hash + Eq, T: Clone + Send + Sync + Debug> {
    resources: Mutex<Pool<Resource<T>>>,
    sources: RwLock<HashMap<K, Handle<Resource<T>>>>,
}

unsafe impl<K: Debug + Hash + Eq, T: Clone + Send + Sync + Debug + 'static> Send
    for ResourceType<K, T>
{
}
unsafe impl<K: Debug + Hash + Eq, T: Clone + Send + Sync + Debug + 'static> Sync
    for ResourceType<K, T>
{
}

impl<K: Debug + Hash + Eq + Clone, T: Clone + Send + Sync + Debug + 'static> ResourceType<K, T> {
    pub fn get_or_load<
        F: Future<Output = Result<T, Error>> + Send + 'static,
        LOAD: FnOnce(K) -> F,
    >(
        &self,
        source: K,
        load: LOAD,
    ) -> Handle<Resource<T>> {
        let sources = self.sources.upgradable_read();
        if let Some(handle) = sources.get(&source) {
            *handle
        } else {
            let mut sources = RwLockUpgradableReadGuard::upgrade(sources);
            if let Some(handle) = sources.get(&source) {
                *handle
            } else {
                let load = spawn(load(source.clone()));
                let resource = Resource::Loading(load);
                let handle = self.resources.lock().push(resource);
                sources.insert(source, handle);
                handle
            }
        }
    }

    async fn wait(&self, handle: Handle<Resource<T>>) -> Result<T, Error> {
        loop {
            {
                let mut resources = self.resources.lock();
                let resource = resources.get_mut(handle).ok_or(Error::FailedToLoad)?;
                if resource.is_ready() {
                    resource.resolve();
                    match resource {
                        Resource::Failed(_) => return Err(Error::FailedToLoad),
                        Resource::Loaded(resource) => return Ok(resource.clone()),
                        _ => panic!("Resource must be resolved at this point"),
                    }
                }
            }
            yield_now().await;
        }
    }
}

pub trait ResourceCache {
    fn get_or_load_image(&self, reference: AssetReference) -> Handle<Resource<ImageHandle>>;
    fn get_or_load_model(&self, reference: AssetReference) -> Handle<Resource<Arc<RenderModel>>>;
    fn get_or_load_material(
        &self,
        source: MeshAssetMaterial,
    ) -> Handle<Resource<Arc<BasicMaterial>>>;
}

pub struct ResourceManager {
    device: Arc<GraphicsDevice>,
    cache: Arc<PipelineCache>,
    uniforms: Arc<ShaderUniforms>,
    images: Arc<ResourceType<AssetReference, ImageHandle>>,
    materials: Arc<ResourceType<MeshAssetMaterial, Arc<BasicMaterial>>>,
    models: Arc<ResourceType<AssetReference, Arc<RenderModel>>>,
}

impl ResourceCache for Arc<ResourceManager> {
    fn get_or_load_image(&self, reference: AssetReference) -> Handle<Resource<ImageHandle>> {
        self.images.get_or_load(reference, |source| {
            ResourceManager::load_image(self.clone(), source)
        })
    }

    fn get_or_load_model(&self, reference: AssetReference) -> Handle<Resource<Arc<RenderModel>>> {
        self.models.get_or_load(reference, |reference| {
            ResourceManager::load_model(self.clone(), reference)
        })
    }

    fn get_or_load_material(
        &self,
        source: MeshAssetMaterial,
    ) -> Handle<Resource<Arc<BasicMaterial>>> {
        self.materials.get_or_load(source, |source| {
            ResourceManager::load_material(self.clone(), source)
        })
    }
}

impl ResourceManager {
    async fn load_image(
        cache: Arc<ResourceManager>,
        source: AssetReference,
    ) -> Result<ImageHandle, Error> {
        let asset: ImageAsset = load_asset_from_vfs(source).await?;
        let image = cache.device.create_image(
            ImageCreateDesc::texture(
                asset.format,
                [asset.dims[0] as usize, asset.dims[1] as usize],
            )
            .sampled()
            .transfer_desitnation(),
        )?;
        cache.device.upload_image_data(
            image,
            0,
            asset.mips.iter().map(|mip| ImageUploadData { data: mip }),
        )?;
        Ok(image)
    }

    async fn load_model(
        cache: Arc<ResourceManager>,
        source: AssetReference,
    ) -> Result<Arc<RenderModel>, Error> {
        let asset: ModelAsset = load_asset_from_vfs(source).await?;
        let mut builder = RenderModelBuilder::new(&asset.vertices, &asset.indices);
        for mesh in &asset.meshes {
            let mut mesh_builder =
                RenderMeshBuilder::new(mesh.first_vertex as usize, mesh.first_index as usize)
                    .bounds(BoundingBox::from_extent(
                        Vec3::from_array(mesh.bounds.0),
                        Vec3::from_array(mesh.bounds.1),
                    ))
                    .position_scale(mesh.position_scale)
                    .uv_scale(mesh.uv_scale);
            for surface in &mesh.surfaces {
                let material =
                    cache.get_or_load_material(asset.materials[surface.material as usize].clone());
                let material = cache.materials.wait(material).await?;
                mesh_builder.surface(surface.first_index, surface.index_count, material.as_ref());
            }
            builder.add_mesh(mesh_builder);
        }
        for node in &asset.nodes {
            builder.add_node(
                node.parent,
                &node.name,
                Affine3A::from_scale_rotation_translation(
                    Vec3::from_array(node.scale),
                    Quat::from_array(node.rotation),
                    Vec3::from_array(node.translation),
                ),
            );
        }
        for (node, mesh) in asset.node_to_mesh {
            builder.attach_mesh(node, mesh);
        }
        Ok(Arc::new(builder.build(cache.device.clone())?))
    }

    async fn load_material(
        cache: Arc<ResourceManager>,
        material: MeshAssetMaterial,
    ) -> Result<Arc<BasicMaterial>, Error> {
        let ty = match material.blend {
            kiri_assets::MeshMaterialBlend::Opaque => BasicMaterialType::Opaque,
            kiri_assets::MeshMaterialBlend::AlphaBlend => BasicMaterialType::Transparent,
            kiri_assets::MeshMaterialBlend::AlphaTest(test) => BasicMaterialType::Masked(test),
        };
        let material = BasicMaterialBuilder {
            ty,
            emissive_power: material.emissive_power,
            basic_color: Self::load_image_reference(cache.clone(), material.base_color).await?,
            metallic_roughnbess: Self::load_image_reference(
                cache.clone(),
                material.metallic_roughness,
            )
            .await?,
            normals: Self::load_image_reference(cache.clone(), material.normals).await?,
            occlusion: Self::load_image_reference(cache.clone(), material.occlusion).await?,
            emissive: Self::load_image_reference(cache.clone(), material.emissive).await?,
        }
        .build(cache.device.clone(), cache.uniforms.clone(), &cache.cache)?;
        Ok(Arc::new(material))
    }

    async fn load_image_reference(
        cache: Arc<ResourceManager>,
        reference: ImageReference,
    ) -> Result<ImageHandle, Error> {
        match reference {
            ImageReference::External(asset_reference) => {
                cache
                    .images
                    .wait(cache.get_or_load_image(asset_reference))
                    .await
            }
            ImageReference::Embedded(embedded_image) => {
                let image = cache.device.create_image(
                    ImageCreateDesc::texture(
                        embedded_image.format,
                        [
                            embedded_image.dims[0] as usize,
                            embedded_image.dims[1] as usize,
                        ],
                    )
                    .sampled()
                    .transfer_desitnation(),
                )?;
                cache.device.upload_image_data(
                    image,
                    0,
                    [ImageUploadData {
                        data: &embedded_image.pixels,
                    }],
                )?;
                Ok(image)
            }
        }
    }
}
