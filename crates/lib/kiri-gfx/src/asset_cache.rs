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

use arrayvec::ArrayVec;
use bevy_tasks::{block_on, IoTaskPool, Task};
use bytes::Bytes;
use kiri_assets::{
    Asset, AssetReference, AssetSource, GltfAsset, GltfMeshSource, GltfSceneSource, ImageAsset,
    ImageAssetSource, ImageAssetType, MeshMaterialBlend, ShaderAssetSource, StaticMeshAsset,
};
use kiri_backend::{
    BindGroupDesc, BufferCreateDesc, BufferHandle, BufferSlice, ImageAspect, ImageCreateDesc,
    ImageHandle, ImageSubresourceData, InputVertexStreamDesc, PipelineHandle, ProgramHandle,
    RasterPipelineCreateDesc, RenderDevice, RenderPassHandle, ShaderDesc, ShaderStage,
};
use kiri_common::{DynamicAllocator, Handle, Pool};
use kiri_vfs::vfs_load;
use log::{debug, error};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{
    Error, PbrMaterialShaderData, RenderMeshMaterial, RenderMeshShaderData, RenderMeshSurface,
    RenderScene, RenderSceneGroup, StaticRenderMesh, PBR_MATERIAL_BIND_GROUP,
    STATIC_MESH_BIND_GROUP,
};

pub type StaticMeshHandle = Handle<StaticRenderMesh>;
pub type SceneHandle = Handle<RenderSceneGroup>;

type StaticMeshPool = Pool<StaticRenderMesh>;
type ScenePool = Pool<RenderSceneGroup>;

type LoadingImageTask = Task<()>;

#[derive(Debug, Default)]
struct AssetLifetimeTracker<T: Hash + Eq + PartialEq>(
    RwLock<HashMap<T, (AtomicUsize, AssetReference)>>,
);

impl<T: Hash + Eq + PartialEq> AssetLifetimeTracker<T> {
    fn track(&self, handle: T, reference: AssetReference) {
        self.0
            .write()
            .insert(handle, (AtomicUsize::new(1), reference));
    }

    fn add_ref(&self, handle: T) {
        self.0.read().get(&handle).iter().for_each(|(count, _)| {
            count.fetch_add(1, Ordering::AcqRel);
        });
    }

    fn release(&self, handle: T) -> Option<AssetReference> {
        let mut tracker = self.0.write();
        if let Some((count, reference)) = tracker.get(&handle) {
            let reference = *reference;
            if count.fetch_sub(1, Ordering::AcqRel) == 0 {
                tracker.remove(&handle);
                return Some(reference);
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RasterPipelineDesc {
    pub vertex_shader: Option<String>,
    pub fragment_shader: Option<String>,
    pub layout: &'static [BindGroupDesc<'static>],
    pub streams: &'static [InputVertexStreamDesc<'static>],
    pub pass: RenderPassHandle,
    pub subpass: u32,
    pub desc: RasterPipelineCreateDesc,
}

impl RasterPipelineDesc {
    pub fn new(
        layout: &'static [BindGroupDesc<'static>],
        streams: &'static [InputVertexStreamDesc<'static>],
    ) -> Self {
        Self {
            vertex_shader: None,
            fragment_shader: None,
            layout,
            pass: Default::default(),
            subpass: 0,
            streams,
            desc: Default::default(),
        }
    }

    pub fn vertex_shader(mut self, name: &str) -> Self {
        self.vertex_shader = Some(name.to_owned());
        self
    }

    pub fn fragment_shader(mut self, name: &str) -> Self {
        self.fragment_shader = Some(name.to_owned());
        self
    }

    pub fn pass(mut self, pass: RenderPassHandle, subpass: usize) -> Self {
        self.pass = pass;
        self.subpass = subpass as u32;
        self
    }

    pub fn desc(&mut self) -> &mut RasterPipelineCreateDesc {
        &mut self.desc
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProgramKey {
    vertex_shader: Option<String>,
    fragment_shader: Option<String>,
    layout: &'static [BindGroupDesc<'static>],
}

#[derive(Debug)]
pub struct AssetCache {
    device: Arc<RenderDevice>,
    image_assets: RwLock<HashMap<AssetReference, ImageHandle>>,
    loading_images: Mutex<Vec<LoadingImageTask>>,
    mesh_pool: BufferHandle,
    mesh_allocator: Mutex<DynamicAllocator>,
    static_meshes: RwLock<StaticMeshPool>,
    static_mesh_assets: RwLock<HashMap<AssetReference, StaticMeshHandle>>,
    scenes: RwLock<ScenePool>,
    scene_assets: RwLock<HashMap<AssetReference, SceneHandle>>,
    image_tracking: AssetLifetimeTracker<ImageHandle>,
    static_mesh_tracking: AssetLifetimeTracker<StaticMeshHandle>,
    scene_tracking: AssetLifetimeTracker<SceneHandle>,
    programs: RwLock<HashMap<ProgramKey, ProgramHandle>>,
    shaders: RwLock<HashMap<AssetReference, Bytes>>,
    pipelines: RwLock<HashMap<RasterPipelineDesc, PipelineHandle>>,
}

const MESH_POOL_SIZE: usize = 256 * 1024 * 1024;

impl Drop for AssetCache {
    fn drop(&mut self) {
        debug!("Asset cache cleanup");
        self.static_meshes.write().drain().for_each(|mesh| {
            self.device.destroy_bind_group(mesh.object_bind_group);
            mesh.materials
                .iter()
                .for_each(|material| self.device.destroy_bind_group(material.bind_group));
        });
        self.image_assets
            .write()
            .drain()
            .for_each(|(_, handle)| self.device.destroy_image(handle));
        self.device.destroy_buffer(self.mesh_pool);
    }
}

impl AssetCache {
    pub fn new(device: &Arc<RenderDevice>) -> Result<Arc<Self>, Error> {
        debug!("Create asset cache");
        Ok(Arc::new(Self {
            device: device.clone(),
            image_assets: Default::default(),
            loading_images: Default::default(),
            mesh_pool: device.create_buffer(
                BufferCreateDesc::gpu(MESH_POOL_SIZE)
                    .index_buffer()
                    .veretex_buffer()
                    .storage_buffer()
                    .transfer_destination()
                    .name("Geometry"),
                None,
            )?,
            mesh_allocator: Mutex::new(DynamicAllocator::new(
                MESH_POOL_SIZE,
                device.properties().buffer_allocation_granularity,
            )),
            static_meshes: Default::default(),
            static_mesh_assets: Default::default(),
            scenes: Default::default(),
            scene_assets: Default::default(),
            image_tracking: Default::default(),
            static_mesh_tracking: Default::default(),
            scene_tracking: Default::default(),
            programs: Default::default(),
            shaders: Default::default(),
            pipelines: Default::default(),
        }))
    }

    pub fn get_or_load_image(&self, source: ImageAssetSource) -> Result<ImageHandle, Error> {
        Ok(self.get_or_load_image_by_reference(source.reference(), source.ty)?)
    }

    fn get_or_load_image_by_reference(
        &self,
        reference: AssetReference,
        ty: ImageAssetType,
    ) -> Result<ImageHandle, kiri_backend::Error> {
        let images = self.image_assets.upgradable_read();
        if let Some(image) = images.get(&reference) {
            self.image_tracking.add_ref(*image);
            Ok(*image)
        } else {
            let mut images = RwLockUpgradableReadGuard::upgrade(images);
            if let Some(image) = images.get(&reference) {
                self.image_tracking.add_ref(*image);
                Ok(*image)
            } else {
                let handle = self.device.create_image(
                    ImageCreateDesc::new(ty.uncompressed_format(), [1, 1])
                        .sampled()
                        .transfer_desitnation(),
                    Some(&[ImageSubresourceData {
                        data: &ty.default_values(),
                        row_pitch: 0,
                    }]),
                )?;
                images.insert(reference, handle);
                self.loading_images
                    .lock()
                    .push(IoTaskPool::get().spawn(Self::load_image(
                        self.device.clone(),
                        reference,
                        handle,
                    )));
                self.image_tracking.track(handle, reference);
                Ok(handle)
            }
        }
    }

    pub fn get_or_load_static_mesh(
        &self,
        source: &GltfMeshSource,
    ) -> Result<StaticMeshHandle, Error> {
        self.load_static_mesh(source.reference())
    }

    fn load_static_mesh(&self, reference: AssetReference) -> Result<StaticMeshHandle, Error> {
        let meshes = self.static_mesh_assets.upgradable_read();
        if let Some(handle) = meshes.get(&reference) {
            self.static_mesh_tracking.add_ref(*handle);
            Ok(*handle)
        } else {
            let mut meshes = RwLockUpgradableReadGuard::upgrade(meshes);
            if let Some(handle) = meshes.get(&reference) {
                self.static_mesh_tracking.add_ref(*handle);
                Ok(*handle)
            } else {
                let handle = self.do_load_static_mesh(reference)?;
                meshes.insert(reference, handle);
                self.static_mesh_tracking.track(handle, reference);
                Ok(handle)
            }
        }
    }

    async fn load_image(device: Arc<RenderDevice>, reference: AssetReference, handle: ImageHandle) {
        debug!("Load image {}", reference);
        if let Err(err) = Self::do_load_image(&device, reference, handle) {
            error!("Failed to load image: {}", err);
        } else {
            debug!("Image {} loaded", reference);
        }
    }

    pub fn tick(&self) -> Result<(), Error> {
        let mut i = 0;
        let mut loading_images = self.loading_images.lock();
        while i < loading_images.len() {
            if loading_images[i].is_finished() {
                block_on(loading_images.remove(i))
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    fn do_load_image(
        device: &RenderDevice,
        reference: AssetReference,
        handle: ImageHandle,
    ) -> Result<(), Error> {
        let data = vfs_load(reference)?;
        let asset = ImageAsset::load(data)?;
        Self::upload_image(device, handle, reference, asset)
    }

    fn upload_image(
        device: &RenderDevice,
        handle: ImageHandle,
        reference: AssetReference,
        asset: ImageAsset,
    ) -> Result<(), Error> {
        let data = asset
            .mips
            .iter()
            .map(|x| ImageSubresourceData {
                data: x,
                row_pitch: 0,
            })
            .collect::<Vec<_>>();
        device.update_image(
            handle,
            ImageCreateDesc::new(asset.format, asset.dims)
                .mip_levels(asset.mips.len())
                .sampled()
                .transfer_desitnation()
                .name(&reference.to_string()),
            Some(&data),
        )?;
        Ok(())
    }

    pub fn unload_image(&self, handle: ImageHandle) {
        if let Some(reference) = self.image_tracking.release(handle) {
            self.remove_image(reference);
        }
    }

    fn remove_image(&self, reference: AssetReference) {
        self.image_assets
            .write()
            .remove(&reference)
            .iter()
            .copied()
            .for_each(|handle| self.device.destroy_image(handle));
    }

    fn do_load_static_mesh(&self, reference: AssetReference) -> Result<StaticMeshHandle, Error> {
        debug!("Load static mesh {}", reference);
        let data = vfs_load(reference)?;
        let asset = StaticMeshAsset::load(data)?;
        self.upload_static_mesh(asset)
    }

    fn upload_static_mesh(&self, asset: StaticMeshAsset) -> Result<StaticMeshHandle, Error> {
        let mut allocator = self.mesh_allocator.lock();
        let vertex_offset = allocator
            .allocate(mem::size_of_val(&asset.vertices))
            .ok_or(Error::OutOfMeshMemory)?;
        let index_offset = allocator
            .allocate(mem::size_of_val(&asset.indices))
            .ok_or(Error::OutOfMeshMemory)?;
        drop(allocator);
        self.device
            .update_buffer(self.mesh_pool, vertex_offset, &asset.vertices)?;
        self.device
            .update_buffer(self.mesh_pool, index_offset, &asset.indices)?;
        let object_bind_group = self.device.create_bind_group(&STATIC_MESH_BIND_GROUP)?;
        let mut surfaces = Vec::with_capacity(asset.surfaces.len());
        let mut materials = Vec::with_capacity(asset.materials.len());
        for material in &asset.materials {
            materials.push((
                material.clone(),
                self.device.create_bind_group(&PBR_MATERIAL_BIND_GROUP)?,
            ));
        }
        for surface in &asset.surfaces {
            surfaces.push(RenderMeshSurface {
                first_index: surface.first_index,
                index_count: surface.index_count,
                material_index: surface.material as usize,
            });
        }
        let mut mesh = StaticRenderMesh {
            vertices: BufferSlice::new(self.mesh_pool, vertex_offset),
            indices: BufferSlice::new(self.mesh_pool, index_offset),
            object_bind_group,
            surfaces,
            materials: Default::default(),
        };
        self.device.update_bind_groups(|context| {
            context.push_uniform(
                mesh.object_bind_group,
                "object",
                &[RenderMeshShaderData {
                    position_scale: asset.positon_scale,
                    uv1_scale: asset.uv_scale[0],
                    uv2_scale: asset.uv_scale[1],
                }],
            )?;
            for material in &materials {
                let cutoff = if let MeshMaterialBlend::AlphaTest(cutoff) = material.0.blend {
                    cutoff
                } else {
                    1.0
                };
                context.push_uniform(
                    material.1,
                    "material",
                    &[PbrMaterialShaderData {
                        emissive_power: material.0.emissive_power,
                        alpha_cutoff: cutoff,
                    }],
                )?;
                let mut mesh_images = Vec::default();
                for (name, (reference, ty)) in &material.0.images {
                    let image = self.get_or_load_image_by_reference(*reference, *ty)?;
                    mesh_images.push(image);
                    context.bind_image(material.1, name, image, ImageAspect::Color)?;
                }
                mesh.materials.push(RenderMeshMaterial {
                    bind_group: material.1,
                    images: mesh_images,
                })
            }
            Ok(())
        })?;
        let handle = self.static_meshes.write().push(mesh);
        Ok(handle)
    }

    pub fn unload_static_mesh(&self, handle: StaticMeshHandle) {
        if let Some(reference) = self.static_mesh_tracking.release(handle) {
            self.remove_static_mesh(reference);
        }
    }

    fn remove_static_mesh(&self, reference: AssetReference) {
        if let Some(handle) = self.static_mesh_assets.write().remove(&reference) {
            if let Some(mesh) = self.static_meshes.write().remove(handle) {
                mesh.materials.iter().for_each(|material| {
                    self.device.destroy_bind_group(material.bind_group);
                    material
                        .images
                        .iter()
                        .for_each(|image| self.unload_image(*image));
                });
                self.device.destroy_bind_group(mesh.object_bind_group);
            }
        }
    }

    pub fn get_or_load_scene(&self, source: GltfSceneSource) -> Result<SceneHandle, Error> {
        self.load_scene(source.reference())
    }

    fn load_scene(&self, reference: AssetReference) -> Result<SceneHandle, Error> {
        let scenes = self.scene_assets.upgradable_read();
        if let Some(handle) = scenes.get(&reference) {
            self.scene_tracking.add_ref(*handle);
            Ok(*handle)
        } else {
            let mut scenes = RwLockUpgradableReadGuard::upgrade(scenes);
            if let Some(handle) = scenes.get(&reference) {
                self.scene_tracking.add_ref(*handle);
                Ok(*handle)
            } else {
                let handle = self.do_load_scene(reference)?;
                self.scene_tracking.track(handle, reference);
                scenes.insert(reference, handle);
                Ok(handle)
            }
        }
    }

    fn do_load_scene(&self, reference: AssetReference) -> Result<SceneHandle, Error> {
        debug!("Load scene asset: {}", reference);
        let data = vfs_load(reference)?;
        let asset = GltfAsset::load(data)?;
        let mut scenes = HashMap::default();
        for (name, scene) in asset.scenes {
            let mut render_scene = RenderScene::default();
            for mesh in scene.meshes {
                render_scene.meshes.push(self.load_static_mesh(mesh)?);
            }
            scene.bones.into_iter().for_each(|bone| {
                render_scene.parents.push(bone.parent);
                render_scene
                    .local_transforms
                    .push(glam::Mat4::from_scale_rotation_translation(
                        bone.scale.into(),
                        glam::Quat::from_array(bone.rotation),
                        bone.translation.into(),
                    ));
            });
            render_scene.names = scene
                .bone_names
                .into_iter()
                .map(|(name, index)| (name, index as usize))
                .collect();
            render_scene.node_to_mesh = scene
                .bone_to_mesh
                .into_iter()
                .map(|(node, mesh)| (node as usize, mesh as usize))
                .collect();
            scenes.insert(name, render_scene);
        }
        let scene = RenderSceneGroup { scenes };
        let handle = self.scenes.write().push(scene);
        Ok(handle)
    }

    fn get_or_load_program(
        &self,
        vertex_shader: Option<String>,
        fragment_shader: Option<String>,
        layout: &'static [BindGroupDesc<'static>],
    ) -> Result<ProgramHandle, Error> {
        let key = ProgramKey {
            vertex_shader: vertex_shader.clone(),
            fragment_shader: fragment_shader.clone(),
            layout,
        };
        let programs = self.programs.upgradable_read();

        if let Some(program) = programs.get(&key) {
            Ok(*program)
        } else {
            let mut programs = RwLockUpgradableReadGuard::upgrade(programs);
            if let Some(program) = programs.get(&key) {
                Ok(*program)
            } else {
                let vertex_shader_code = if let Some(vertex_shader) = vertex_shader {
                    Some(self.get_or_load_shader(ShaderAssetSource::vertex(&vertex_shader))?)
                } else {
                    None
                };
                let fragment_shader_code = if let Some(fragment_sahder) = fragment_shader {
                    Some(self.get_or_load_shader(ShaderAssetSource::fragment(&fragment_sahder))?)
                } else {
                    None
                };
                let mut shaders = ArrayVec::<_, 2>::new();
                if let Some(vertex_shader) = &vertex_shader_code {
                    shaders.push(ShaderDesc::vertex(vertex_shader));
                }
                if let Some(fragment_shader) = &fragment_shader_code {
                    shaders.push(ShaderDesc::fragment(fragment_shader));
                }
                debug_assert!(!shaders.is_empty(), "Need at least one shader");
                let program = self.device.create_program(&shaders, layout)?;
                programs.insert(key, program);
                Ok(program)
            }
        }
    }

    fn get_or_load_shader(&self, source: ShaderAssetSource) -> Result<Bytes, Error> {
        let reference = source.reference();
        let shaders = self.shaders.upgradable_read();
        if let Some(shader) = shaders.get(&reference) {
            Ok(shader.clone())
        } else {
            let mut shaders = RwLockUpgradableReadGuard::upgrade(shaders);
            if let Some(shader) = shaders.get(&reference) {
                Ok(shader.clone())
            } else {
                let data = vfs_load(reference)?;
                shaders.insert(reference, data.clone());
                Ok(data)
            }
        }
    }

    pub fn get_or_create_pipeline(
        &self,
        desc: RasterPipelineDesc,
    ) -> Result<PipelineHandle, Error> {
        let pipelines = self.pipelines.upgradable_read();
        if let Some(pipeline) = pipelines.get(&desc) {
            Ok(*pipeline)
        } else {
            let mut pipelines = RwLockUpgradableReadGuard::upgrade(pipelines);
            if let Some(pipeline) = pipelines.get(&desc) {
                Ok(*pipeline)
            } else {
                debug!("Create pipeline {:?}", desc);
                let program = self.get_or_load_program(
                    desc.vertex_shader.clone(),
                    desc.fragment_shader.clone(),
                    desc.layout,
                )?;
                let pipeline = self.device.create_pipeline(
                    program,
                    desc.pass,
                    desc.subpass,
                    desc.streams,
                    &desc.desc,
                );
                pipelines.insert(desc, pipeline);
                Ok(pipeline)
            }
        }
    }
}
