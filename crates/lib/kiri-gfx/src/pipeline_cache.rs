// Copyright (C) 2024-2025 gigablaster

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

use std::{collections::HashMap, sync::Arc};

use kiri_assets::{load_asset_from_vfs, AssetSource, ShaderAsset, ShaderAssetSource};
use kiri_backend::vulkan::{
    DescriptorLayoutDesc, GraphicsDevice, InputVertexStreamLayout, RasterPipelineCreateDesc,
    RasterPipelineHandle, RenderPassLayout,
};
use kiri_common::block_on;
use kiri_vfs::AssetReference;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use crate::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RasterPipelineCacheEntry {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub pass_layout: &'static RenderPassLayout<'static>,
    pub descriptors_layout: &'static [DescriptorLayoutDesc<'static>],
    pub input_layout: &'static [InputVertexStreamLayout<'static>],
}

impl RasterPipelineCacheEntry {
    pub fn new(
        vertex_shader: &str,
        fragment_shader: &str,
        input_layout: &'static [InputVertexStreamLayout<'static>],
        render_pass: &'static RenderPassLayout<'static>,
        descriptors_layout: &'static [DescriptorLayoutDesc<'static>],
    ) -> Self {
        Self {
            vertex_shader: vertex_shader.into(),
            fragment_shader: fragment_shader.into(),
            pass_layout: render_pass,
            descriptors_layout,
            input_layout,
        }
    }
}

#[derive(Debug)]
pub struct PipelineCache {
    device: Arc<GraphicsDevice>,
    raster_pipelines: RwLock<HashMap<RasterPipelineCacheEntry, RasterPipelineHandle>>,
    shaders: RwLock<HashMap<AssetReference, Arc<ShaderAsset>>>,
}

unsafe impl Send for PipelineCache {}
unsafe impl Sync for PipelineCache {}

impl PipelineCache {
    pub fn new(device: Arc<GraphicsDevice>) -> Arc<Self> {
        Arc::new(Self {
            raster_pipelines: Default::default(),
            shaders: Default::default(),
            device,
        })
    }

    fn get_or_create_shader(&self, shader: ShaderAssetSource) -> Result<Arc<ShaderAsset>, Error> {
        let reference = shader.reference();
        let effects = self.shaders.upgradable_read();
        if let Some(effect) = effects.get(&reference) {
            Ok(effect.clone())
        } else {
            let mut effects = RwLockUpgradableReadGuard::upgrade(effects);
            if let Some(effect) = effects.get(&reference) {
                Ok(effect.clone())
            } else {
                let asset = Arc::new(block_on(load_asset_from_vfs::<ShaderAsset>(reference))?);
                effects.insert(reference, asset.clone());
                Ok(asset)
            }
        }
    }

    pub fn get_or_create_raster_pipeline(
        &self,
        vertex_shader: &str,
        fragment_shader: &str,
        pass_layout: &'static RenderPassLayout<'static>,
        descriptors_layout: &'static [DescriptorLayoutDesc<'static>],
        input_layout: &'static [InputVertexStreamLayout<'static>],
        specialization: &[(u32, u32)],
        desc: RasterPipelineCreateDesc,
    ) -> Result<RasterPipelineHandle, Error> {
        let pipelines = self.raster_pipelines.upgradable_read();
        let key = RasterPipelineCacheEntry::new(
            vertex_shader,
            fragment_shader,
            input_layout,
            pass_layout,
            descriptors_layout,
        );
        if let Some(handle) = pipelines.get(&key) {
            Ok(*handle)
        } else {
            let mut pipelines = RwLockUpgradableReadGuard::upgrade(pipelines);
            if let Some(handle) = pipelines.get(&key) {
                Ok(*handle)
            } else {
                let vertex_shader =
                    self.get_or_create_shader(ShaderAssetSource::vertex(vertex_shader))?;
                let fragment_shader =
                    self.get_or_create_shader(ShaderAssetSource::fragment(fragment_shader))?;
                let pipeline = self.device.create_raster_pipeline(
                    &vertex_shader.bytecode,
                    &fragment_shader.bytecode,
                    descriptors_layout,
                    pass_layout,
                    input_layout,
                    specialization,
                    desc,
                );
                pipelines.insert(key, pipeline);
                Ok(pipeline)
            }
        }
    }
}
