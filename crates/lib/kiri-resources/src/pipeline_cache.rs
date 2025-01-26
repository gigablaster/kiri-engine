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

use kiri_assets::{CompiledAssetPath, EffectAsset, SourceAssetPath};
use kiri_backend::{DescriptorSetLayoutDesc, InputVertexStreamLayout, RenderPassLayout};
use kiri_common::block_on;
use kiri_gfx::{RasterPipelineDesc, RasterPipelineHandle, RenderEffect, Renderer};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use crate::{load_asset_from_vfs, Error};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RasterPipelineCacheEntry {
    pub effect: String,
    pub technique: String,
    pub pass_layout: &'static RenderPassLayout<'static>,
    pub descriptors_layout: &'static [DescriptorSetLayoutDesc<'static>],
    pub input_layout: &'static [InputVertexStreamLayout<'static>],
}

impl RasterPipelineCacheEntry {
    pub fn new(
        effect: &str,
        technique: &str,
        input_layout: &'static [InputVertexStreamLayout<'static>],
        render_pass: &'static RenderPassLayout<'static>,
        descriptors_layout: &'static [DescriptorSetLayoutDesc<'static>],
    ) -> Self {
        Self {
            effect: effect.into(),
            technique: technique.into(),
            pass_layout: render_pass,
            descriptors_layout,
            input_layout,
        }
    }
}

#[derive(Debug)]
pub struct PipelineCache {
    renderer: Arc<Renderer>,
    raster_pipelines: RwLock<HashMap<RasterPipelineCacheEntry, RasterPipelineHandle>>,
    render_effects: RwLock<HashMap<CompiledAssetPath, Arc<RenderEffect>>>,
}

unsafe impl Send for PipelineCache {}
unsafe impl Sync for PipelineCache {}

impl PipelineCache {
    pub fn new(renderer: Arc<Renderer>) -> Self {
        Self {
            raster_pipelines: Default::default(),
            render_effects: Default::default(),
            renderer,
        }
    }

    fn get_or_create_render_effect(
        &self,
        descriptor_layout: &'static [DescriptorSetLayoutDesc<'static>],
        effect: &str,
    ) -> Result<Arc<RenderEffect>, Error> {
        let key = SourceAssetPath::new(effect).compiled()?;
        let effects = self.render_effects.upgradable_read();
        if let Some(effect) = effects.get(&key) {
            Ok(effect.clone())
        } else {
            let mut effects = RwLockUpgradableReadGuard::upgrade(effects);
            if let Some(effect) = effects.get(&key) {
                Ok(effect.clone())
            } else {
                let asset = block_on(load_asset_from_vfs::<EffectAsset>(&key))?;
                let effect = Arc::new(RenderEffect::new(&self.renderer, descriptor_layout, asset)?);
                effects.insert(key, effect.clone());
                Ok(effect)
            }
        }
    }

    pub fn get_or_create_raster_pipeline(
        &self,
        effect: &str,
        technique: &str,
        pass_layout: &'static RenderPassLayout<'static>,
        descriptors_layout: &'static [DescriptorSetLayoutDesc<'static>],
        input_layout: &'static [InputVertexStreamLayout<'static>],
    ) -> Result<RasterPipelineHandle, Error> {
        let pipelines = self.raster_pipelines.upgradable_read();
        let key = RasterPipelineCacheEntry::new(
            effect,
            technique,
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
                let render_effect = self.get_or_create_render_effect(descriptors_layout, effect)?;
                let techinque = render_effect
                    .techinque(technique)
                    .ok_or(Error::RenderTechinqueNotFound(technique.into()))?;

                let pipeline = self.renderer.create_raster_pipeline(RasterPipelineDesc {
                    program: render_effect.program,
                    pass_layout,
                    input_layout,
                    specialization: techinque.spec.to_vec(),
                    desc: techinque.desc,
                });
                pipelines.insert(key, pipeline);
                Ok(pipeline)
            }
        }
    }
}
