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

use std::{collections::HashMap, fmt::Debug, hash::Hash, sync::Arc, u32};

use kiri_assets::{CompiledAssetPath, ShaderAsset, SourceAssetPath};
use kiri_backend::{
    DescriptorSetLayoutDesc, InputVertexStreamLayout, RasterPipelineCreateDesc, RenderPassLayout,
    ShaderDesc,
};
use kiri_common::{block_on, futures::future};
use kiri_gfx::{ProgramHandle, RasterPipelineDesc, RasterPipelineHandle, Renderer};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use crate::{load_asset_from_vfs, Error};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RasterPipelineCacheEntry {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub pass_layout: &'static RenderPassLayout<'static>,
    pub descriptors_layout: &'static [DescriptorSetLayoutDesc<'static>],
    pub input_layout: &'static [InputVertexStreamLayout<'static>],
    pub specialization: Vec<(u32, u32)>,
    pub desc: RasterPipelineCreateDesc,
}

impl RasterPipelineCacheEntry {
    pub fn new(
        vertex_shader: &str,
        fragment_shader: &str,
        input_layout: &'static [InputVertexStreamLayout<'static>],
        render_pass: &'static RenderPassLayout<'static>,
        descriptors_layout: &'static [DescriptorSetLayoutDesc<'static>],
    ) -> Self {
        Self {
            vertex_shader: vertex_shader.into(),
            fragment_shader: fragment_shader.into(),
            pass_layout: render_pass,
            descriptors_layout,
            input_layout,
            specialization: Default::default(),
            desc: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct PipelineCache {
    renderer: Arc<Renderer>,
    raster_pipelines: RwLock<HashMap<RasterPipelineCacheEntry, RasterPipelineHandle>>,
    raster_programs: RwLock<HashMap<(CompiledAssetPath, CompiledAssetPath), ProgramHandle>>,
}

unsafe impl Send for PipelineCache {}
unsafe impl Sync for PipelineCache {}

impl PipelineCache {
    pub fn new(renderer: Arc<Renderer>) -> Self {
        Self {
            raster_pipelines: Default::default(),
            raster_programs: Default::default(),
            renderer,
        }
    }

    pub fn get_or_create_raster_program(
        &self,
        layout: &'static [DescriptorSetLayoutDesc<'static>],
        vertex_shader: &str,
        fragment_shader: &str,
    ) -> Result<ProgramHandle, Error> {
        let key = (
            SourceAssetPath::new(vertex_shader).compiled()?,
            SourceAssetPath::new(fragment_shader).compiled()?,
        );
        let programs = self.raster_programs.upgradable_read();
        if let Some(program) = programs.get(&key) {
            Ok(*program)
        } else {
            let mut programs = RwLockUpgradableReadGuard::upgrade(programs);
            if let Some(program) = programs.get(&key) {
                Ok(*program)
            } else {
                let (vertex_shader, fragment_shader) = block_on(future::try_join(
                    load_asset_from_vfs::<ShaderAsset>(&key.0),
                    load_asset_from_vfs::<ShaderAsset>(&key.1),
                ))?;
                let program = self.renderer.create_program(
                    layout,
                    &[
                        ShaderDesc::vertex(&vertex_shader.bytecode),
                        ShaderDesc::fragment(&fragment_shader.bytecode),
                    ],
                )?;
                programs.insert(key, program);
                Ok(program)
            }
        }
    }

    pub fn get_or_create_raster_pipeline(
        &self,
        vertex_shader: &str,
        fragment_shader: &str,
        pass_layout: &'static RenderPassLayout<'static>,
        descriptors_layout: &'static [DescriptorSetLayoutDesc<'static>],
        input_layout: &'static [InputVertexStreamLayout<'static>],
        desc: RasterPipelineCreateDesc,
        specialization: Option<&[(u32, u32)]>,
    ) -> Result<RasterPipelineHandle, Error> {
        let program =
            self.get_or_create_raster_program(descriptors_layout, vertex_shader, fragment_shader)?;
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
                let pipeline = self.renderer.create_raster_pipeline(RasterPipelineDesc {
                    program,
                    pass_layout,
                    input_layout,
                    specialization: specialization.unwrap_or_default().to_vec(),
                    desc,
                });
                pipelines.insert(key, pipeline);
                Ok(pipeline)
            }
        }
    }
}
