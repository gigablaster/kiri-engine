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

use std::{collections::HashMap, future::IntoFuture, sync::Arc};

use crate::Renderer;
use bytes::Bytes;
use futures::{FutureExt, TryFutureExt};
use kiri_assets::{LoadOrCompileAsset, ShaderAsset, ShaderAssetSource, ShaderType};
use kiri_backend::{
    ash::vk, compile_raster_pipeline, RasterPipeline, RasterPipelineCreateDesc, RasterProgram,
    RenderDevice, RenderPassLayout, ShaderDesc,
};
use kiri_common::{spawn, spawn_io};
use log::debug;
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};
use turbosloth::{
    async_trait, lazy::LazyIdentity, IntoLazy, Lazy, LazyCache, LazyWorker, RunContext,
};

use crate::Error;

pub struct CompileRasterProgram {
    device: Arc<RenderDevice>,
    shaders: Vec<Lazy<ShaderAsset>>,
}

#[async_trait]
impl LazyWorker for CompileRasterProgram {
    type Output = Result<Arc<RasterProgram>, Error>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let shaders =
            futures::future::try_join_all(self.shaders.iter().cloned().map(|foo| foo.eval(&ctx)))
                .await?
                .iter()
                .cloned()
                .collect::<Vec<_>>();
        let descs = shaders
            .iter()
            .map(|shader| ShaderDesc::new(shader.ty.into(), &shader.bytecode))
            .collect::<Vec<_>>();

        Ok(Arc::new(RasterProgram::new(&self.device, &descs)?))
    }
}

impl CompileRasterProgram {
    pub fn new(device: &Arc<RenderDevice>, shaders: Vec<Lazy<ShaderAsset>>) -> Self {
        Self {
            device: device.clone(),
            shaders,
        }
    }
}
pub struct RasterPipelineHandle(usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RasterPipelineDesc {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub render_pass: &'static RenderPassLayout<'static>,
    pub specialization: Vec<(u32, u32)>,
    pub desc: RasterPipelineCreateDesc,
}

impl RasterPipelineDesc {
    pub fn new(
        vertex_shader: &str,
        fragment_shader: &str,
        render_pass: &'static RenderPassLayout<'static>,
    ) -> Self {
        Self {
            vertex_shader: vertex_shader.into(),
            fragment_shader: fragment_shader.into(),
            render_pass,
            specialization: Default::default(),
            desc: Default::default(),
        }
    }

    pub fn specialization(mut self, slot: u32, value: u32) -> Self {
        self.specialization.push((slot, value));
        self
    }

    pub fn pipeline_desc(mut self, value: RasterPipelineCreateDesc) -> Self {
        self.desc = value;
        self
    }
}

pub struct CompileRasterPipeline {
    renderer: Arc<Renderer>,
    program: Lazy<Arc<RasterProgram>>,
    pass_layout: &'static RenderPassLayout<'static>,
    specialization: Vec<(u32, u32)>,
    desc: RasterPipelineCreateDesc,
}

#[async_trait]
impl LazyWorker for CompileRasterPipeline {
    type Output = Result<Arc<RasterPipeline>, Error>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let program = self.program.eval(&ctx).await?;
        Ok(Arc::new(compile_raster_pipeline(
            &self.renderer.device,
            self.renderer.pipeline_cache,
            &program,
            self.pass_layout,
            &self.specialization,
            self.desc,
            None,
        )?))
    }
}

struct RasterPipelineCacheEntry {
    program: Lazy<Arc<RasterProgram>>,
    pipeline: Option<Arc<RasterPipeline>>,
}

pub struct PipelineCache {
    pub renderer: Arc<Renderer>,
    cache: Arc<LazyCache>,
    pipelines: RwLock<Vec<RasterPipelineCacheEntry>>,
    programs: RwLock<HashMap<(String, String), Arc<RasterProgram>>>,
    handle_to_pipeline: RwLock<HashMap<RasterPipelineDesc, RasterPipelineHandle>>,
}

impl PipelineCache {
    pub fn new(renderer: &Arc<Renderer>) -> Arc<Self> {
        Arc::new(Self {
            renderer: renderer.clone(),
            cache: LazyCache::create(),
            programs: Default::default(),
            pipelines: Default::default(),
            handle_to_pipeline: Default::default(),
        })
    }

    async fn get_or_compile_raster_program(
        &self,
        vertex_shader: &str,
        fragment_shader: &str,
    ) -> Result<Arc<RasterProgram>, Error> {
        let (vertex_shader, fragment_shader) = futures::future::try_join(
            spawn_io(
                LoadOrCompileAsset::new(ShaderAssetSource::vertex(vertex_shader))
                    .into_lazy()
                    .eval(&self.cache),
            ),
            spawn_io(
                LoadOrCompileAsset::new(ShaderAssetSource::fragment(fragment_shader))
                    .into_lazy()
                    .eval(&self.cache),
            ),
        )
        .await?;
        Ok(Arc::new(RasterProgram::new(
            &self.renderer.device,
            &[
                ShaderDesc::vertex(&vertex_shader.bytecode),
                ShaderDesc::fragment(&fragment_shader.bytecode),
            ],
        )?))
    }
}
