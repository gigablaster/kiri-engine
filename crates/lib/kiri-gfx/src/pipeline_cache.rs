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

use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    path::{Path, PathBuf},
    sync::Arc,
    u32,
};

use kiri_assets::{load_or_compile_asset, ShaderAssetSource};
use kiri_backend::{
    ash::vk::{self},
    compile_raster_pipeline, load_or_create_pipeline_cache, save_pipeline_cache,
    DescriptorSetLayoutDesc, InputVertexStreamLayout, Pipeline, RasterPipelineCreateDesc,
    RasterProgram, RenderDevice, RenderPassLayout, ShaderDesc,
};
use kiri_common::{block_on, spawn};
use log::warn;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockUpgradableReadGuard};
use turbosloth::{async_trait, IntoLazy, Lazy, LazyCache, LazyWorker, RunContext};

use crate::Error;

#[derive(Debug, Clone)]
pub struct CompileRasterProgram {
    device: Arc<RenderDevice>,
    layout: &'static [DescriptorSetLayoutDesc<'static>],
    vertex_shader: String,
    fragment_shader: String,
}

impl Hash for CompileRasterProgram {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vertex_shader.hash(state);
        self.fragment_shader.hash(state);
    }
}

#[async_trait]
impl LazyWorker for CompileRasterProgram {
    type Output = Result<RasterProgram, Error>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        let (vertex_shader, fragment_shader) = futures::future::try_join(
            spawn(load_or_compile_asset(ShaderAssetSource::vertex(
                self.vertex_shader,
            ))),
            spawn(load_or_compile_asset(ShaderAssetSource::fragment(
                self.fragment_shader,
            ))),
        )
        .await?;
        Ok(RasterProgram::new(
            &self.device,
            self.layout,
            &[
                ShaderDesc::vertex(&vertex_shader.bytecode),
                ShaderDesc::fragment(&fragment_shader.bytecode),
            ],
        )?)
    }
}

impl CompileRasterProgram {
    pub fn new(
        device: &Arc<RenderDevice>,
        layout: &'static [DescriptorSetLayoutDesc<'static>],
        vertex_shader: String,
        fragment_shader: String,
    ) -> Self {
        Self {
            device: device.clone(),
            layout,
            vertex_shader,
            fragment_shader,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RasterPipelineHandle(u32);

impl From<RasterPipelineHandle> for u32 {
    fn from(value: RasterPipelineHandle) -> Self {
        value.0
    }
}

impl From<u32> for RasterPipelineHandle {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Default for RasterPipelineHandle {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

impl RasterPipelineHandle {
    pub fn valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineDesc {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub pass_layout: &'static RenderPassLayout<'static>,
    pub descriptors_layout: &'static [DescriptorSetLayoutDesc<'static>],
    pub input_layout: &'static [InputVertexStreamLayout<'static>],
    pub specialization: Vec<(u32, u32)>,
    pub desc: RasterPipelineCreateDesc,
}

impl PipelineDesc {
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

    pub fn specialization(mut self, slot: u32, value: u32) -> Self {
        self.specialization.push((slot, value));
        self
    }

    pub fn pipeline_desc(mut self, value: RasterPipelineCreateDesc) -> Self {
        self.desc = value;
        self
    }
}

#[derive(Clone)]
pub struct CompileRasterPipeline {
    device: Arc<RenderDevice>,
    pipeline_cache: vk::PipelineCache,
    program: Lazy<RasterProgram>,
    pass_layout: &'static RenderPassLayout<'static>,
    input_layout: &'static [InputVertexStreamLayout<'static>],
    specialization: Vec<(u32, u32)>,
    desc: RasterPipelineCreateDesc,
}

impl Hash for CompileRasterPipeline {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.program.hash(state);
        self.pass_layout.hash(state);
        self.specialization.hash(state);
        self.desc.hash(state);
    }
}

#[async_trait]
impl LazyWorker for CompileRasterPipeline {
    type Output = Result<Pipeline, Error>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let program = self.program.eval(&ctx).await?;
        Ok(compile_raster_pipeline(
            &self.device,
            self.pipeline_cache,
            &program,
            self.pass_layout,
            self.input_layout,
            &self.specialization,
            self.desc,
            None,
        )?)
    }
}

impl CompileRasterPipeline {
    pub fn new(
        device: &Arc<RenderDevice>,
        pipeline_cache: vk::PipelineCache,
        program: Lazy<RasterProgram>,
        desc: &PipelineDesc,
    ) -> Self {
        Self {
            device: device.clone(),
            pipeline_cache,
            program,
            pass_layout: desc.pass_layout,
            input_layout: desc.input_layout,
            specialization: desc.specialization.clone(),
            desc: desc.desc,
        }
    }
}

struct PipelineCacheEntry {
    compile: Lazy<Pipeline>,
    pipeline: Option<Arc<Pipeline>>,
}

impl Debug for PipelineCacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCacheEntry")
            .field("compile", &self.compile.debug_name)
            .field("pipeline", &self.pipeline)
            .finish()
    }
}
pub struct PipelineCache {
    device: Arc<RenderDevice>,
    cache: Arc<LazyCache>,
    pipeline_cache: vk::PipelineCache,
    pipeline_cache_path: Option<PathBuf>,
    raster_pipelines: Mutex<Vec<PipelineCacheEntry>>,
    raster_handle_to_pipeline: RwLock<HashMap<PipelineDesc, RasterPipelineHandle>>,
}

unsafe impl Send for PipelineCache {}
unsafe impl Sync for PipelineCache {}

impl Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("device", &self.device)
            .field("pipeline_cache", &self.pipeline_cache)
            .field("pipeline_cache_path", &self.pipeline_cache_path)
            .field("raster_pipelines", &self.raster_pipelines)
            .field("raster_handle_to_pipeline", &self.raster_handle_to_pipeline)
            .finish()
    }
}

pub(super) struct PipelineResolver<'a> {
    cache: Arc<LazyCache>,
    raster_pipelines: MutexGuard<'a, Vec<PipelineCacheEntry>>,
}

impl Debug for PipelineResolver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCacheResolver")
            .field("raster_pipelines", &self.raster_pipelines)
            .finish()
    }
}

impl PipelineResolver<'_> {
    pub fn resolve_raster_pipeline(
        &self,
        handle: RasterPipelineHandle,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
        let pipeline = self
            .raster_pipelines
            .get(handle.0 as usize)
            .ok_or(Error::InvalidRasterPipeline(handle))?;
        if let Some(pipeline) = &pipeline.pipeline {
            Ok((pipeline.pipeline, pipeline.pipeline_layout))
        } else {
            let compiled = block_on(pipeline.compile.eval(&self.cache))?;
            Ok((compiled.pipeline, compiled.pipeline_layout))
        }
    }
}

impl PipelineCache {
    pub fn new<P: AsRef<Path>>(device: Arc<RenderDevice>, cache_path: Option<P>) -> Self {
        let pipeline_cache_path = cache_path.map(|path| path.as_ref().to_path_buf());
        let pipeline_cache = pipeline_cache_path
            .clone()
            .map(|path| {
                load_or_create_pipeline_cache(&device, path).unwrap_or(vk::PipelineCache::null())
            })
            .unwrap_or(vk::PipelineCache::null());

        Self {
            pipeline_cache,
            pipeline_cache_path,
            cache: LazyCache::create(),
            raster_pipelines: Default::default(),
            raster_handle_to_pipeline: Default::default(),
            device,
        }
    }

    pub fn get_or_create_raster_pipeline(&self, desc: PipelineDesc) -> RasterPipelineHandle {
        let handles = self.raster_handle_to_pipeline.upgradable_read();
        if let Some(handle) = handles.get(&desc) {
            *handle
        } else {
            let mut handles = RwLockUpgradableReadGuard::upgrade(handles);
            if let Some(handle) = handles.get(&desc) {
                *handle
            } else {
                let mut pipelines = self.raster_pipelines.lock();
                let handle = RasterPipelineHandle(pipelines.len() as u32);
                let program = CompileRasterProgram::new(
                    &self.device,
                    desc.descriptors_layout,
                    desc.vertex_shader.clone(),
                    desc.fragment_shader.clone(),
                )
                .into_lazy();
                let compile =
                    CompileRasterPipeline::new(&self.device, self.pipeline_cache, program, &desc)
                        .into_lazy();
                pipelines.push(PipelineCacheEntry {
                    compile,
                    pipeline: None,
                });
                handles.insert(desc, handle);
                handle
            }
        }
    }

    pub(super) fn resolve(&self) -> PipelineResolver {
        PipelineResolver {
            cache: self.cache.clone(),
            raster_pipelines: self.raster_pipelines.lock(),
        }
    }
    pub async fn compile_pending_pipelines(&self) -> Result<(), Error> {
        puffin::profile_function!();
        let mut pipelines = self.raster_pipelines.lock();
        futures::future::try_join_all(pipelines.iter_mut().filter_map(|pipeline| {
            if !pipeline.compile.is_up_to_date() || pipeline.pipeline.is_none() {
                pipeline.pipeline = None;
                Some(spawn(pipeline.compile.eval(&self.cache)))
            } else {
                None
            }
        }))
        .await?;
        for pipeline in pipelines.iter_mut() {
            if pipeline.pipeline.is_none() {
                pipeline.pipeline = Some(block_on(pipeline.compile.eval(&self.cache))?);
            }
        }
        Ok(())
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        if let Some(path) = &self.pipeline_cache_path {
            if self.pipeline_cache != vk::PipelineCache::null() {
                save_pipeline_cache(&self.device, self.pipeline_cache, &path)
                    .map_err(|x| {
                        warn!("Failed to safe pipeline cache: {}", x);
                    })
                    .ok();
                unsafe {
                    self.device
                        .raw
                        .destroy_pipeline_cache(self.pipeline_cache, None);
                }
            }
        }
    }
}
