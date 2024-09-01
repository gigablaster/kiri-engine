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

use std::{collections::HashMap, sync::Arc};

use bytes::Bytes;
use kiri_assets::{ShaderAssetSource, ShaderType};
use kiri_backend::{
    DescriptorSetLayoutDesc, InputVertexStreamLayout, RasterPipelineCreateDesc, RenderPassLayout,
    ShaderDesc,
};
use kiri_gfx::{PipelineHandle, ProgramHandle, Renderer};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{load_or_compile_asset, Error};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RasterPipelineDesc {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub render_pass: &'static RenderPassLayout<'static>,
    pub input_layout: &'static [InputVertexStreamLayout<'static>],
    pub descriptor_layout: &'static [DescriptorSetLayoutDesc<'static>],
    pub specialization: Vec<(u32, u32)>,
    pub desc: RasterPipelineCreateDesc,
}

impl RasterPipelineDesc {
    pub fn new(
        vertex_shader: &str,
        fragment_shader: &str,
        render_pass: &'static RenderPassLayout<'static>,
        input_layout: &'static [InputVertexStreamLayout<'static>],
        descriptor_layout: &'static [DescriptorSetLayoutDesc<'static>],
    ) -> Self {
        Self {
            vertex_shader: vertex_shader.into(),
            fragment_shader: fragment_shader.into(),
            render_pass,
            input_layout,
            specialization: Default::default(),
            desc: Default::default(),
            descriptor_layout,
        }
    }

    pub fn specialization(mut self, slot: u32, value: u32) -> Self {
        self.specialization.push((slot, value));
        self
    }

    pub fn desc(&mut self) -> &mut RasterPipelineCreateDesc {
        &mut self.desc
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProgramKey(String, String);

#[derive(Debug)]
pub struct PipelineCache {
    renderer: Arc<Renderer>,
    shaders: Mutex<HashMap<(String, ShaderType), Bytes>>,
    programs: Mutex<HashMap<ProgramKey, ProgramHandle>>,
    raster_pipelines: RwLock<HashMap<RasterPipelineDesc, PipelineHandle>>,
}

impl PipelineCache {
    pub fn new(renderer: &Arc<Renderer>) -> Arc<Self> {
        Arc::new(Self {
            renderer: renderer.clone(),
            shaders: Default::default(),
            programs: Default::default(),
            raster_pipelines: Default::default(),
        })
    }

    fn get_or_load_shader(&self, name: &str, ty: ShaderType) -> Result<Bytes, Error> {
        let mut shaders = self.shaders.lock();
        let key = (name.into(), ty);
        if let Some(shader) = shaders.get(&key) {
            Ok(shader.clone())
        } else {
            let shader = load_or_compile_asset(&ShaderAssetSource::new(name, ty))?;
            let bytecode: Bytes = shader.bytecode.into();
            shaders.insert(key, bytecode.clone());
            Ok(bytecode)
        }
    }

    fn get_or_load_program(
        &self,
        layout: &'static [DescriptorSetLayoutDesc<'static>],
        vertex_shader: &str,
        fragment_shader: &str,
    ) -> Result<ProgramHandle, Error> {
        let mut programs = self.programs.lock();
        let key = ProgramKey(vertex_shader.into(), fragment_shader.into());
        if let Some(program) = programs.get(&key) {
            Ok(*program)
        } else {
            let vertex_shader = self.get_or_load_shader(vertex_shader, ShaderType::Vertex)?;
            let fragment_shader = self.get_or_load_shader(fragment_shader, ShaderType::Fragment)?;
            let program = self.renderer.create_program(
                layout,
                &[
                    ShaderDesc::vertex(&vertex_shader),
                    ShaderDesc::fragment(&fragment_shader),
                ],
            )?;
            programs.insert(key, program);
            Ok(program)
        }
    }

    pub fn get_or_create_raster_pipeline(
        &self,
        desc: RasterPipelineDesc,
    ) -> Result<PipelineHandle, Error> {
        let pipelines = self.raster_pipelines.upgradable_read();
        if let Some(pipeline) = pipelines.get(&desc) {
            Ok(*pipeline)
        } else {
            let mut pipelines = RwLockUpgradableReadGuard::upgrade(pipelines);
            if let Some(pipeline) = pipelines.get(&desc) {
                Ok(*pipeline)
            } else {
                let program = self.get_or_load_program(
                    desc.descriptor_layout,
                    &desc.vertex_shader,
                    &desc.fragment_shader,
                )?;
                let pipeline = self.renderer.create_pipeline(
                    program,
                    desc.render_pass,
                    desc.input_layout,
                    &desc.specialization,
                    desc.desc,
                );
                pipelines.insert(desc, pipeline);
                Ok(pipeline)
            }
        }
    }
}
