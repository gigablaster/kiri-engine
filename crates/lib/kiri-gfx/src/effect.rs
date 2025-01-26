// Copyright (C) 2025 gigablaster

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

use std::collections::HashMap;

use kiri_assets::EffectAsset;
use kiri_backend::{DescriptorSetLayoutDesc, RasterPipelineCreateDesc, ShaderDesc};

use crate::{Error, ProgramHandle, Renderer};

#[derive(Debug)]
pub struct RenderTechinque {
    pub desc: RasterPipelineCreateDesc,
    pub spec: Vec<(u32, u32)>,
}

#[derive(Debug)]
pub struct RenderEffect {
    pub program: ProgramHandle,
    pub descriptor_layout: &'static [DescriptorSetLayoutDesc<'static>],
    techinques: HashMap<String, RenderTechinque>,
}

impl RenderEffect {
    pub fn new(
        renderer: &Renderer,
        descriptor_layout: &'static [DescriptorSetLayoutDesc<'static>],
        asset: EffectAsset,
    ) -> Result<Self, Error> {
        let program = renderer.create_program(
            descriptor_layout,
            &[
                ShaderDesc::vertex(&asset.vertex_shader),
                ShaderDesc::fragment(&asset.fragment_shader),
            ],
        )?;
        let techinques = asset
            .techniques
            .into_iter()
            .map(|(name, techinque)| {
                (
                    name,
                    RenderTechinque {
                        desc: techinque.desc.into(),
                        spec: techinque
                            .spec
                            .unwrap_or_default()
                            .into_iter()
                            .collect::<Vec<_>>(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        Ok(Self {
            program,
            descriptor_layout,
            techinques,
        })
    }

    pub fn techinque(&self, name: &str) -> Option<&RenderTechinque> {
        self.techinques.get(name)
    }
}
