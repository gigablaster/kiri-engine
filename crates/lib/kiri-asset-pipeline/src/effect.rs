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

use std::{
    collections::HashMap,
    io::{self},
    path::Path,
};

use kiri_assets::{EffectAsset, ShaderType, SourceAssetPath, Technique};
use serde::{Deserialize, Serialize};

use crate::{
    glsl::{compile_glsl, is_glsl_changed},
    read_to_end, AssetPipelineContext, AssetSource, ImportAsset,
};

#[derive(Debug, Serialize, Deserialize)]
struct Effect {
    pub vertex: String,
    pub fragment: String,
    pub techniques: HashMap<String, Technique>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RenderEffectSource(SourceAssetPath);

impl AssetSource for RenderEffectSource {
    fn source(&self) -> &SourceAssetPath {
        &self.0
    }

    fn changed(&self, timestamp: std::time::SystemTime) -> bool {
        if self.0.changed(timestamp) {
            return true;
        }
        if let Ok(desc) = self.load_desc() {
            return is_glsl_changed(desc.vertex, timestamp)
                || is_glsl_changed(desc.fragment, timestamp);
        }
        true
    }
}

impl RenderEffectSource {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self(SourceAssetPath::new(path))
    }

    fn load_desc(&self) -> io::Result<Effect> {
        Ok(
            serde_json::from_slice::<Effect>(&read_to_end(self.0.full_source_path())?)
                .map_err(io::Error::other)?,
        )
    }
}

impl ImportAsset<EffectAsset> for RenderEffectSource {
    fn import<I: AssetPipelineContext>(self, _context: &I) -> io::Result<EffectAsset> {
        let desc = self.load_desc()?;
        Ok(EffectAsset {
            vertex_shader: compile_glsl(desc.vertex, ShaderType::Vertex)?,
            fragment_shader: compile_glsl(desc.fragment, ShaderType::Fragment)?,
            techniques: desc.techniques,
        })
    }
}
