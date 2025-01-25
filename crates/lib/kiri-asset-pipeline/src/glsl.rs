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
    io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use kiri_assets::{ShaderAsset, ShaderType, SourceAssetPath};
use shader_prepper::{IncludeProvider, ResolvedIncludePath};

use crate::{read_to_end, AssetPipelineContext, AssetSource, ImportAsset};

#[derive(Debug)]
pub struct GlslShaderSource {
    source: SourceAssetPath,
    ty: ShaderType,
}

impl AssetSource for GlslShaderSource {
    fn source(&self) -> &SourceAssetPath {
        &self.source
    }

    fn changed(&self, timestamp: std::time::SystemTime) -> bool {
        if self.source.changed(timestamp) {
            return true;
        }

        if let Ok(result) = Self::are_includes_changed(self.source.as_ref(), timestamp) {
            return result;
        }

        return false;
    }
}

impl GlslShaderSource {
    fn are_includes_changed(path: &str, timestamp: std::time::SystemTime) -> io::Result<bool> {
        Ok(shader_prepper::process_file(
            path,
            &mut ShaderIncludeProvider::default(),
            PathBuf::new(),
        )
        .map_err(|err| io::Error::other(format!("Shader processing failed: {}", err)))?
        .iter()
        .map(|chunk| SourceAssetPath::new(&chunk.file).changed(timestamp))
        .any(|x| x))
    }
}

#[derive(Debug, Default)]
struct ShaderIncludeProvider {}

impl IncludeProvider for ShaderIncludeProvider {
    type IncludeContext = PathBuf;

    fn resolve_path(
        &self,
        path: &str,
        context: &Self::IncludeContext,
    ) -> Result<
        shader_prepper::ResolvedInclude<Self::IncludeContext>,
        shader_prepper::BoxedIncludeProviderError,
    > {
        let path = PathBuf::from(path);
        let full = context.join(path);
        let root = full.parent().unwrap_or(Path::new("")).to_owned();
        Ok(shader_prepper::ResolvedInclude {
            resolved_path: ResolvedIncludePath(full.to_str().unwrap_or_default().into()),
            context: root,
        })
    }

    fn get_include(
        &mut self,
        path: &shader_prepper::ResolvedIncludePath,
    ) -> Result<String, shader_prepper::BoxedIncludeProviderError> {
        let data = read_to_end(&SourceAssetPath::new(&path.0))?;
        Ok(String::from_utf8_lossy(&data).into_owned())
    }
}

impl ImportAsset<ShaderAsset> for GlslShaderSource {
    fn import(self, _context: &impl AssetPipelineContext) -> io::Result<ShaderAsset> {
        let target = match self.ty {
            ShaderType::Vertex => "-fshader-stage=vertex",
            ShaderType::Fragment => "-fshader-stage=fragment",
        };
        let child = Command::new("glslc")
            .arg(target)
            .arg("--target-env=vulkan1.1")
            .arg("-I")
            .arg(self.source.parent())
            .arg("-o")
            .arg("-")
            .arg(self.source.full_source_path())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|x| io::Error::other(format!("Failed to spawn shader compiler: {}", x)))?;

        let result = child
            .wait_with_output()
            .map_err(|x| io::Error::other(format!("Shader compilation failed: {}", x)))?;
        if result.status.success() {
            Ok(ShaderAsset {
                ty: self.ty,
                bytecode: result.stdout,
            })
        } else {
            Err(io::Error::other(
                String::from_utf8_lossy(&result.stderr).to_string(),
            ))
        }
    }
}
