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
    time::SystemTime,
};

use kiri_assets::{ShaderType, SourceAssetPath};
use shader_prepper::{IncludeProvider, ResolvedIncludePath};

use crate::read_to_end;

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
        let data = read_to_end(SourceAssetPath::new(&path.0))?;
        Ok(String::from_utf8_lossy(&data).into_owned())
    }
}

pub fn is_glsl_changed<P: AsRef<Path>>(path: P, timestamp: SystemTime) -> bool {
    let source = SourceAssetPath::new(path);
    if source.changed(timestamp) {
        return true;
    }

    if let Ok(result) = are_includes_changed(source.as_ref(), timestamp) {
        return result;
    }

    false
}

#[derive(Debug, Default)]
struct ShaderIncludeProvider {}

fn are_includes_changed(path: &str, timestamp: std::time::SystemTime) -> io::Result<bool> {
    Ok(
        shader_prepper::process_file(path, &mut ShaderIncludeProvider::default(), PathBuf::new())
            .map_err(|err| io::Error::other(format!("Shader processing failed: {}", err)))?
            .iter()
            .any(|x| SourceAssetPath::new(&x.file).changed(timestamp)),
    )
}

pub fn compile_glsl<P: AsRef<Path>>(path: P, ty: ShaderType) -> io::Result<Vec<u8>> {
    let target = match ty {
        ShaderType::Vertex => "-fshader-stage=vertex",
        ShaderType::Fragment => "-fshader-stage=fragment",
        ShaderType::Compute => "-fshader-stage=compute",
    };
    let path = SourceAssetPath::new(path);
    let child = Command::new("glslc")
        .arg(target)
        .arg("--target-env=vulkan1.1")
        .arg("-I")
        .arg(path.parent())
        .arg("-o")
        .arg("-")
        .arg(path.full_source_path())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|x| io::Error::other(format!("Failed to spawn shader compiler: {}", x)))?;

    let result = child
        .wait_with_output()
        .map_err(|x| io::Error::other(format!("Shader compilation failed: {}", x)))?;
    if result.status.success() {
        Ok(result.stdout)
    } else {
        Err(io::Error::other(
            String::from_utf8_lossy(&result.stderr).to_string(),
        ))
    }
}
