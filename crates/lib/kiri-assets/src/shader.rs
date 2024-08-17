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

use std::{
    hash::{Hash, Hasher},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use shader_prepper::{IncludeProvider, ResolvedIncludePath};
use speedy::{Readable, Writable};

use crate::{
    get_absolute_asset_path, is_asset_changed, read_to_end, Asset, AssetSource, Error, ImportAsset,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct ShaderAssetSource {
    pub path: String,
    pub ty: ShaderType,
}

impl ShaderAssetSource {
    pub fn fragment(path: &str) -> Self {
        Self {
            path: path.to_owned(),
            ty: ShaderType::Fragment,
        }
    }

    pub fn vertex(path: &str) -> Self {
        Self {
            path: path.to_owned(),
            ty: ShaderType::Vertex,
        }
    }
}

impl AssetSource for ShaderAssetSource {
    fn reference(&self) -> kiri_vfs::AssetReference {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish().into()
    }

    fn changed(&self, last_update: std::time::SystemTime) -> bool {
        if is_asset_changed(&self.path, last_update) {
            return true;
        }
        if let Ok(result) = are_includes_changed(&self.path, last_update) {
            return result;
        }

        false
    }
}

fn are_includes_changed(path: &str, timestamp: std::time::SystemTime) -> Result<bool, Error> {
    Ok(
        shader_prepper::process_file(path, &mut ShaderIncludeProvider::default(), PathBuf::new())
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?
            .iter()
            .map(|chunk| is_asset_changed(&chunk.file, timestamp))
            .any(|x| x),
    )
}

#[derive(Debug, Readable, Writable)]
pub struct ShaderAsset {
    pub ty: ShaderType,
    pub bytecode: Vec<u8>,
}

impl ShaderType {
    pub fn target(&self) -> &str {
        match self {
            ShaderType::Vertex => "-fshader-stage=vertex",
            ShaderType::Fragment => "-fshader-stage=fragment",
        }
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
        let data = read_to_end(get_absolute_asset_path(&path.0)?)?;
        Ok(String::from_utf8_lossy(&data).into_owned())
    }
}

impl Asset for ShaderAsset {
    fn load(data: bytes::Bytes) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(&data)?)
    }

    fn save(&self) -> std::io::Result<bytes::Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

impl ImportAsset<ShaderAssetSource> for ShaderAsset {
    fn import(
        source: ShaderAssetSource,
        _context: &dyn crate::AssetImportContext,
    ) -> Result<Self, Error> {
        let code = shader_prepper::process_file(
            &source.path,
            &mut ShaderIncludeProvider::default(),
            PathBuf::new(),
        )
        .map_err(|err| Error::ProcessingFailed(err.to_string()))?
        .iter()
        .map(|chunk| chunk.source.clone())
        .collect::<Vec<_>>()
        .concat();

        let mut child = Command::new("glslc")
            .arg(source.ty.target())
            .arg("--target-env=vulkan1.1")
            .arg("-o")
            .arg("-")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|x| Error::ProcessingFailed(x.to_string()))?;
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(code.as_bytes())
            .unwrap();

        let result = child
            .wait_with_output()
            .map_err(|x| Error::ProcessingFailed(x.to_string()))?;
        if result.status.success() {
            Ok(ShaderAsset {
                ty: source.ty,
                bytecode: result.stdout,
            })
        } else {
            Err(Error::ProcessingFailed(
                String::from_utf8_lossy(&result.stderr).into(),
            ))
        }
    }
}
