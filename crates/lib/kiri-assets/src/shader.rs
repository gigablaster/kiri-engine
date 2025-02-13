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
    hash::{Hash, Hasher},
    io::{self, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

use kiri_backend::ash::vk;
use shader_prepper::{IncludeProvider, ResolvedIncludePath};
use siphasher::sip::SipHasher;
use speedy::{Readable, Writable};

use crate::{read_to_end, Asset, AssetSource, SourceAssetPath};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Debug, Readable, Writable)]
pub struct ShaderAsset {
    pub ty: ShaderType,
    pub bytecode: Vec<u8>,
}

impl From<ShaderType> for vk::ShaderStageFlags {
    fn from(value: ShaderType) -> Self {
        match value {
            ShaderType::Vertex => Self::VERTEX,
            ShaderType::Fragment => Self::FRAGMENT,
            ShaderType::Compute => Self::COMPUTE,
        }
    }
}

impl Asset for ShaderAsset {
    const TYPE: uuid::Uuid = uuid::uuid!("d6fb342d-938f-4ac0-9253-466f37725244");

    fn serialize<W: Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderAssetSource {
    pub path: SourceAssetPath,
    pub ty: ShaderType,
}

impl ShaderAssetSource {
    pub fn vertex<S: AsRef<str>>(path: S) -> Self {
        Self {
            path: path.as_ref().into(),
            ty: ShaderType::Vertex,
        }
    }

    pub fn fragment<S: AsRef<str>>(path: S) -> Self {
        Self {
            path: path.as_ref().into(),
            ty: ShaderType::Fragment,
        }
    }
}

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

impl AssetSource for ShaderAssetSource {
    fn changed(&self, timestamp: std::time::SystemTime) -> bool {
        is_glsl_changed(&self.path, timestamp)
    }

    fn reference(&self) -> kiri_vfs::AssetReference {
        let mut hasher = SipHasher::default();
        self.path.compiled().unwrap().hash(&mut hasher);
        self.ty.hash(&mut hasher);
        hasher.finish().into()
    }
}
