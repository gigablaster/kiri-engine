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
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use kiri_backend::ash::vk;
use speedy::{Readable, Writable};

use crate::Asset;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

// impl ShaderAssetSource {
//     pub fn new(path: &str, ty: ShaderType) -> Self {
//         Self {
//             path: path.into(),
//             ty,
//         }
//     }

//     pub fn fragment(path: String) -> Self {
//         Self {
//             path: path,
//             ty: ShaderType::Fragment,
//         }
//     }

//     pub fn vertex(path: String) -> Self {
//         Self {
//             path: path,
//             ty: ShaderType::Vertex,
//         }
//     }
// }

// impl AssetSource for ShaderAssetSource {
//     fn reference(&self) -> AssetReference {
//         AssetReference::new(self)
//     }

//     fn changed(&self, last_update: std::time::SystemTime) -> bool {
//         if is_asset_changed(&self.path, last_update) {
//             return true;
//         }
//         if let Ok(result) = are_includes_changed(&self.path, last_update) {
//             return result;
//         }

//         false
//     }
// }

// fn are_includes_changed(path: &str, timestamp: std::time::SystemTime) -> io::Result<bool> {
//     Ok(
//         shader_prepper::process_file(path, &mut ShaderIncludeProvider::default(), PathBuf::new())
//             .map_err(|err| io::Error::other(format!("Shader processing failed: {}", err)))?
//             .iter()
//             .map(|chunk| is_asset_changed(&chunk.file, timestamp))
//             .any(|x| x),
//     )
// }

#[derive(Debug, Readable, Writable)]
pub struct ShaderAsset {
    pub ty: ShaderType,
    pub bytecode: Vec<u8>,
}

// impl ShaderType {
//     pub fn target(&self) -> &str {
//         match self {
//             ShaderType::Vertex => "-fshader-stage=vertex",
//             ShaderType::Fragment => "-fshader-stage=fragment",
//         }
//     }
// }

impl From<ShaderType> for vk::ShaderStageFlags {
    fn from(value: ShaderType) -> Self {
        match value {
            ShaderType::Vertex => Self::VERTEX,
            ShaderType::Fragment => Self::FRAGMENT,
        }
    }
}

// #[derive(Debug, Default)]
// struct ShaderIncludeProvider {}

// impl IncludeProvider for ShaderIncludeProvider {
//     type IncludeContext = PathBuf;

//     fn resolve_path(
//         &self,
//         path: &str,
//         context: &Self::IncludeContext,
//     ) -> Result<
//         shader_prepper::ResolvedInclude<Self::IncludeContext>,
//         shader_prepper::BoxedIncludeProviderError,
//     > {
//         let path = PathBuf::from(path);
//         let full = context.join(path);
//         let root = full.parent().unwrap_or(Path::new("")).to_owned();
//         Ok(shader_prepper::ResolvedInclude {
//             resolved_path: ResolvedIncludePath(full.to_str().unwrap_or_default().into()),
//             context: root,
//         })
//     }

//     fn get_include(
//         &mut self,
//         path: &shader_prepper::ResolvedIncludePath,
//     ) -> Result<String, shader_prepper::BoxedIncludeProviderError> {
//         let data = read_to_end(get_full_source_asset_path(&path.0)?)?;
//         Ok(String::from_utf8_lossy(&data).into_owned())
//     }
// }

impl Asset for ShaderAsset {
    const TYPE: uuid::Uuid = uuid::uuid!("d6fb342d-938f-4ac0-9253-466f37725244");

    fn serialize<W: Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}

// impl ImportAsset<ShaderAsset> for ShaderAssetSource {
//     fn import(&self) -> io::Result<ShaderAsset> {
//         let child = Command::new("glslc")
//             .arg(self.ty.target())
//             .arg("--target-env=vulkan1.3")
//             .arg("-I")
//             .arg(SOURCE_ASSETS_PATH)
//             .arg("-o")
//             .arg("-")
//             .arg(Path::new(SOURCE_ASSETS_PATH).join(&self.path))
//             .stdout(Stdio::piped())
//             .spawn()
//             .map_err(|x| io::Error::other(format!("Failed to spawn shader compiler: {}", x)))?;

//         let result = child
//             .wait_with_output()
//             .map_err(|x| io::Error::other(format!("Shader compilation failed: {}", x)))?;
//         if result.status.success() {
//             Ok(ShaderAsset {
//                 ty: self.ty,
//                 bytecode: result.stdout,
//             })
//         } else {
//             Err(io::Error::other(
//                 String::from_utf8_lossy(&result.stderr).to_string(),
//             ))
//         }
//     }
// }
