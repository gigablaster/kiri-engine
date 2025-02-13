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
    process::{Command, Stdio},
};

use kiri_assets::{ShaderAsset, ShaderAssetSource, ShaderType, SourceAssetPath};

use crate::ImportAsset;

impl ImportAsset<ShaderAsset> for ShaderAssetSource {
    fn import<I: crate::AssetPipelineContext>(self, _context: &I) -> io::Result<ShaderAsset> {
        let target = match self.ty {
            ShaderType::Vertex => "-fshader-stage=vertex",
            ShaderType::Fragment => "-fshader-stage=fragment",
            ShaderType::Compute => "-fshader-stage=compute",
        };
        let child = Command::new("glslc")
            .arg(target)
            .arg("--target-env=vulkan1.1")
            .arg("-I")
            .arg(self.path.parent())
            .arg("-o")
            .arg("-")
            .arg(self.path.full_source_path())
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
