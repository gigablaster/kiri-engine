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

mod effect;
mod glsl;
mod gltf;
mod image;
mod mesh_builder;

pub use effect::*;
pub use gltf::*;
pub use image::*;

use std::{
    fs,
    io::{self, Read},
    path::Path,
    time::SystemTime,
};

use kiri_assets::{Asset, CompiledAssetPath, SourceAssetPath};

pub trait AssetSource: Clone {
    fn source(&self) -> &SourceAssetPath;
    fn changed(&self, timestamp: SystemTime) -> bool;
}

pub trait ImportAsset<T: Asset> {
    fn import<I: AssetPipelineContext>(self, context: &I) -> io::Result<T>;
}

pub(crate) fn read_to_end<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path.as_ref())?;
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
}

pub trait AssetPipelineContext {
    fn import_image(&self, image: ImageSource) -> CompiledAssetPath;
}
