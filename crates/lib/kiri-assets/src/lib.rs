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
mod gltf;
mod image;
mod mesh_builder;
mod shader;

pub const ROOT_DATA_PATH: &str = "assets";
pub const ASSET_CACHE_PATH: &str = ".cache";

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO failed: {0:?}")]
    Io(io::Error),
    #[error("Import failed: {0}")]
    ImportFailed(String),
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub trait AssetSource: Send + Sync {
    fn reference(&self) -> AssetReference;
    fn changed(&self, last_update: SystemTime) -> bool;
}

pub trait Asset: Sized + Send + Sync {
    fn load(data: Bytes) -> io::Result<Self>;
    fn save(&self) -> io::Result<Bytes>;
}

pub trait ImportAsset<T: AssetSource>: Asset + Send + Sync {
    fn import(source: T, context: &dyn AssetImportContext) -> Result<Self, Error>;
}

pub trait AssetImportContext: Send + Sync {
    fn import_image(&self, source: ImageAssetSource) -> AssetReference;
    fn import_static_mesh(&self, source: GltfMeshSource, data: MeshAssetBuilder) -> AssetReference;
}

use std::{
    env, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    time::SystemTime,
};

use bytes::Bytes;
pub use gltf::*;
pub use image::*;
pub use kiri_vfs::AssetReference;
pub use mesh_builder::*;
pub use shader::*;

use thiserror::Error;

pub(crate) fn read_to_end<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path.as_ref())?;
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
}

pub fn get_relative_asset_path<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let root = env::current_dir()?.canonicalize()?.join(ROOT_DATA_PATH);
    // Is this path relative to data folder? Check this option.
    let path = if !path.as_ref().exists() {
        root.join(path)
    } else {
        path.as_ref().into()
    };
    let path = path.canonicalize()?;

    Ok(path.strip_prefix(root).unwrap().into())
}

pub fn get_absolute_asset_path<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let root = env::current_dir()?.canonicalize()?.join(ROOT_DATA_PATH);
    Ok(root.join(get_relative_asset_path(path.as_ref())?))
}

pub fn get_cached_asset_path(asset: AssetReference) -> PathBuf {
    Path::new(ASSET_CACHE_PATH).join(format!("{}.bin", asset))
}

pub(crate) fn is_asset_changed<P: AsRef<Path>>(path: P, timestamp: SystemTime) -> bool {
    if let Ok(path) = get_absolute_asset_path(path) {
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                return modified > timestamp;
            }
            if let Ok(created) = metadata.created() {
                return created > timestamp;
            }
        }
    }
    false
}
