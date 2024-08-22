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

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO failed: {0:?}")]
    Io(#[from] io::Error),
    #[error("Import failed: {0}")]
    ImportFailed(String),
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),
}

pub trait AssetSource: Send + Sync {
    fn reference(&self) -> AssetReference;
    fn changed(&self, last_update: SystemTime) -> bool;
}

pub trait Asset: Sized + Send + Sync {
    const TYPE: Uuid;

    fn deserialize<R: Read>(r: R) -> io::Result<Self>;
    fn serialize<W: Write>(&self, w: W) -> io::Result<()>;
}

const MAGICK: [u8; 4] = [b'K', b'R', b'A', b'S'];
const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AssetHeader {
    magic: [u8; 4],
    version: u32,
    ty: Uuid,
}

impl AssetHeader {
    fn new<T: Asset>() -> Self {
        Self {
            magic: MAGICK,
            version: VERSION,
            ty: T::TYPE,
        }
    }

    pub fn is_valid<T: Asset>(&self) -> bool {
        self.magic == MAGICK && self.version == VERSION && self.ty == T::TYPE
    }
}

impl<'a, C: Context> Readable<'a, C> for AssetHeader {
    fn read_from<R: speedy::Reader<'a, C>>(
        reader: &mut R,
    ) -> Result<Self, <C as speedy::Context>::Error> {
        Ok(Self {
            magic: reader.read_value()?,
            version: reader.read_u32()?,
            ty: Uuid::from_u128(reader.read_u128()?),
        })
    }
}

impl<C: Context> Writable<C> for AssetHeader {
    fn write_to<T: ?Sized + speedy::Writer<C>>(
        &self,
        writer: &mut T,
    ) -> Result<(), <C as Context>::Error> {
        writer.write_value(&self.magic)?;
        writer.write_u32(self.version)?;
        writer.write_u128(self.ty.as_u128())?;
        Ok(())
    }
}

pub fn load_asset<T: Asset, R: Read>(r: R) -> io::Result<T> {
    let mut reader = BufReader::new(r);
    let header = AssetHeader::read_from_stream_buffered(reader.by_ref())?;
    if !header.is_valid::<T>() {
        return Err(io::Error::other("Asset header isn't valid"));
    }
    T::deserialize(&mut reader)
}

pub fn save_asset<T: Asset, W: Write>(w: W, asset: T) -> io::Result<()> {
    let mut w = w;
    AssetHeader::new::<T>().write_to_stream(w.by_ref())?;
    asset.serialize(w)
}

pub trait ImportAsset<T: Asset>: AssetSource + Send + Sync {
    fn import(&self) -> Result<T, Error>;
}

use std::{
    env, fs,
    io::{self, BufReader, Read, Write},
    path::{self, Path, PathBuf},
    time::SystemTime,
};

pub use gltf::*;
pub use image::*;
pub use kiri_vfs::AssetReference;
use kiri_vfs::{ROOT_COMPILED_ASSETS_PATH, ROOT_SOURCE_ASSETS_PATH};
use mesh_builder::*;
pub use shader::*;

use speedy::{Context, Readable, Writable};
use thiserror::Error;
use uuid::Uuid;

pub(crate) fn read_to_end<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path.as_ref())?;
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
}

pub fn get_relative_asset_path<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let root = path::absolute(env::current_dir()?.join(ROOT_SOURCE_ASSETS_PATH))?;
    // Is this path relative to data folder? Check this option.
    let path = if !path.as_ref().exists() {
        root.join(path)
    } else {
        path.as_ref().into()
    };
    let path = path::absolute(path)?;

    Ok(path.strip_prefix(root).unwrap().into())
}

pub fn get_absolute_asset_path<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let root = env::current_dir()?
        .canonicalize()?
        .join(ROOT_SOURCE_ASSETS_PATH);
    Ok(root.join(get_relative_asset_path(path.as_ref())?))
}

pub fn get_compiled_asset_path<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let root = path::absolute(env::current_dir()?.join(ROOT_COMPILED_ASSETS_PATH))?;

    Ok(root.join(get_relative_asset_path(path.as_ref())?))
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
