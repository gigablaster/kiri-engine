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

// mod gltf;
mod image;
mod model;
mod shader;

pub use image::*;
pub use model::*;
use normalize_path::NormalizePath;
pub use shader::*;

use std::{
    fmt::Display,
    fs,
    io::{self, Cursor, Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

use kiri_vfs::{AssetReference, SOURCE_ASSETS_PATH};
use speedy::{Context, Readable, Writable};
use uuid::Uuid;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Writable, Readable)]
pub struct CompiledAssetPath(String);

impl AsRef<str> for CompiledAssetPath {
    fn as_ref(&self) -> &str {
        self.0.as_str()
    }
}

impl From<CompiledAssetPath> for PathBuf {
    fn from(value: CompiledAssetPath) -> PathBuf {
        Self::from(value.0)
    }
}

impl AsRef<Path> for CompiledAssetPath {
    fn as_ref(&self) -> &Path {
        Path::new(&self.0)
    }
}

impl From<&str> for CompiledAssetPath {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SourceAssetPath(PathBuf);

impl From<&str> for SourceAssetPath {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl From<String> for SourceAssetPath {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl From<&Path> for SourceAssetPath {
    fn from(value: &Path) -> Self {
        Self(value.to_path_buf())
    }
}

impl From<PathBuf> for SourceAssetPath {
    fn from(value: PathBuf) -> Self {
        Self(value)
    }
}

impl AsRef<Path> for SourceAssetPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<str> for SourceAssetPath {
    fn as_ref(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl SourceAssetPath {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self(path.as_ref().to_path_buf())
    }

    /// Create compiled asset path and replace extension if needed.
    ///
    /// File path is normalized and checked against root to prevent any form of accessing
    /// data outside of proper folder.
    pub fn compiled(&self) -> io::Result<CompiledAssetPath> {
        let path = self.full_source_path();
        let root = Self::source_assets_root();
        assert!(path.starts_with(&root));
        let name = path
            .with_extension("")
            .strip_prefix(root)
            .map_err(io::Error::other)?
            .to_str()
            .unwrap()
            .replace('\\', "/")
            .to_ascii_lowercase();
        Ok(CompiledAssetPath(name))
    }

    pub fn changed(&self, timestamp: SystemTime) -> bool {
        let path = self.full_source_path();
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                return modified > timestamp;
            }
            if let Ok(created) = metadata.created() {
                return created > timestamp;
            }
        }
        false
    }

    pub fn full_source_path(&self) -> PathBuf {
        Self::source_assets_root().join(&self.0).normalize()
    }

    pub fn parent(&self) -> PathBuf {
        self.full_source_path().parent().unwrap().to_path_buf()
    }

    pub fn source_assets_root() -> PathBuf {
        PathBuf::from(SOURCE_ASSETS_PATH)
    }
}

pub trait AssetSource {
    fn changed(&self, timestamp: SystemTime) -> bool;
    fn reference(&self) -> AssetReference;
}

pub trait Asset: Sized + Send + Sync + 'static {
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

    pub fn valid<T: Asset>(&self) -> bool {
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

pub fn save_asset<T: Asset, W: Write>(w: W, asset: &T) -> io::Result<()> {
    let mut w = w;
    AssetHeader::new::<T>().write_to_stream(&mut w)?;
    asset.serialize(w)
}

pub fn load_asset<T: Asset>(data: &[u8]) -> io::Result<T> {
    let mut reader = Cursor::new(data);
    let header = AssetHeader::read_from_stream_unbuffered(&mut reader)?;
    if !header.valid::<T>() {
        return Err(io::Error::other("Asset header isn't valid"));
    }
    T::deserialize(&mut reader)
}

#[cfg(test)]
mod test {
    use crate::{CompiledAssetPath, SourceAssetPath};

    #[test]
    fn source_asset_in_root() {
        assert_eq!(
            CompiledAssetPath::from("aaa"),
            SourceAssetPath::new("aaa.png").compiled().unwrap()
        );
    }

    #[test]
    fn source_asset_in_folder() {
        assert_eq!(
            CompiledAssetPath::from("foo/bar"),
            SourceAssetPath::new("foo/bar.jpg").compiled().unwrap()
        )
    }
}

pub fn read_to_end<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path.as_ref())?;
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
}
