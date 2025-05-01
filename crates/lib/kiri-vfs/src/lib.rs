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

mod packed;

use std::{
    fmt::Display,
    fs::File,
    io::{self, Read},
    path::{self, Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use bytes::Bytes;
use normalize_path::NormalizePath;
use once_cell::sync::Lazy;
pub use packed::*;
use parking_lot::RwLock;
use speedy::{Readable, Writable};

pub const SOURCE_ASSETS_PATH: &str = "assets";
pub const COMPILED_ASSETS_PATH: &str = "data";
pub const PACKED_ASSETS_PATH: &str = "data.bin";

#[async_trait]
pub trait ArchiveLoad: Send + Sync + 'static {
    async fn load(&self, reference: AssetReference) -> io::Result<Bytes>;
}

pub trait Archive: ArchiveLoad {
    fn exist(&self, reference: AssetReference) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Readable, Writable, Hash)]
pub struct AssetReference(u64);

impl Display for AssetReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:08x}", self.0))
    }
}

impl From<AssetReference> for u64 {
    fn from(value: AssetReference) -> Self {
        value.0
    }
}

impl From<u64> for AssetReference {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

static ARCHIVES: Lazy<RwLock<Vec<Arc<dyn Archive>>>> = Lazy::new(|| {
    let mut archives = Vec::<Arc<dyn Archive>>::default();
    // Default data pack
    if let Ok(pack) = PackedArchive::open(COMPILED_ASSETS_PATH) {
        archives.push(Arc::new(pack));
    }
    // Compiled assets outside of data pack
    archives.push(Arc::new(FileSystemArchive::new(COMPILED_ASSETS_PATH)));
    RwLock::new(archives)
});

pub fn vfs_register_archive(archive: Arc<dyn Archive>) {
    ARCHIVES.write().insert(0, archive);
}

pub async fn vfs_load(reference: AssetReference) -> io::Result<Bytes> {
    let archive = ARCHIVES
        .read()
        .iter()
        .find(|&x| x.exist(reference))
        .cloned()
        .ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Asset {} not found", reference),
        ))?;
    archive.load(reference).await
}

pub fn vfs_exist(reference: AssetReference) -> bool {
    let archives = ARCHIVES.read();
    archives.iter().any(|x| x.exist(reference))
}

#[derive(Debug)]
pub struct FileSystemArchive {
    root: PathBuf,
}

#[async_trait]
impl ArchiveLoad for FileSystemArchive {
    async fn load(&self, reference: AssetReference) -> io::Result<Bytes> {
        let path = self.root.join(
            PathBuf::from(format!("{}.asset", reference))
                .try_normalize()
                .expect("Path can't go outside of it's root"),
        );
        // TODO: use platform-specific async IO
        let mut file = File::open(path)?;
        let mut data = vec![0u8; file.metadata()?.len() as usize];
        file.read_exact(&mut data)?;
        Ok(data.into())
    }
}

impl Archive for FileSystemArchive {
    fn exist(&self, reference: AssetReference) -> bool {
        self.root
            .join(
                PathBuf::from(format!("{}.asset", reference))
                    .try_normalize()
                    .expect("Path can't go outside of it's root"),
            )
            .exists()
    }
}

impl FileSystemArchive {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: path::absolute(root).unwrap(),
        }
    }
}
