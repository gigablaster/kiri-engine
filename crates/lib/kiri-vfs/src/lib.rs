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
    hash::{Hash, Hasher},
    io::{self, Read},
    os::windows::fs::MetadataExt,
    path::{self, Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use bytes::Bytes;
use once_cell::sync::Lazy;
pub use packed::*;
use parking_lot::RwLock;
use siphasher::sip;
use speedy::{Readable, Writable};

pub const ROOT_SOURCE_ASSETS_PATH: &str = "assets";
pub const ROOT_COMPILED_ASSETS_PATH: &str = ".cache";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub struct AssetReference(u64);

impl Display for AssetReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:8.8x}", self.0)
    }
}

impl AssetReference {
    pub fn new<H: Hash>(asset: H) -> AssetReference {
        let mut hasher = sip::SipHasher::default();
        asset.hash(&mut hasher);
        Self(hasher.finish())
    }
}

unsafe impl Send for AssetReference {}
unsafe impl Sync for AssetReference {}

#[async_trait]
pub trait ArchiveLoad: Send + Sync + 'static {
    async fn load(&self, reference: AssetReference) -> io::Result<Bytes>;
}

pub trait Archive: ArchiveLoad {
    fn exist(&self, reference: AssetReference) -> bool;
}

static ARCHIVES: Lazy<RwLock<Vec<Arc<dyn Archive>>>> = Lazy::new(|| {
    let mut archives = Vec::<Arc<dyn Archive>>::default();
    // Data pack
    if let Ok(pack) = PackedArchive::open("data.bin") {
        archives.push(Arc::new(pack));
    }
    // Compiled assets outside of data pack
    archives.push(Arc::new(FileSystemArchive::new(".cache")));
    RwLock::new(archives)
});

pub fn vfs_register_archive(archive: Arc<dyn Archive>) {
    ARCHIVES.write().insert(0, archive);
}

pub async fn vfs_load(reference: AssetReference) -> io::Result<Bytes> {
    let archive = ARCHIVES
        .read()
        .iter()
        .cloned()
        .find(|x| x.exist(reference))
        .ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            reference.to_string(),
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
        // TODO: use platform-specific async IO
        let mut file = File::open(self.path(reference)?)?;
        let mut data = vec![0u8; file.metadata()?.file_size() as usize];
        file.read_exact(&mut data)?;
        Ok(data.into())
    }
}
impl Archive for FileSystemArchive {
    fn exist(&self, reference: AssetReference) -> bool {
        if let Ok(result) = self.path(reference) {
            result.exists()
        } else {
            false
        }
    }
}

impl FileSystemArchive {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: path::absolute(root).unwrap(),
        }
    }

    fn path(&self, reference: AssetReference) -> io::Result<PathBuf> {
        path::absolute(self.root.join(format!("{}.asset", reference)))
    }
}
