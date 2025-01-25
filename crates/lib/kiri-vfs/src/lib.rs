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
    fs::File,
    io::{self, Read},
    os::windows::fs::MetadataExt,
    path::{self, Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use bytes::Bytes;
use normalize_path::NormalizePath;
use once_cell::sync::Lazy;
pub use packed::*;
use parking_lot::RwLock;

pub const SOURCE_ASSETS_PATH: &str = "assets";
pub const COMPILED_ASSETS_PATH: &str = "data";
pub const PACKED_ASSETS_PATH: &str = "data.bin";

#[async_trait]
pub trait ArchiveLoad: Send + Sync + 'static {
    async fn load(&self, path: &str) -> io::Result<Bytes>;
}

pub trait Archive: ArchiveLoad {
    fn exist(&self, path: &str) -> bool;
}

static ARCHIVES: Lazy<RwLock<Vec<Arc<dyn Archive>>>> = Lazy::new(|| {
    let mut archives = Vec::<Arc<dyn Archive>>::default();
    // Data pack
    if let Ok(pack) = PackedArchive::open(COMPILED_ASSETS_PATH) {
        archives.push(Arc::new(pack));
    }
    // Compiled assets outside of data pack
    archives.push(Arc::new(FileSystemArchive::new(COMPILED_ASSETS_PATH)));
    // Source assets
    archives.push(Arc::new(FileSystemArchive::new(SOURCE_ASSETS_PATH)));
    RwLock::new(archives)
});

pub fn vfs_register_archive(archive: Arc<dyn Archive>) {
    ARCHIVES.write().insert(0, archive);
}

pub async fn vfs_load<P: AsRef<str>>(path: P) -> io::Result<Bytes> {
    let archive = ARCHIVES
        .read()
        .iter()
        .cloned()
        .find(|x| x.exist(path.as_ref()))
        .ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            path.as_ref().to_owned(),
        ))?;
    archive.load(path.as_ref()).await
}

pub fn vfs_exist<P: AsRef<str>>(path: P) -> bool {
    let archives = ARCHIVES.read();
    archives.iter().any(|x| x.exist(path.as_ref()))
}

#[derive(Debug)]
pub struct FileSystemArchive {
    root: PathBuf,
}

#[async_trait]
impl ArchiveLoad for FileSystemArchive {
    async fn load(&self, path: &str) -> io::Result<Bytes> {
        let path = self.root.join(
            PathBuf::from(path)
                .try_normalize()
                .expect("Path can't go outside of it's root"),
        );
        // TODO: use platform-specific async IO
        let mut file = File::open(path)?;
        let mut data = vec![0u8; file.metadata()?.file_size() as usize];
        file.read_exact(&mut data)?;
        Ok(data.into())
    }
}

impl Archive for FileSystemArchive {
    fn exist(&self, path: &str) -> bool {
        self.root
            .join(
                PathBuf::from(path)
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
