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

mod packed;

use std::{
    fmt::Display,
    fs::File,
    io::{self, Read, Write},
    os::windows::fs::MetadataExt,
    path::PathBuf,
};

use bytes::Bytes;
use lazy_static::lazy_static;
pub use packed::*;
use parking_lot::RwLock;
use speedy::{Readable, Writable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub struct AssetReference(u64);

impl From<u64> for AssetReference {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl Display for AssetReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:8.8x}", self.0)
    }
}

unsafe impl Send for AssetReference {}
unsafe impl Sync for AssetReference {}

pub trait Archive: Send + Sync {
    fn load(&self, reference: AssetReference) -> io::Result<Bytes>;
    fn save(&self, reference: AssetReference, data: Bytes) -> io::Result<()>;
}

lazy_static! {
    static ref ARCHIVES: RwLock<Vec<Box<dyn Archive>>> = {
        let mut archives = Vec::<Box<dyn Archive>>::default();
        if let Ok(pack) = PackedArchive::open("data.bin") {
            archives.push(Box::new(pack));
        }
        archives.push(Box::new(LocalCache::default()));
        RwLock::new(archives)
    };
    static ref LOCAL_CACHE: LocalCache = Default::default();
}

pub fn vfs_register_archive(archive: Box<dyn Archive>) {
    ARCHIVES.write().insert(0, archive);
}

pub fn vfs_load(reference: AssetReference) -> io::Result<Bytes> {
    let archives = ARCHIVES.read();
    archives
        .iter()
        .find_map(|x| x.load(reference).ok())
        .ok_or(io::Error::other(format!("Asset {} not found", reference)))
}

#[derive(Debug, Default)]
pub struct LocalCache {}

const LOCAL_CACHE_PATH: &str = ".cache";

impl Archive for LocalCache {
    fn load(&self, reference: AssetReference) -> io::Result<Bytes> {
        let path = PathBuf::from(LOCAL_CACHE_PATH).join(format!("{}.bin", reference));
        let mut file = File::open(path)?;
        let size = file.metadata()?.file_size() as usize;
        let size = if size == 0 { 1 } else { size };
        let mut data = vec![0u8; size];
        file.read_exact(&mut data)?;
        Ok(data.into())
    }

    fn save(&self, reference: AssetReference, data: Bytes) -> io::Result<()> {
        let path = PathBuf::from(LOCAL_CACHE_PATH).join(format!("{}.bin", reference));
        let mut file = File::create(path)?;
        file.write_all(&data)?;
        Ok(())
    }
}
