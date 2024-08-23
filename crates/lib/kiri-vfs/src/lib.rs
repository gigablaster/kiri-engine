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
    path::{self, Path, PathBuf},
};

use lazy_static::lazy_static;
pub use packed::*;
use parking_lot::RwLock;
use speedy::{Readable, Writable};

pub const ROOT_SOURCE_ASSETS_PATH: &str = "assets";
pub const ROOT_COMPILED_ASSETS_PATH: &str = "data";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Readable, Writable)]
pub struct AssetReference(String);

impl Display for AssetReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AssetReference {
    pub fn new<P: AsRef<str>>(name: P) -> AssetReference {
        Self(name.as_ref().to_owned().replace('\\', "/"))
    }

    pub fn compiled(&self) -> AssetReference {
        let mut name = if let Some((base, _)) = self.0.rsplit_once(".") {
            base.to_owned()
        } else {
            self.0.clone()
        };
        name.push_str(".asset");
        name.into()
    }

    pub fn normalized(&self) -> AssetReference {
        let name = if let Some((base, _)) = self.0.rsplit_once(".") {
            base.to_owned()
        } else {
            self.0.clone()
        };
        name.to_ascii_lowercase().into()
    }

    pub fn as_path(&self) -> &Path {
        Path::new(&self.0)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&Path> for AssetReference {
    fn from(value: &Path) -> Self {
        Self::new(value.to_str().unwrap())
    }
}

impl From<PathBuf> for AssetReference {
    fn from(value: PathBuf) -> Self {
        Self::new(value.to_str().unwrap())
    }
}

impl From<&str> for AssetReference {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for AssetReference {
    fn from(value: String) -> Self {
        Self::new(&value)
    }
}

unsafe impl Send for AssetReference {}
unsafe impl Sync for AssetReference {}

pub trait Archive: Send + Sync {
    fn load(&self, reference: &AssetReference) -> io::Result<Box<dyn Read>>;
    fn save(&self, reference: &AssetReference) -> io::Result<Box<dyn Write>>;
    fn exist(&self, reference: &AssetReference) -> bool;
}

lazy_static! {
    static ref ARCHIVES: RwLock<Vec<Box<dyn Archive>>> = {
        let mut archives = Vec::<Box<dyn Archive>>::default();
        // Data pack
        if let Ok(pack) = PackedArchive::open("data.bin") {
            archives.push(Box::new(pack));
        }
        // Compiled assets
        archives.push(Box::new(FileSystemArchive::new("data")));
        // Raw assets
        archives.push(Box::new(FileSystemArchive::new("assets")));
        RwLock::new(archives)
    };
}

pub fn vfs_register_archive(archive: Box<dyn Archive>) {
    ARCHIVES.write().insert(0, archive);
}

pub fn vfs_load(reference: &AssetReference) -> io::Result<Box<dyn Read>> {
    let archives = ARCHIVES.read();
    archives
        .iter()
        .find_map(|x| x.load(reference).ok())
        .ok_or(io::Error::other(format!("Asset {} not found", reference)))
}

pub fn vfs_exist(reference: &AssetReference) -> bool {
    let archives = ARCHIVES.read();
    archives.iter().any(|x| x.exist(reference))
}

#[derive(Debug)]
pub struct FileSystemArchive {
    root: PathBuf,
}

impl Archive for FileSystemArchive {
    fn load(&self, reference: &AssetReference) -> io::Result<Box<dyn Read>> {
        Ok(Box::new(File::open(self.path(reference)?)?))
    }

    fn save(&self, reference: &AssetReference) -> io::Result<Box<dyn Write>> {
        Ok(Box::new(File::create(self.path(reference)?)?))
    }

    fn exist(&self, reference: &AssetReference) -> bool {
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
    fn path(&self, reference: &AssetReference) -> io::Result<PathBuf> {
        let path = path::absolute(self.root.join(reference.as_path()))?;
        if !path.starts_with(&self.root) {
            return Err(io::Error::other(
                "Can't access resources outside of root path",
            ));
        }

        Ok(path)
    }
}
