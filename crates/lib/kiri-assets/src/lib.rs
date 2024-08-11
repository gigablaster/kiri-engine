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
mod image;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetReference(u64);

impl From<u64> for AssetReference {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    ImportFailed(String),
    ProcessingFailed(String),
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub trait AssetSource: Send + Sync {
    fn reference(&self) -> AssetReference;
}

pub trait Asset: Sized {
    fn load(data: Bytes) -> io::Result<Self>;
    fn save(&self) -> io::Result<Bytes>;
}

pub trait ImportAsset<T: AssetSource>: Asset {
    fn import(source: T, context: &dyn AssetImportContext) -> Result<Self, Error>;
}

pub trait AssetImportContext {
    fn import_image(&self, source: ImageSource) -> Result<AssetReference, Error>;
}

use std::{
    fs,
    io::{self, Read},
    path::Path,
};

use bytes::Bytes;
pub use image::*;

pub(crate) fn read_to_end<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path.as_ref())?;
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
}
