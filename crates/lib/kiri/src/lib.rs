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

mod mesh;
mod pass;
mod resource_manager;
mod temp_images;

use std::io;

pub use mesh::*;
pub use pass::*;
pub use resource_manager::*;
pub use temp_images::*;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Backend error: {0}")]
    BackendError(kiri_backend::Error),
    #[error("IO error: {0}")]
    IoError(io::Error),
    #[error("Asset import error: {0}")]
    AssetImportError(kiri_assets::Error),
    #[error("Out of mesh memory")]
    OutOfMeshMemory,
}

impl From<kiri_backend::Error> for Error {
    fn from(value: kiri_backend::Error) -> Self {
        Self::BackendError(value)
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::IoError(value)
    }
}

impl From<kiri_assets::Error> for Error {
    fn from(value: kiri_assets::Error) -> Self {
        Self::AssetImportError(value)
    }
}

pub enum RenderOrder {
    Opaque,
    Transparent,
}
