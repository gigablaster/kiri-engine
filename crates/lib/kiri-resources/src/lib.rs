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
mod pipeline_cache;
mod resource_cache;

use std::io;

use kiri_assets::{load_asset, Asset};
use kiri_common::spawn_io;
use kiri_vfs::vfs_load;
pub use pipeline_cache::*;
pub use resource_cache::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Backend error: {0}")]
    BackendError(#[from] kiri_backend::Error),
    #[error("Renderer error: {0}")]
    RendererError(#[from] kiri_gfx::Error),
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Out of mesh memory")]
    OutOfMeshMemory,
    #[error("Too many uniforms")]
    TooManyUniforms,
    #[error("Failed to create material")]
    MaterialCreationFailed,
    #[error("Resource loading failed")]
    ResourceLoadingFailed,
    #[error("Material not found")]
    MaterialNotFound,
    #[error("Invalud model handle {0:?}")]
    InvalidModelHandle(ModelHandle),
    #[error("Render techinque {0} not found")]
    RenderTechinqueNotFound(String),
}

pub(crate) async fn load_asset_from_vfs<T: Asset>(path: impl AsRef<str>) -> io::Result<T> {
    let data = spawn_io(vfs_load(path.as_ref().to_owned())).await?;
    load_asset::<T>(&data)
}
