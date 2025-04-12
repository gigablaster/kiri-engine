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

// mod render_world;
mod postporcess;
mod render;
mod resource_manager;
mod scene;
use std::io;

use thiserror::Error;

// pub use render_world::*;
pub use postporcess::*;
pub use render::*;
pub use resource_manager::*;
pub use scene::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Renderer error: {0}")]
    GfxError(#[from] kiri_gfx::Error),
    #[error("Backend error: {0}")]
    BackendError(#[from] kiri_backend::Error),
    #[error("IO failed: {0}")]
    IoError(#[from] io::Error),
    #[error("Resource failed to load")]
    FailedToLoad,
}

unsafe impl Send for Error {}
