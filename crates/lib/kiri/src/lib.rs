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

mod gpu;
mod mesh;
mod pipeline_cache;
mod resource_cache;
mod scene;
mod uniforms;
// mod scene;

use std::io;

pub use mesh::*;
pub use pipeline_cache::*;
pub use resource_cache::*;
pub use scene::*;
pub use uniforms::*;
// pub use scene::*;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Backend error: {0}")]
    BackendError(kiri_backend::Error),
    #[error("Renderer error: {0}")]
    RendererError(kiri_gfx::Error),
    #[error("IO error: {0}")]
    IoError(io::Error),
    #[error("Asset import error: {0}")]
    AssetImportError(kiri_assets::Error),
    #[error("Out of mesh memory")]
    OutOfMeshMemory,
    #[error("Too many uniforms")]
    TooManyUniforms,
}

impl From<kiri_backend::Error> for Error {
    fn from(value: kiri_backend::Error) -> Self {
        Self::BackendError(value)
    }
}

impl From<kiri_gfx::Error> for Error {
    fn from(value: kiri_gfx::Error) -> Self {
        Self::RendererError(value)
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

#[derive(Debug, Default, Clone, Copy)]
pub struct Bounds {
    pub center: glam::Vec3,
    pub radius: f32,
}

impl Bounds {
    pub fn from_array_and_radius(center: [f32; 3], radius: f32) -> Self {
        Self {
            center: glam::Vec3::from_array(center),
            radius,
        }
    }

    pub fn transform(self, transform: glam::Affine3A) -> Self {
        let (scale, _, _) = transform.to_scale_rotation_translation();
        let scale = scale.max_element();
        Self {
            center: transform.transform_point3(self.center),
            radius: self.radius * scale,
        }
    }
}
