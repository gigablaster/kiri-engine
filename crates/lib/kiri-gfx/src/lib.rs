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

mod context;
mod descriptors;
mod dispatch;
mod draw_stream;
mod dynamic;
mod error;
mod image_pool;
mod material;
mod mesh;
pub mod passes;
mod pipeline_cache;
mod renderer;
mod staging;
mod texture;
mod uniforms;

pub use context::*;
pub use descriptors::*;
pub use dispatch::*;
pub use draw_stream::*;
use dynamic::*;
pub use error::*;
pub use image_pool::*;
pub use material::*;
pub use mesh::*;
pub use pipeline_cache::*;
pub use renderer::*;
use staging::*;
pub use texture::*;
use uniforms::*;

#[derive(Debug, Clone, Copy)]
pub struct ImageUploadData<'a> {
    pub data: &'a [u8],
}

impl<'a> ImageUploadData<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}
