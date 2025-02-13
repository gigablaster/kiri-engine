// Copyright (C) 2025 gigablaster

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

mod glsl;
mod gltf;
mod image;
mod mesh_builder;

pub use gltf::*;
use kiri_vfs::AssetReference;

use std::io::{self};

use kiri_assets::{Asset, ImageAssetSource};

pub trait ImportAsset<T: Asset> {
    fn import<I: AssetPipelineContext>(self, context: &I) -> io::Result<T>;
}

pub trait AssetPipelineContext {
    fn import_image(&self, image: ImageAssetSource) -> AssetReference;
}
