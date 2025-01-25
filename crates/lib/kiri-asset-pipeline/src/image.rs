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

use std::{io, path::Path};

use crate::{read_to_end, AssetPipelineContext, ImportAsset};
use image::{imageops::FilterType, ImageBuffer};
use intel_tex_2::{bc5, bc7};
use kiri_assets::{ImageAsset, SourceAssetPath};
use kiri_backend::ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageAssetType {
    Rgba,
    Rg,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageSource {
    pub source: SourceAssetPath,
    pub ty: ImageAssetType,
    pub srgb: bool,
}

impl ImageSource {
    pub fn new<S: AsRef<Path>>(path: S) -> Self {
        Self {
            source: SourceAssetPath::new(path),
            ty: ImageAssetType::Rgba,
            srgb: false,
        }
    }

    pub fn ty(mut self, value: ImageAssetType) -> Self {
        self.ty = value;
        self
    }

    pub fn srgb(mut self, value: bool) -> Self {
        self.srgb = value;
        self
    }

    pub fn compressed_format(&self) -> vk::Format {
        match self.ty {
            ImageAssetType::Rgba if self.srgb => vk::Format::BC7_SRGB_BLOCK,
            ImageAssetType::Rgba => vk::Format::BC7_UNORM_BLOCK,
            ImageAssetType::Rg => vk::Format::BC5_UNORM_BLOCK,
        }
    }

    pub fn uncompressed_format(&self) -> vk::Format {
        match self.ty {
            ImageAssetType::Rgba if self.srgb => vk::Format::A8B8G8R8_SRGB_PACK32,
            _ => vk::Format::A8B8G8R8_UNORM_PACK32,
        }
    }
}

impl ImportAsset<ImageAsset> for ImageSource {
    fn import(self, _context: &impl AssetPipelineContext) -> io::Result<ImageAsset> {
        let data = read_to_end(&self.source)?;
        let mut image =
            image::load_from_memory(&data).map_err(|x| io::Error::other(x.to_string()))?;
        let dims = [image.width(), image.height()];
        let is_pow2 = dims[0].is_power_of_two() && dims[1].is_power_of_two();
        if is_pow2 && dims[0] > 16 && dims[1] > 16 {
            // Generate and compress mips
            let bc = match self.ty {
                ImageAssetType::Rg => BcMode::Bc5,
                _ => BcMode::Bc7,
            };

            let mut current_dims = dims;
            let mut mips = Vec::new();
            while current_dims[0] >= 4 && current_dims[1] >= 4 {
                let mip = block_compress(image.to_rgba8(), bc);
                mips.push(mip);
                current_dims = [current_dims[0] >> 1, current_dims[1] >> 1];
                image = image.resize(current_dims[0], current_dims[1], FilterType::Lanczos3);
            }
            Ok(ImageAsset {
                format: self.compressed_format(),
                dims,
                mips,
            })
        } else {
            // Uncompressed image with single mip
            Ok(ImageAsset {
                format: self.uncompressed_format(),
                dims,
                mips: vec![image.to_rgba8().into_raw()],
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BcMode {
    Bc5,
    Bc7,
}

impl BcMode {
    fn block_bytes(self) -> usize {
        match self {
            BcMode::Bc5 => 16,
            BcMode::Bc7 => 16,
        }
    }
}

fn block_compress(image: ImageBuffer<image::Rgba<u8>, Vec<u8>>, bc: BcMode) -> Vec<u8> {
    let block_count = intel_tex_2::divide_up_by_multiple(image.width() * image.height(), 16);
    let needs_alpha = bc == BcMode::Bc7 && image.pixels().any(|px| px.0[3] != 255);
    let block_bytes = bc.block_bytes();
    let mut compressed_bytes = vec![0u8; block_count as usize * block_bytes];
    match bc {
        BcMode::Bc5 => {
            let surface = intel_tex_2::RgbaSurface {
                width: image.width(),
                height: image.height(),
                stride: image.width() * 4,
                data: image.as_raw(),
            };
            bc5::compress_blocks_into(&surface, &mut compressed_bytes)
        }
        BcMode::Bc7 => {
            let surface = intel_tex_2::RgbaSurface {
                width: image.width(),
                height: image.height(),
                stride: image.width() * 4,
                data: image.as_raw(),
            };

            let settings = if needs_alpha {
                bc7::alpha_basic_settings()
            } else {
                bc7::opaque_basic_settings()
            };
            bc7::compress_blocks_into(&settings, &surface, &mut compressed_bytes)
        }
    }

    compressed_bytes
}
