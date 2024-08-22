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

use std::{
    hash::{Hash, Hasher},
    time::SystemTime,
};

use ash::vk;
use bytes::Bytes;
use image::{imageops::FilterType, ImageBuffer};
use intel_tex_2::{bc5, bc7};
use speedy::{Context, Readable, Writable};

use crate::{
    get_absolute_asset_path, is_asset_changed, read_to_end, Asset, AssetImportContext, AssetSource,
    Error, ImportAsset,
};

#[derive(Debug)]
pub struct ImageAsset {
    pub format: vk::Format,
    pub dims: [u32; 2],
    pub mips: Vec<Vec<u8>>,
}

impl<'a, C: Context> Readable<'a, C> for ImageAsset {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let format = vk::Format::from_raw(reader.read_i32()?);
        Ok(Self {
            format,
            dims: reader.read_value()?,
            mips: reader.read_value()?,
        })
    }
}

impl<C: Context> Writable<C> for ImageAsset {
    fn write_to<T: ?Sized + speedy::Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_i32(self.format.as_raw())?;
        writer.write_value(&self.dims)?;
        writer.write_value(&self.mips)?;
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImageData {
    Path(String),
    Bytes(Bytes),
    Color([u8; 4]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub enum ImageAssetType {
    Color,
    NonColor,
    Normal,
    MetallicRoughness,
    Occlusion,
    Emissive,
}

impl ImageAssetType {
    pub fn srgb(self) -> bool {
        self == Self::Color
    }

    pub fn uncompressed_format(self) -> vk::Format {
        match self {
            ImageAssetType::Color => vk::Format::R8G8B8A8_SRGB,
            _ => vk::Format::R8G8B8A8_UNORM,
        }
    }

    pub fn compressed_format(self) -> vk::Format {
        match self {
            ImageAssetType::Color => vk::Format::BC7_SRGB_BLOCK,
            ImageAssetType::Normal => vk::Format::BC5_UNORM_BLOCK,
            _ => vk::Format::BC7_UNORM_BLOCK,
        }
    }

    pub fn default_values(self) -> [u8; 4] {
        match self {
            ImageAssetType::Color => [127, 127, 127, 255],
            ImageAssetType::NonColor => [127, 127, 127, 255],
            ImageAssetType::Normal => [0, 0, 255, 255],
            ImageAssetType::MetallicRoughness => [0, 0, 0, 255],
            ImageAssetType::Occlusion => [255, 0, 0, 255],
            ImageAssetType::Emissive => [0, 0, 0, 255],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageAssetSource {
    pub ty: ImageAssetType,
    pub data: ImageData,
}

impl ImageAssetSource {
    pub fn from_file(path: &str) -> Self {
        Self {
            ty: ImageAssetType::Color,
            data: ImageData::Path(path.replace("\\", "/")),
        }
    }

    pub fn from_color(color: [u8; 4]) -> Self {
        Self {
            ty: ImageAssetType::Color,
            data: ImageData::Color(color),
        }
    }

    pub fn ty(mut self, value: ImageAssetType) -> Self {
        self.ty = value;
        self
    }
}

impl AssetSource for ImageAssetSource {
    fn reference(&self) -> crate::AssetReference {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish().into()
    }

    fn changed(&self, last_update: SystemTime) -> bool {
        if let ImageData::Path(path) = &self.data {
            is_asset_changed(path, last_update)
        } else {
            false
        }
    }
}

impl Asset for ImageAsset {
    fn load(data: Bytes) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(&data)?)
    }

    fn save(&self) -> std::io::Result<Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

impl ImportAsset<ImageAssetSource> for ImageAsset {
    fn import(source: ImageAssetSource, _context: &dyn AssetImportContext) -> Result<Self, Error> {
        // Load image data
        let data = match &source.data {
            ImageData::Path(path) => read_to_end(get_absolute_asset_path(path)?)?.into(),
            ImageData::Bytes(data) => data.clone(),
            ImageData::Color(color) => {
                // Special case - just return 1x1 image with color
                return Ok(Self {
                    format: source.ty.uncompressed_format(),
                    dims: [1, 1],
                    mips: vec![color.to_vec()],
                });
            }
        };
        // Load image
        let image =
            image::load_from_memory(&data).map_err(|x| Error::ImportFailed(x.to_string()))?;
        let dims = [image.width(), image.height()];
        let is_pow2 = dims[0].is_power_of_two() && dims[1].is_power_of_two();
        if is_pow2 && dims[0] > 16 && dims[1] > 16 {
            // Generate and compress mips
            let bc = match source.ty {
                ImageAssetType::Normal => BcMode::Bc5,
                _ => BcMode::Bc7,
            };

            let mut current_dims = dims;
            let mut mips = Vec::new();
            while current_dims[0] >= 4 && current_dims[1] >= 4 {
                let mip = block_compress(
                    image
                        .resize(current_dims[0], current_dims[1], FilterType::Lanczos3)
                        .to_rgba8(),
                    bc,
                );
                mips.push(mip);
                current_dims = [current_dims[0] >> 1, current_dims[1] >> 1];
            }
            Ok(Self {
                format: source.ty.compressed_format(),
                dims,
                mips,
            })
        } else {
            // Uncompressed image with single mip
            Ok(Self {
                format: source.ty.uncompressed_format(),
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
            let surface = intel_tex_2::RgSurface {
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
