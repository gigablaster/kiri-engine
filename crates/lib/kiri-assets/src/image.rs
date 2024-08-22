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

use std::time::SystemTime;

use ash::vk;
use image::{imageops::FilterType, ImageBuffer};
use intel_tex_2::{bc5, bc7};
use kiri_vfs::AssetReference;
use speedy::{Context, Readable, Writable};
use uuid::uuid;

use crate::{
    get_absolute_asset_path, is_asset_changed, read_to_end, Asset, AssetSource, Error, ImportAsset,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub enum ImageAssetType {
    Srgba,
    Rgba,
    Rg,
}

impl ImageAssetType {
    pub fn srgb(self) -> bool {
        self == Self::Srgba
    }

    pub fn uncompressed_format(self) -> vk::Format {
        match self {
            ImageAssetType::Srgba => vk::Format::R8G8B8A8_SRGB,
            _ => vk::Format::R8G8B8A8_UNORM,
        }
    }

    pub fn compressed_format(self) -> vk::Format {
        match self {
            ImageAssetType::Srgba => vk::Format::BC7_SRGB_BLOCK,
            ImageAssetType::Rg => vk::Format::BC5_UNORM_BLOCK,
            _ => vk::Format::BC7_UNORM_BLOCK,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageAssetSource {
    pub path: String,
    pub ty: ImageAssetType,
}

impl ImageAssetSource {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_owned(),
            ty: ImageAssetType::Srgba,
        }
    }

    pub fn ty(mut self, value: ImageAssetType) -> Self {
        self.ty = value;
        self
    }
}

impl AssetSource for ImageAssetSource {
    fn reference(&self) -> crate::AssetReference {
        AssetReference::new(&self.path)
    }

    fn changed(&self, last_update: SystemTime) -> bool {
        is_asset_changed(&self.path, last_update)
    }
}

impl Asset for ImageAsset {
    const TYPE: uuid::Uuid = uuid!("01cca425-e0f4-4cbe-8513-62aaed5e35d5");

    fn serialize<W: std::io::Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(ImageAsset::read_from_stream_unbuffered(r)?)
    }
}

impl ImportAsset<ImageAsset> for ImageAssetSource {
    fn import(&self) -> Result<ImageAsset, Error> {
        // Load image data
        let data = read_to_end(get_absolute_asset_path(&self.path)?)?;
        // Load image
        let mut image =
            image::load_from_memory(&data).map_err(|x| Error::ImportFailed(x.to_string()))?;
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
                format: self.ty.compressed_format(),
                dims,
                mips,
            })
        } else {
            // Uncompressed image with single mip
            Ok(ImageAsset {
                format: self.ty.uncompressed_format(),
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
