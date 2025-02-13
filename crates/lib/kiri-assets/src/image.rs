// Copyright (C) 2024-2025 gigablaster

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
    path::Path,
};

use kiri_backend::ash::vk;
use speedy::{Context, Readable, Writable};

use crate::{Asset, AssetSource, SourceAssetPath};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageAssetType {
    Rgba,
    Rg,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageAssetSource {
    pub source: SourceAssetPath,
    pub ty: ImageAssetType,
    pub srgb: bool,
}

impl ImageAssetSource {
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

impl AssetSource for ImageAssetSource {
    fn changed(&self, timestamp: std::time::SystemTime) -> bool {
        self.source.changed(timestamp)
    }

    fn reference(&self) -> kiri_vfs::AssetReference {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.source.compiled().unwrap().hash(&mut hasher);
        self.ty.hash(&mut hasher);
        self.srgb.hash(&mut hasher);
        hasher.finish().into()
    }
}

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

impl Asset for ImageAsset {
    const TYPE: uuid::Uuid = uuid::uuid!("01cca425-e0f4-4cbe-8513-62aaed5e35d5");

    fn serialize<W: std::io::Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(ImageAsset::read_from_stream_unbuffered(r)?)
    }
}
