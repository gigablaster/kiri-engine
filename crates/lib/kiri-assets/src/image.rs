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

use kiri_backend::ash::vk;
use speedy::{Context, Readable, Writable};

use crate::Asset;

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
