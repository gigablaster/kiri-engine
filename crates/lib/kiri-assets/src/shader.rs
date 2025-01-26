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

use std::io::Write;

use kiri_backend::ash::vk;
use speedy::{Readable, Writable};

use crate::Asset;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Debug, Readable, Writable)]
pub struct ShaderAsset {
    pub ty: ShaderType,
    pub bytecode: Vec<u8>,
}

impl From<ShaderType> for vk::ShaderStageFlags {
    fn from(value: ShaderType) -> Self {
        match value {
            ShaderType::Vertex => Self::VERTEX,
            ShaderType::Fragment => Self::FRAGMENT,
            ShaderType::Compute => Self::COMPUTE,
        }
    }
}

impl Asset for ShaderAsset {
    const TYPE: uuid::Uuid = uuid::uuid!("d6fb342d-938f-4ac0-9253-466f37725244");

    fn serialize<W: Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}
