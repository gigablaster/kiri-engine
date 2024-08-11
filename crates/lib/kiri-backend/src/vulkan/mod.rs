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

mod barrier;
mod buffer;
mod context;
mod draw_stream;
mod drop_list;
mod error;
mod frame;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod program;
mod staging;
mod swapchain;

use ash::vk;
pub use barrier::*;
pub use buffer::*;
pub use context::*;
pub use draw_stream::*;
use drop_list::*;
pub use error::*;
use frame::*;
pub use image::*;
pub use instance::*;
pub use physical_device::*;
pub use pipeline::*;
use program::*;
pub use swapchain::*;

use crate::Format;

type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
type GpuMemory = gpu_alloc::MemoryBlock<vk::DeviceMemory>;

impl From<Format> for vk::Format {
    fn from(value: Format) -> Self {
        match value {
            Format::INVALID => vk::Format::UNDEFINED,
            Format::R8_UNORM => vk::Format::R8_UNORM,
            Format::R8_SNORM => vk::Format::R8_SNORM,
            Format::R8_USCALED => vk::Format::R8_USCALED,
            Format::R8_SSCALED => vk::Format::R8_SSCALED,
            Format::R8_UINT => vk::Format::R8_UINT,
            Format::R8_SINT => vk::Format::R8_SINT,
            Format::R8_SRGB => vk::Format::R8_SRGB,
            Format::RG8_UNORM => vk::Format::R8G8_UNORM,
            Format::RG8_SNORM => vk::Format::R8G8_SNORM,
            Format::RG8_USCALED => vk::Format::R8G8_USCALED,
            Format::RG8_SSCALED => vk::Format::R8G8_SSCALED,
            Format::RG8_UINT => vk::Format::R8G8_UINT,
            Format::RG8_SINT => vk::Format::R8G8_SINT,
            Format::RG8_SRGB => vk::Format::R8G8_SRGB,
            Format::RGB8_UNORM => vk::Format::R8G8B8_UNORM,
            Format::RGB8_SNORM => vk::Format::R8G8B8_SNORM,
            Format::RGB8_USCALED => vk::Format::R8G8B8_USCALED,
            Format::RGB8_SSCALED => vk::Format::R8G8B8_SSCALED,
            Format::RGB8_UINT => vk::Format::R8G8B8_UINT,
            Format::RGB8_SINT => vk::Format::R8G8B8_SINT,
            Format::RGB8_SRGB => vk::Format::R8G8B8_SRGB,
            Format::RGBA8_UNORM => vk::Format::R8G8B8A8_UNORM,
            Format::RGBA8_SNORM => vk::Format::R8G8B8A8_SNORM,
            Format::RGBA8_USCALED => vk::Format::R8G8B8A8_USCALED,
            Format::RGBA8_SSCALED => vk::Format::R8G8B8A8_SSCALED,
            Format::RGBA8_UINT => vk::Format::R8G8B8A8_UINT,
            Format::RGBA8_SINT => vk::Format::R8G8B8A8_SINT,
            Format::RGBA8_SRGB => vk::Format::R8G8B8A8_SRGB,
            Format::BGR8_UNORM => vk::Format::B8G8R8_UNORM,
            Format::BGR8_SNORM => vk::Format::B8G8R8_SNORM,
            Format::BGR8_USCALED => vk::Format::B8G8R8_USCALED,
            Format::BGR8_SSCALED => vk::Format::B8G8R8_SSCALED,
            Format::BGR8_UINT => vk::Format::B8G8R8_UINT,
            Format::BGR8_SINT => vk::Format::B8G8R8_UINT,
            Format::BGR8_SRGB => vk::Format::B8G8R8_SRGB,
            Format::BGRA8_UNORM => vk::Format::B8G8R8A8_UNORM,
            Format::BGRA8_SNORM => vk::Format::B8G8R8A8_SNORM,
            Format::BGRA8_USCALED => vk::Format::B8G8R8A8_USCALED,
            Format::BGRA8_SSCALED => vk::Format::B8G8R8A8_SSCALED,
            Format::BGRA8_UINT => vk::Format::B8G8R8A8_UINT,
            Format::BGRA8_SINT => vk::Format::B8G8R8A8_SINT,
            Format::BGRA8_SRGB => vk::Format::B8G8R8A8_SRGB,
            Format::R16_UNORM => vk::Format::R16_UNORM,
            Format::R16_SNORM => vk::Format::R16_SNORM,
            Format::R16_USCALED => vk::Format::R16_USCALED,
            Format::R16_SSCALED => vk::Format::R16_SSCALED,
            Format::R16_UINT => vk::Format::R16_UINT,
            Format::R16_SINT => vk::Format::R16_SINT,
            Format::R16_SFLOAT => vk::Format::R16_SFLOAT,
            Format::RG16_UNORM => vk::Format::R16G16_UNORM,
            Format::RG16_SNORM => vk::Format::R16G16_SNORM,
            Format::RG16_USCALED => vk::Format::R16G16_USCALED,
            Format::RG16_SSCALED => vk::Format::R16G16_SSCALED,
            Format::RG16_UINT => vk::Format::R16G16_UINT,
            Format::RG16_SINT => vk::Format::R16G16_SINT,
            Format::RG16_SFLOAT => vk::Format::R16G16_SFLOAT,
            Format::RGB16_UNORM => vk::Format::R16G16B16_UNORM,
            Format::RGB16_SNORM => vk::Format::R16G16B16_SNORM,
            Format::RGB16_USCALED => vk::Format::R16G16B16_USCALED,
            Format::RGB16_SSCALED => vk::Format::R16G16B16_USCALED,
            Format::RGB16_UINT => vk::Format::R16G16B16_UINT,
            Format::RGB16_SINT => vk::Format::R16G16B16_SINT,
            Format::RGB16_SFLOAT => vk::Format::R16G16B16_SFLOAT,
            Format::RGBA16_UNORM => vk::Format::R16G16B16A16_UNORM,
            Format::RGBA16_SNORM => vk::Format::R16G16B16A16_SNORM,
            Format::RGBA16_USCALED => vk::Format::R16G16B16A16_USCALED,
            Format::RGBA16_SSCALED => vk::Format::R16G16B16A16_SSCALED,
            Format::RGBA16_UINT => vk::Format::R16G16B16A16_UINT,
            Format::RGBA16_SINT => vk::Format::R16G16B16A16_SINT,
            Format::RGBA16_SFLOAT => vk::Format::R16G16B16A16_SFLOAT,
            Format::R32_UINT => vk::Format::R32_UINT,
            Format::R32_SINT => vk::Format::R32_SINT,
            Format::R32_SFLOAT => vk::Format::R32_SFLOAT,
            Format::RG32_UINT => vk::Format::R32G32_UINT,
            Format::RG32_SINT => vk::Format::R32G32_SINT,
            Format::RG32_SFLOAT => vk::Format::R32G32_SFLOAT,
            Format::RGB32_UINT => vk::Format::R32G32B32_UINT,
            Format::RGB32_SINT => vk::Format::R32G32B32_SINT,
            Format::RGB32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
            Format::RGBA32_UINT => vk::Format::R32G32B32A32_UINT,
            Format::RGBA32_SINT => vk::Format::R32G32B32A32_SINT,
            Format::RGBA32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
            Format::D16 => vk::Format::D16_UNORM,
            Format::D24 => vk::Format::X8_D24_UNORM_PACK32,
            Format::D32 => vk::Format::D32_SFLOAT,
            Format::D16_S8 => vk::Format::D16_UNORM_S8_UINT,
            Format::D24_S8 => vk::Format::D24_UNORM_S8_UINT,
            Format::BC1_RGB_UNORM => vk::Format::BC1_RGB_UNORM_BLOCK,
            Format::BC1_RGB_SRGB => vk::Format::BC1_RGB_SRGB_BLOCK,
            Format::BC1_RGBA_UNORM => vk::Format::BC1_RGBA_UNORM_BLOCK,
            Format::BC1_RGBA_SRGB => vk::Format::BC1_RGBA_SRGB_BLOCK,
            Format::BC2_UNORM => vk::Format::BC2_UNORM_BLOCK,
            Format::BC2_SRGB => vk::Format::BC2_SRGB_BLOCK,
            Format::BC3_UNORM => vk::Format::BC3_UNORM_BLOCK,
            Format::BC3_SRGB => vk::Format::BC3_SRGB_BLOCK,
            Format::BC4_UNORM => vk::Format::BC4_UNORM_BLOCK,
            Format::BC4_SNORM => vk::Format::BC4_SNORM_BLOCK,
            Format::BC5_UNORM => vk::Format::BC5_UNORM_BLOCK,
            Format::BC5_SNORM => vk::Format::BC5_SNORM_BLOCK,
            Format::BC6_UFLOAT => vk::Format::BC6H_UFLOAT_BLOCK,
            Format::BC6_SFLOAT => vk::Format::BC6H_SFLOAT_BLOCK,
            Format::BC7_UNORM => vk::Format::BC7_UNORM_BLOCK,
            Format::BC7_SRGB => vk::Format::BC7_SRGB_BLOCK,
        }
    }
}
