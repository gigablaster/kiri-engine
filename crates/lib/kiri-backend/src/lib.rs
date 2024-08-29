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

mod buffer;
mod drop_list;
mod error;
mod frame;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod program;
mod render_device;
mod swapchain;

use ash::vk;
pub use buffer::*;
use drop_list::*;
pub use error::*;
pub use frame::*;
pub use image::*;
pub use instance::*;
pub use physical_device::*;
pub use pipeline::*;
pub use program::*;
pub use render_device::*;
pub use swapchain::*;

pub type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
pub type GpuMemoryBlock = gpu_alloc::MemoryBlock<vk::DeviceMemory>;
