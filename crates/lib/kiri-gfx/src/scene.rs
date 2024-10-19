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

use kiri_backend::{
    ash::vk::{self},
    RenderPassLayout,
};

pub const GBUFFER_RENDER_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[
        vk::Format::A2R10G10B10_UNORM_PACK32,
        vk::Format::A2R10G10B10_UNORM_PACK32,
        vk::Format::A2R10G10B10_UNORM_PACK32,
        vk::Format::R16G16B16A16_SFLOAT,
    ],
    depth: Some(vk::Format::X8_D24_UNORM_PACK32),
};

pub const SHADOW_MAP_RENDER_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[],
    depth: Some(vk::Format::X8_D24_UNORM_PACK32),
};
