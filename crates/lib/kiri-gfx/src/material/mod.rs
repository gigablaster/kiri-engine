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

use std::{fmt::Debug, sync::Arc};

use kiri_backend::{
    ash::vk,
    vulkan::{
        BufferSlice, DescriptorDesc, DescriptorHandle, DescriptorLayoutDesc, GraphicsDevice,
        RasterPipelineHandle, RenderPassLayout,
    },
};

use crate::ShaderUniforms;

mod basic;

pub use basic::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderGroup {
    Opaque,
    Masked,
    Transparent,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialRenderData {
    pub group: RenderGroup,
    pub priority: usize,
    pub depth: RasterPipelineHandle,
    pub main: RasterPipelineHandle,
    pub descriptor: DescriptorHandle,
}

pub trait Material {
    fn create_render_data(&self) -> MaterialRenderData;
}

pub const ZPASS_RENDER_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[],
    depth: Some(vk::Format::D24_UNORM_S8_UINT),
};

pub const MAIN_RENDER_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[vk::Format::R16G16B16A16_SFLOAT],
    depth: Some(vk::Format::D24_UNORM_S8_UINT),
};

pub const SCENE_DESCRIPTOR_LAYOUT: DescriptorLayoutDesc = DescriptorLayoutDesc {
    layout: &[(
        0,
        DescriptorDesc {
            name: "pass",
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            count: 1,
        },
    )],
    compute_groups_size: None,
};

pub const INSTANCE_DESCRIPTOR_LAYOUT: DescriptorLayoutDesc = DescriptorLayoutDesc {
    layout: &[(
        0,
        DescriptorDesc {
            name: "instances",
            ty: vk::DescriptorType::STORAGE_BUFFER,
            count: 1,
        },
    )],
    compute_groups_size: None,
};
