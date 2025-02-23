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

use kiri_backend::{
    ash::vk,
    vulkan::{
        DescriptorDesc, DescriptorLayoutDesc, InputVertexAttrubute, InputVertexStreamLayout,
        RenderPassLayout, EMPTY_DESCRIPTOR_LAYOUT,
    },
};

const POSTPROCESS_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[vk::Format::A2R10G10B10_UNORM_PACK32],
    depth: None,
};

const POSTPROCESS_INPUT_LAYOUT: [InputVertexStreamLayout; 1] = [InputVertexStreamLayout {
    streams: &[
        InputVertexAttrubute {
            location: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0,
        },
        InputVertexAttrubute {
            location: 1,
            format: vk::Format::R32G32_SFLOAT,
            offset: 8,
        },
    ],
    stride: 16,
}];

static POSTPROCESS_DESCRIPTOR_LAYOUT: DescriptorLayoutDesc = DescriptorLayoutDesc {
    layout: &[
        (
            0,
            DescriptorDesc {
                name: "main",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            1,
            DescriptorDesc {
                name: "params",
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
            },
        ),
    ],
    compute_groups_size: None,
};

static POSTPROCESS_DESCRIPTOR_SET_LAYOUT: [DescriptorLayoutDesc; 4] = [
    POSTPROCESS_DESCRIPTOR_LAYOUT,
    EMPTY_DESCRIPTOR_LAYOUT,
    EMPTY_DESCRIPTOR_LAYOUT,
    EMPTY_DESCRIPTOR_LAYOUT,
];
