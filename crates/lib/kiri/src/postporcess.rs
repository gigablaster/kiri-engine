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
    ash::vk::{self, DisplayPowerStateEXT},
    vulkan::{
        DescriptorDesc, DescriptorHandle, DescriptorLayoutDesc, DescriptorSetCreateData,
        GraphicsDevice, ImageHandle, InputVertexAttrubute, InputVertexStreamLayout,
        RasterPipelineCreateDesc, RasterPipelineHandle, RenderPassLayout, EMPTY_DESCRIPTOR_LAYOUT,
    },
    DrawStream, DrawStreamBuilder, FrameDispatcher, ImageDependency, RenderTarget,
};
use kiri_gfx::{PipelineCache, RenderTargetPool, TransientImage};

use crate::RenderView;

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

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PostprocessVertex {
    position: [f32; 2],
    uv: [f32; 2],
}
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct TonemappingGpuData {
    pub expouse: f32,
}

pub fn postprocess<'a>(
    dispatcher: &'a FrameDispatcher,
    view: RenderView,
    cache: &PipelineCache,
    pool: &'a RenderTargetPool,
    hdr: ImageHandle,
) -> Result<TransientImage<'a>, kiri_gfx::Error> {
    let tonemapping = cache.get_or_create_raster_pipeline(
        "shaders/fullscreen",
        "shaders/tonemapping",
        &POSTPROCESS_PASS_LAYOUT,
        &POSTPROCESS_DESCRIPTOR_SET_LAYOUT,
        &POSTPROCESS_INPUT_LAYOUT,
        &[],
        RasterPipelineCreateDesc::default()
            .depth_write(false)
            .depth_test(vk::CompareOp::ALWAYS),
    )?;
    let ldr = pool.image(
        vk::Format::A2R10G10B10_UNORM_PACK32,
        dispatcher.backbuffer_desc().dims,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    )?;
    let tonemapping_ds = dispatcher.temp_descriptor(DescriptorSetCreateData {
        layout: POSTPROCESS_DESCRIPTOR_LAYOUT,
        stages: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        images: &[hdr],
        unifoms: &[dispatcher.push_dynamic_data(&[TonemappingGpuData {
            expouse: view.expouse,
        }])?],
        ..Default::default()
    })?;
    dispatcher.render_pass(
        "tonemapping",
        kiri_backend::RenderArea::AllTarget,
        &[RenderTarget::new(ldr.handle).initial_layout(vk::ImageLayout::UNDEFINED)],
        None,
        &[ImageDependency::color(hdr).desired_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
        [fullscreen_quad(dispatcher, tonemapping, tonemapping_ds)?],
    );
    Ok(ldr)
}

fn fullscreen_quad(
    dispatcher: &FrameDispatcher,
    pipeline: RasterPipelineHandle,
    ds: DescriptorHandle,
) -> Result<DrawStream, kiri_gfx::Error> {
    let mut stream = DrawStreamBuilder::default();
    let vb = dispatcher.push_dynamic_data(&[
        PostprocessVertex {
            position: [-1.0, -1.0],
            uv: [0.0, 0.0],
        },
        PostprocessVertex {
            position: [1.0, -1.0],
            uv: [1.0, 0.0],
        },
        PostprocessVertex {
            position: [1.0, 1.0],
            uv: [1.0, 1.0],
        },
        PostprocessVertex {
            position: [-1.0, 1.0],
            uv: [0.0, 1.0],
        },
    ])?;
    let ib = dispatcher.push_dynamic_data(&[2u16, 1u16, 0u16, 3u16, 2u16, 0u16])?;
    stream.set_pipeline(pipeline);
    stream.set_descriptor(0, Some(ds));
    stream.set_vertex_buffer(0, vb.into());
    stream.set_index_buffer(ib.into());
    stream.draw(0, 6, 0, 1);
    Ok(stream.build())
}
