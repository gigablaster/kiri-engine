// Copyright (C) 2023-2024 gigablaster

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
    fs::{create_dir_all, File},
    io::{self},
    path::Path,
    slice,
    sync::Arc,
};

use ash::vk::{self, CompareOp, UUID_SIZE};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use log::{info, warn};

use crate::{AsVulkan, Error, Program, RenderDevice, RenderPass};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PipelineBlendDesc {
    pub src: vk::BlendFactor,
    pub dst: vk::BlendFactor,
    pub op: vk::BlendOp,
}

impl PipelineBlendDesc {
    pub fn new(src: vk::BlendFactor, dst: vk::BlendFactor, op: vk::BlendOp) -> Self {
        Self { src, dst, op }
    }
}

/// Data to create pipeline.
///
/// Contains all data to create new pipeline.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RasterPipelineCreateDesc {
    /// Blend data, None if opaque. Order: color, alpha
    pub blend: Option<(PipelineBlendDesc, PipelineBlendDesc)>,
    /// Culling
    pub cull: Option<vk::CullModeFlags>,
    /// Depth testing
    pub depth_test: Option<vk::CompareOp>,
    /// Depth writing
    pub depth_write: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexAttrubuteDesc {
    pub format: vk::Format,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexStreamLayoutDesc<'a> {
    pub streams: &'a [InputVertexAttrubuteDesc],
    pub stride: usize,
}

pub trait PipelineVertex {
    fn layout() -> &'static [InputVertexStreamLayoutDesc<'static>];
}

impl<'a> InputVertexStreamLayoutDesc<'a> {
    fn build(&self, binding: usize) -> (u32, Vec<vk::VertexInputAttributeDescription>) {
        let attributes = self
            .streams
            .iter()
            .enumerate()
            .map(|(index, attr)| vk::VertexInputAttributeDescription {
                location: index as u32,
                binding: binding as u32,
                format: attr.format.into(),
                offset: attr.offset as u32,
            })
            .collect();

        (self.stride as u32, attributes)
    }
}

impl Default for RasterPipelineCreateDesc {
    fn default() -> Self {
        Self {
            blend: None,
            cull: None,
            depth_test: Some(CompareOp::LESS_OR_EQUAL),
            depth_write: true,
        }
    }
}

impl RasterPipelineCreateDesc {
    pub fn blending(mut self, color: PipelineBlendDesc, alpha: PipelineBlendDesc) -> Self {
        self.blend = Some((color, alpha));

        self
    }

    pub fn cull(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull = Some(mode);

        self
    }

    pub fn alpha_blend(mut self) -> Self {
        self.blend = Some((
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC1_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC1_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
        ));
        self
    }

    pub fn premultiplied(mut self) -> Self {
        self.blend = Some((
            PipelineBlendDesc::new(
                vk::BlendFactor::ONE,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
            PipelineBlendDesc::new(
                vk::BlendFactor::ONE,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
        ));
        self
    }

    pub fn additive(mut self) -> Self {
        self.blend = Some((
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE,
                vk::BlendOp::ADD,
            ),
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE,
                vk::BlendOp::ADD,
            ),
        ));
        self
    }

    pub fn depth_write(mut self) -> Self {
        self.depth_write = true;

        self
    }

    pub fn depth_test(mut self, value: vk::CompareOp) -> Self {
        self.depth_test = Some(value);

        self
    }
}

#[derive(Debug)]
pub struct Pipeline {
    device: Arc<RenderDevice>,
    _program: Arc<Program>,
    raw: vk::Pipeline,
}

#[derive(Debug)]
pub struct RasterPipeline {
    base: Pipeline,
    _render_pass: Arc<RenderPass>,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { self.device.get().destroy_pipeline(self.raw, None) }
    }
}

impl AsVulkan<vk::Pipeline> for Pipeline {
    fn as_vulkan(&self) -> vk::Pipeline {
        self.raw
    }
}

impl AsVulkan<vk::Pipeline> for RasterPipeline {
    fn as_vulkan(&self) -> vk::Pipeline {
        self.base.as_vulkan()
    }
}

impl Pipeline {
    pub fn raster(
        device: &Arc<RenderDevice>,
        cache: vk::PipelineCache,
        input: &[InputVertexStreamLayoutDesc],
        program: &Arc<Program>,
        render_pass: &Arc<RenderPass>,
        subpass: usize,
        desc: RasterPipelineCreateDesc,
    ) -> Result<RasterPipeline, Error> {
        let shader_create_info = program
            .shaders()
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(shader.stage())
                    .module(shader.as_vulkan())
                    .name(shader.entry())
            })
            .collect::<Vec<_>>();

        let assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let streams = input
            .iter()
            .enumerate()
            .map(|(index, stream)| stream.build(index))
            .collect::<Vec<_>>();

        let strides = streams
            .iter()
            .map(|(stride, _)| stride)
            .copied()
            .collect::<Vec<_>>();
        let attributes = streams
            .iter()
            .flat_map(|(_, attributes)| attributes)
            .copied()
            .collect::<Vec<_>>();
        let vertex_binding_desc = strides
            .iter()
            .enumerate()
            .map(|(index, _)| {
                vk::VertexInputBindingDescription::default()
                    .stride(strides[index] as _)
                    .binding(attributes[index].binding)
                    .input_rate(vk::VertexInputRate::VERTEX)
            })
            .collect::<Vec<_>>();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_desc)
            .vertex_attribute_descriptions(&attributes);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default();

        let rasterizer_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_bias_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .cull_mode(desc.cull.unwrap_or(vk::CullModeFlags::NONE))
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let mut depthstencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .stencil_test_enable(false)
            .depth_write_enable(desc.depth_write);

        if let Some(depth_compare) = desc.depth_test {
            depthstencil_state = depthstencil_state
                .depth_test_enable(true)
                .depth_compare_op(depth_compare);
        }

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_attachment = if let Some((color, alpha)) = desc.blend {
            color_blend_attachment
                .blend_enable(true)
                .src_color_blend_factor(color.src)
                .dst_color_blend_factor(color.dst)
                .color_blend_op(color.op)
                .src_alpha_blend_factor(alpha.src)
                .dst_alpha_blend_factor(alpha.dst)
                .alpha_blend_op(alpha.op)
        } else {
            color_blend_attachment.blend_enable(false)
        };
        let blending_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(slice::from_ref(&color_blend_attachment))
            .logic_op_enable(false);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .layout(program.as_vulkan())
            .stages(&shader_create_info)
            .vertex_input_state(&vertex_input)
            .dynamic_state(&dynamic_state_create_info)
            .viewport_state(&viewport_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&blending_state)
            .input_assembly_state(&assembly_state_create_info)
            .rasterization_state(&rasterizer_state)
            .depth_stencil_state(&depthstencil_state)
            .render_pass(render_pass.as_vulkan())
            .subpass(subpass as _);

        let pipeline = unsafe {
            device.get().create_graphics_pipelines(
                cache,
                slice::from_ref(&pipeline_create_info),
                None,
            )
        }?[0];

        Ok(RasterPipeline {
            base: Pipeline {
                device: device.clone(),
                _program: program.clone(),
                raw: pipeline,
            },
            _render_pass: render_pass.clone(),
        })
    }
}

const MAGICK: [u8; 4] = *b"PLCH";
const VERSION: u32 = 1;

#[derive(Debug, PartialEq, Eq)]
struct Header {
    pub magic: [u8; 4],
    pub version: u32,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            magic: MAGICK,
            version: VERSION,
        }
    }
}

impl Header {
    pub fn write<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_u32::<LittleEndian>(self.version)?;

        Ok(())
    }

    pub fn read<R: io::Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        Ok(Self {
            magic,
            version: r.read_u32::<LittleEndian>()?,
        })
    }

    pub fn validate(&self) -> bool {
        self.magic == MAGICK && self.version >= VERSION
    }
}

#[derive(Debug)]
struct PipelineDiskCache {
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
    uuid: [u8; UUID_SIZE],
    data: Vec<u8>,
}

impl PipelineDiskCache {
    pub fn new(device: &RenderDevice, data: &[u8]) -> Self {
        let pdevice = device.physical_device();
        let vendor_id = pdevice.properties.vendor_id;
        let device_id = pdevice.properties.device_id;
        let driver_version = pdevice.properties.driver_version;
        let uuid = pdevice.properties.pipeline_cache_uuid;

        Self {
            vendor_id,
            device_id,
            driver_version,
            uuid,
            data: data.to_vec(),
        }
    }

    pub fn read<R: io::Read>(mut r: R) -> io::Result<Self> {
        if !Header::read(&mut r)?.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Wrong pipeline cache header",
            ));
        }
        let vendor_id = r.read_u32::<LittleEndian>()?;
        let device_id = r.read_u32::<LittleEndian>()?;
        let driver_version = r.read_u32::<LittleEndian>()?;
        let mut uuid = [0u8; UUID_SIZE];
        r.read_exact(&mut uuid)?;
        let data = Self::read_data(&mut r)?;
        Ok(Self {
            vendor_id,
            device_id,
            driver_version,
            uuid,
            data,
        })
    }

    fn read_data<R: io::Read>(r: &mut R) -> io::Result<Vec<u8>> {
        let size = r.read_u32::<LittleEndian>()?;
        let mut bytes = vec![0u8; size as usize];
        r.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub fn save<W: io::Write>(&self, mut w: W) -> io::Result<()> {
        Header::default().write(&mut w)?;
        w.write_u32::<LittleEndian>(self.vendor_id)?;
        w.write_u32::<LittleEndian>(self.device_id)?;
        w.write_u32::<LittleEndian>(self.driver_version)?;
        w.write_all(&self.uuid)?;
        w.write_u32::<LittleEndian>(self.data.len() as _)?;
        w.write_all(&self.data)?;
        Ok(())
    }
}

pub fn load_or_create_pipeline_cache<P: AsRef<Path>>(
    device: &RenderDevice,
    path: P,
) -> io::Result<vk::PipelineCache> {
    info!("Loading pipeline cache from {:?}", path.as_ref());
    let pdevice = device.physical_device();
    let data = if let Ok(file) = File::open(path) {
        if let Ok(cache) = PipelineDiskCache::read(file) {
            if cache.vendor_id == pdevice.properties.vendor_id
                && cache.device_id == pdevice.properties.device_id
                && cache.driver_version == pdevice.properties.driver_version
                && cache.uuid == pdevice.properties.pipeline_cache_uuid
            {
                Some(cache.data)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let create_info = if let Some(data) = &data {
        vk::PipelineCacheCreateInfo::default().initial_data(data)
    } else {
        vk::PipelineCacheCreateInfo::default()
    };

    let cache = match unsafe { device.get().create_pipeline_cache(&create_info, None) } {
        Ok(cache) => cache,
        Err(_) => {
            // Failed with initial data - so create empty cache.
            warn!("Failed to load pipeline cache. Create new one.");
            let create_info = vk::PipelineCacheCreateInfo::default();
            unsafe { device.get().create_pipeline_cache(&create_info, None) }
                .map_err(|err| io::Error::other(err.to_string()))?
        }
    };

    Ok(cache)
}

pub fn save_pipeline_cache<P: AsRef<Path>>(
    device: &RenderDevice,
    cache: vk::PipelineCache,
    path: P,
) -> io::Result<()> {
    info!("Saving pipeline cache to {:?}", path.as_ref());
    let data = unsafe { device.get().get_pipeline_cache_data(cache) }.map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get pipeline cache data from device: {:?}", err),
        )
    })?;
    create_dir_all(path.as_ref().parent().unwrap())?;
    PipelineDiskCache::new(device, &data).save(File::create(path)?)
}
