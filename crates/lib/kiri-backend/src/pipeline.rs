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
    io::{self, Cursor, Seek, Write},
    mem,
    path::Path,
    slice,
    sync::Arc,
};

use ash::vk::{self, CompareOp, UUID_SIZE};
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use log::{info, warn};

use crate::{Error, Program, RenderDevice};

pub const MAX_COLOR_ATTACHMENTS: usize = 8;
pub const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Copy)]
pub enum AttachmentClearValue {
    Color([f32; 4]),
    DepthStencil(f32, u32),
}

impl From<AttachmentClearValue> for vk::ClearValue {
    fn from(value: AttachmentClearValue) -> Self {
        match value {
            AttachmentClearValue::Color(color) => vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            },
            AttachmentClearValue::DepthStencil(depth, stencil) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderAttachmentLayoutDesc<'a> {
    pub color: &'a [vk::Format],
    pub depth: Option<vk::Format>,
}

impl<'a> RenderAttachmentLayoutDesc<'a> {
    fn build(self) -> vk::PipelineRenderingCreateInfo<'a> {
        let mut info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(self.color);
        if let Some(depth) = self.depth {
            info = info.depth_attachment_format(depth);
        }
        info
    }
}

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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexAttrubute {
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexStreamLayout<'a> {
    pub streams: &'a [InputVertexAttrubute],
    pub stride: u32,
}

pub trait PipelineVertex {
    fn layout() -> &'static [InputVertexStreamLayout<'static>];
}

#[derive(Debug, Clone, Copy)]
pub enum SpecializationValue {
    Int(i32),
    Uint(u32),
    Float(f32),
}

impl SpecializationValue {
    fn write<W: Write>(self, mut w: W) -> io::Result<()> {
        match self {
            SpecializationValue::Int(value) => Ok(w.write_i32::<NativeEndian>(value)?),
            SpecializationValue::Uint(value) => Ok(w.write_u32::<NativeEndian>(value)?),
            SpecializationValue::Float(value) => Ok(w.write_f32::<NativeEndian>(value)?),
        }
    }

    fn size(self) -> usize {
        match self {
            SpecializationValue::Int(_) => mem::size_of::<i32>(),
            SpecializationValue::Uint(_) => mem::size_of::<u32>(),
            SpecializationValue::Float(_) => mem::size_of::<f32>(),
        }
    }
}

impl<'a> InputVertexStreamLayout<'a> {
    fn build(&self, binding: u32) -> (u32, Vec<vk::VertexInputAttributeDescription>) {
        let attributes = self
            .streams
            .iter()
            .enumerate()
            .map(|(index, attr)| vk::VertexInputAttributeDescription {
                location: index as u32,
                binding,
                format: attr.format,
                offset: attr.offset,
            })
            .collect();

        (self.stride, attributes)
    }
}

pub fn compile_raster_pipeline<'a>(
    device: &Arc<RenderDevice>,
    cache: vk::PipelineCache,
    program: &Arc<Program>,
    layout: RenderAttachmentLayoutDesc<'a>,
    streams: &[InputVertexStreamLayout<'a>],
    specialization: &[(u32, SpecializationValue)],
    desc: RasterPipelineCreateDesc,
) -> Result<vk::Pipeline, Error> {
    let mut specialization_values = Cursor::new(Vec::new());
    let specialization_entires = specialization
        .iter()
        .map(|(index, value)| {
            let offset = specialization_values.stream_position().unwrap();
            let size = value.size();
            value.write(&mut specialization_values).unwrap();
            vk::SpecializationMapEntry::default()
                .constant_id(*index)
                .offset(offset as _)
                .size(size)
        })
        .collect::<Vec<_>>();
    let specialization_values = specialization_values.into_inner();
    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(&specialization_entires)
        .data(&specialization_values);

    let shader_create_info = program
        .shaders
        .iter()
        .map(|(shader, stage, entry)| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(*stage)
                .module(*shader)
                .specialization_info(&specialization_info)
                .name(entry)
        })
        .collect::<Vec<_>>();

    let streams = streams
        .iter()
        .enumerate()
        .map(|(index, stream)| stream.build(index as u32))
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

    let assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

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

    let mut rendering_info = layout.build();

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_create_info)
        .dynamic_state(&dynamic_state_create_info)
        .viewport_state(&viewport_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&blending_state)
        .input_assembly_state(&assembly_state_create_info)
        .rasterization_state(&rasterizer_state)
        .depth_stencil_state(&depthstencil_state)
        .vertex_input_state(&vertex_input)
        .push_next(&mut rendering_info);

    let pipeline = unsafe {
        device
            .raw
            .create_graphics_pipelines(cache, slice::from_ref(&pipeline_create_info), None)
    }?[0];

    Ok(pipeline)
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
        let pdevice = &device.physical_device;
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
    let pdevice = &device.physical_device;
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

    let cache = match unsafe { device.raw.create_pipeline_cache(&create_info, None) } {
        Ok(cache) => cache,
        Err(_) => {
            // Failed with initial data - so create empty cache.
            warn!("Failed to load pipeline cache. Create new one.");
            let create_info = vk::PipelineCacheCreateInfo::default();
            unsafe { device.raw.create_pipeline_cache(&create_info, None) }
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
    let data = unsafe { device.raw.get_pipeline_cache_data(cache) }.map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get pipeline cache data from device: {:?}", err),
        )
    })?;
    create_dir_all(path.as_ref().parent().unwrap())?;
    PipelineDiskCache::new(device, &data).save(File::create(path)?)
}
