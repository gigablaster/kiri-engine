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
};

use ash::vk::{self, CompareOp};
use bevy_tasks::ComputeTaskPool;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use kiri_common::Align;
use log::{info, warn};
use uuid::Uuid;

use crate::{
    BlendFactor, BlendOp, CullMode, DepthCompareOp, Error, Format, PhysicalDevice, PipelineHandle,
    ProgramHandle, RenderDevice, RenderTargetLoadOp, RenderTargetStoreOp,
};

use super::{PipelineCompilationContext, RenderPassHandle};

impl From<BlendFactor> for vk::BlendFactor {
    fn from(value: BlendFactor) -> Self {
        match value {
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
            BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        }
    }
}

impl From<BlendOp> for vk::BlendOp {
    fn from(value: BlendOp) -> Self {
        match value {
            BlendOp::Add => vk::BlendOp::ADD,
            BlendOp::Subtract => vk::BlendOp::SUBTRACT,
            BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Min => vk::BlendOp::MIN,
            BlendOp::Max => vk::BlendOp::MAX,
        }
    }
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(value: CullMode) -> Self {
        match value {
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::Back => vk::CullModeFlags::BACK,
            CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
        }
    }
}

impl From<DepthCompareOp> for vk::CompareOp {
    fn from(value: DepthCompareOp) -> Self {
        match value {
            DepthCompareOp::Never => vk::CompareOp::NEVER,
            DepthCompareOp::Less => vk::CompareOp::LESS,
            DepthCompareOp::Equal => vk::CompareOp::EQUAL,
            DepthCompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            DepthCompareOp::Greater => vk::CompareOp::GREATER,
            DepthCompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            DepthCompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            DepthCompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

impl From<RenderTargetLoadOp> for vk::AttachmentLoadOp {
    fn from(value: RenderTargetLoadOp) -> Self {
        match value {
            RenderTargetLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            RenderTargetLoadOp::Load => vk::AttachmentLoadOp::LOAD,
            RenderTargetLoadOp::Discard => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

impl From<RenderTargetStoreOp> for vk::AttachmentStoreOp {
    fn from(value: RenderTargetStoreOp) -> Self {
        match value {
            RenderTargetStoreOp::Store => vk::AttachmentStoreOp::STORE,
            RenderTargetStoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
        }
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexAttrubute {
    pub format: Format,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexStreamLayout<'a> {
    pub streams: &'a [InputVertexAttrubute],
}

pub trait PipelineVertex {
    fn layout() -> &'static [InputVertexStreamLayout<'static>];
}

impl<'a> InputVertexStreamLayout<'a> {
    fn build(&self, binding: usize) -> (u32, Vec<vk::VertexInputAttributeDescription>) {
        let stride = self
            .streams
            .iter()
            .map(|x| x.offset + x.format.size_in_bytes().align(4))
            .max()
            .unwrap();
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

        (stride as u32, attributes)
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

#[derive(Debug, Clone)]
pub(crate) struct CompilePipelineData {
    pub program: ProgramHandle,
    pub pass: RenderPassHandle,
    pub subpass: u32,
    pub streams: &'static [InputVertexStreamLayout<'static>],
    pub desc: RasterPipelineCreateDesc,
}

fn compile_raster_pipeline(
    context: &PipelineCompilationContext,
    data: CompilePipelineData,
    cache: vk::PipelineCache,
) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
    let program = context
        .resolve_program(data.program)
        .ok_or(Error::InvalidProgramHandle(data.program))?;
    let render_pass = context
        .resolve_render_pass(data.pass)
        .ok_or(Error::InvalidRenderPassHandle(data.pass))?;
    let shader_create_info = program
        .shaders
        .iter()
        .map(|shader| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(shader.stage())
                .module(shader.raw)
                .name(shader.entry())
        })
        .collect::<Vec<_>>();

    let assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let streams = data
        .streams
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
        .cull_mode(data.desc.cull.unwrap_or(vk::CullModeFlags::NONE))
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let mut depthstencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .stencil_test_enable(false)
        .depth_write_enable(data.desc.depth_write);

    if let Some(depth_compare) = data.desc.depth_test {
        depthstencil_state = depthstencil_state
            .depth_test_enable(true)
            .depth_compare_op(depth_compare);
    }

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blend_attachment = if let Some((color, alpha)) = data.desc.blend {
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
        .layout(program.pipeline_layout)
        .stages(&shader_create_info)
        .vertex_input_state(&vertex_input)
        .dynamic_state(&dynamic_state_create_info)
        .viewport_state(&viewport_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&blending_state)
        .input_assembly_state(&assembly_state_create_info)
        .rasterization_state(&rasterizer_state)
        .depth_stencil_state(&depthstencil_state)
        .render_pass(render_pass.raw)
        .subpass(data.subpass);

    let pipeline = unsafe {
        context.device.create_graphics_pipelines(
            cache,
            slice::from_ref(&pipeline_create_info),
            None,
        )
    }?[0];

    Ok((pipeline, program.pipeline_layout()))
}

impl RenderDevice {
    /// Create pipeline
    ///
    /// Pipeline will be compiled right before next frame
    pub fn create_pipeline(
        &self,
        program: ProgramHandle,
        pass: RenderPassHandle,
        subpass: u32,
        streams: &'static [InputVertexStreamLayout<'static>],
        desc: &RasterPipelineCreateDesc,
    ) -> PipelineHandle {
        let handle = {
            let mut pipelines = self.pipelines.write();
            pipelines.push((vk::Pipeline::null(), vk::PipelineLayout::null()))
        };
        self.pipelines_to_compile.lock().insert(
            handle,
            CompilePipelineData {
                program,
                pass,
                subpass,
                streams,
                desc: *desc,
            },
        );
        handle
    }

    pub fn destory_pipeline(&self, handle: PipelineHandle) {
        if let Some(pipeline) = self.pipelines.write().remove(handle) {
            unsafe { self.device.destroy_pipeline(pipeline.0, None) }
        }
    }

    pub(crate) async fn compile_pipeline<'a>(
        context: &PipelineCompilationContext<'a>,
        handle: PipelineHandle,
        data: CompilePipelineData,
        cache: vk::PipelineCache,
    ) -> Result<(PipelineHandle, vk::Pipeline, vk::PipelineLayout), Error> {
        let (pipeline, layout) = compile_raster_pipeline(context, data, cache)?;
        Ok((handle, pipeline, layout))
    }

    pub(crate) async fn compile_all_pipelines(&self) -> Result<(), Error> {
        puffin::profile_function!();
        let programs = self.programs.read();
        let render_passes = self.render_passes.read();
        let context = PipelineCompilationContext {
            device: &self.device,
            programs: &programs,
            render_passes: &render_passes,
        };

        let compiled = ComputeTaskPool::get().scope(|s| {
            self.pipelines_to_compile
                .lock()
                .drain()
                .for_each(|(handle, data)| {
                    s.spawn(Self::compile_pipeline(&context, handle, data, self.cache))
                })
        });
        let mut pipelines = self.pipelines.write();
        for result in compiled {
            let (handle, pipeline, layout) = result?;
            if let Some(target) = pipelines.get_mut(handle) {
                *target = (pipeline, layout);
            } else {
                unsafe { self.device.destroy_pipeline(pipeline, None) };
            }
        }
        Ok(())
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
    uuid: Uuid,
    data: Vec<u8>,
}

impl PipelineDiskCache {
    pub fn new(pdevice: &PhysicalDevice, data: &[u8]) -> Self {
        let vendor_id = pdevice.properties.vendor_id;
        let device_id = pdevice.properties.device_id;
        let driver_version = pdevice.properties.driver_version;
        let uuid = Uuid::from_bytes(pdevice.properties.pipeline_cache_uuid);

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
        Ok(Self {
            vendor_id: r.read_u32::<LittleEndian>()?,
            device_id: r.read_u32::<LittleEndian>()?,
            driver_version: r.read_u32::<LittleEndian>()?,
            uuid: Uuid::from_u128(r.read_u128::<LittleEndian>()?),
            data: Self::read_data(&mut r)?,
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
        w.write_u128::<LittleEndian>(self.uuid.as_u128())?;
        w.write_u32::<LittleEndian>(self.data.len() as _)?;
        w.write_all(&self.data)?;
        Ok(())
    }
}

pub(crate) fn load_or_create_pipeline_cache<P: AsRef<Path>>(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
    path: P,
) -> Result<vk::PipelineCache, Error> {
    info!("Loading pipeline cache from {:?}", path.as_ref());
    let data = if let Ok(file) = File::open(path) {
        if let Ok(cache) = PipelineDiskCache::read(file) {
            if cache.vendor_id == pdevice.properties.vendor_id
                && cache.device_id == pdevice.properties.device_id
                && cache.driver_version == pdevice.properties.driver_version
                && cache.uuid == Uuid::from_bytes(pdevice.properties.pipeline_cache_uuid)
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

    let cache = match unsafe { device.create_pipeline_cache(&create_info, None) } {
        Ok(cache) => cache,
        Err(_) => {
            // Failed with initial data - so create empty cache.
            warn!("Failed to load pipeline cache. Create new one.");
            let create_info = vk::PipelineCacheCreateInfo::default();
            unsafe { device.create_pipeline_cache(&create_info, None) }?
        }
    };

    Ok(cache)
}

pub(crate) fn save_pipeline_cache<P: AsRef<Path>>(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
    cache: vk::PipelineCache,
    path: P,
) -> io::Result<()> {
    info!("Saving pipeline cache to {:?}", path.as_ref());
    let data = unsafe { device.get_pipeline_cache_data(cache) }.map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get pipeline cache data from device: {:?}", err),
        )
    })?;
    create_dir_all(path.as_ref().parent().unwrap())?;
    PipelineDiskCache::new(pdevice, &data).save(File::create(path)?)
}
