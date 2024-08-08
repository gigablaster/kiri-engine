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
    fs::File,
    io::{self},
    path::Path,
    slice,
};

use arrayvec::ArrayVec;
use ash::vk::{self};
use bevy_tasks::ComputeTaskPool;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use uuid::Uuid;

use crate::{
    Error, ImageHandle, PhysicalDevice, PipelineCompilationContext, PipelineHandle, ProgramHandle,
    RenderContext,
};

use super::ImagePool;

#[derive(Debug, Clone, Copy)]
pub enum ClearRenderTarget {
    None,
    Color([f32; 4]),
    DepthStencil(f32, u32),
}

impl From<ClearRenderTarget> for vk::ClearValue {
    fn from(value: ClearRenderTarget) -> Self {
        match value {
            ClearRenderTarget::Color(color) => vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            },
            ClearRenderTarget::DepthStencil(depth, stencil) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
            ClearRenderTarget::None => vk::ClearValue::default(),
        }
    }
}

pub(crate) const MAX_COLOR_ATTACHMENTS: usize = 8;
pub(crate) const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Copy)]
pub struct RenderTarget {
    pub image: ImageHandle,
    pub layout: vk::ImageLayout,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear: ClearRenderTarget,
}

impl RenderTarget {
    pub fn color(image: ImageHandle) -> Self {
        Self {
            image,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            clear: ClearRenderTarget::None,
        }
    }

    pub fn depth(image: ImageHandle) -> Self {
        Self {
            image,
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            clear: ClearRenderTarget::None,
        }
    }

    pub fn discard(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    pub fn clear(mut self, color: ClearRenderTarget) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear = color;
        self
    }

    pub fn load(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::LOAD;
        self
    }

    pub(crate) fn build(&self, images: &ImagePool) -> Result<vk::RenderingAttachmentInfo, Error> {
        let view = images
            .get(self.image)
            .ok_or(Error::InvalidImageHandle(self.image))?;
        let info = vk::RenderingAttachmentInfo::default()
            .image_view(*view)
            .image_layout(self.layout)
            .load_op(self.load_op)
            .store_op(self.store_op)
            .clear_value(self.clear.into());
        Ok(info)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RenderPassLayout<'a> {
    pub color: &'a [vk::Format],
    pub depth: Option<vk::Format>,
}

impl<'a> RenderPassLayout<'a> {
    fn build(&self) -> vk::PipelineRenderingCreateInfo<'a> {
        let mut info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&self.color);
        if let Some(depth) = self.depth {
            info.depth_attachment_format = depth;
        }
        info
    }

    pub(crate) fn inheretence(
        &self,
        flags: vk::RenderingFlags,
    ) -> vk::CommandBufferInheritanceRenderingInfo<'a> {
        let mut info = vk::CommandBufferInheritanceRenderingInfo::default()
            .color_attachment_formats(&self.color)
            .flags(flags);
        if let Some(depth) = self.depth {
            info.depth_attachment_format = depth;
        }
        info
    }
}

#[derive(Debug, Default)]
pub struct RenderPass {
    // pub layout: &'a RenderPassLayout<'a>,
    pub depth: Option<RenderTarget>,
    pub color: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
}

impl RenderPass {
    pub fn depth(mut self, depth: RenderTarget) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn color(mut self, color: RenderTarget) -> Self {
        self.color.push(color);
        self
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
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
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

fn compile_raster_pipeline(
    context: &PipelineCompilationContext,
    program: ProgramHandle,
    render_pass_layout: &RenderPassLayout,
    desc: &RasterPipelineCreateDesc,
) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
    let program = context
        .resolve_program(program)
        .ok_or(Error::InvalidProgramHandle(program))?;
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

    let mut rendering_info = render_pass_layout.build();

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .layout(program.pipeline_layout)
        .stages(&shader_create_info)
        .dynamic_state(&dynamic_state_create_info)
        .viewport_state(&viewport_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&blending_state)
        .input_assembly_state(&assembly_state_create_info)
        .rasterization_state(&rasterizer_state)
        .depth_stencil_state(&depthstencil_state)
        .push_next(&mut rendering_info);

    let pipeline = unsafe {
        context.device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            slice::from_ref(&pipeline_create_info),
            None,
        )
    }?[0];

    Ok((pipeline, program.pipeline_layout()))
}

impl<'game> RenderContext<'game> {
    /// Create pipeline
    ///
    /// Pipeline will be compiled right before next frame
    pub fn create_pipeline(
        &self,
        program: ProgramHandle,
        pass_layout: &RenderPassLayout<'static>,
        desc: RasterPipelineCreateDesc,
    ) -> PipelineHandle {
        let handle = {
            let mut pipelines = self.pipelines.write();
            let index = pipelines.len() as u32;
            pipelines.push((vk::Pipeline::null(), vk::PipelineLayout::null()));
            PipelineHandle(index)
        };
        self.pipelines_to_compile
            .lock()
            .insert(handle, (program, pass_layout.clone(), desc));
        handle
    }

    pub(crate) async fn compile_pipeline<'a>(
        context: &PipelineCompilationContext<'a>,
        handle: PipelineHandle,
        program: ProgramHandle,
        pass: &RenderPassLayout<'static>,
        desc: RasterPipelineCreateDesc,
    ) -> Result<(PipelineHandle, vk::Pipeline, vk::PipelineLayout), Error> {
        let (pipeline, layout) = compile_raster_pipeline(context, program, pass, &desc)?;
        Ok((handle, pipeline, layout))
    }

    pub(crate) async fn compile_all_pipelines(&self) -> Result<(), Error> {
        puffin::profile_function!();
        let programs = self.programs.read();
        let context = PipelineCompilationContext {
            device: &self.device,
            programs: &programs,
        };
        let to_compile = self.pipelines_to_compile.lock().drain().collect::<Vec<_>>();

        let compiled = ComputeTaskPool::get().scope(|s| {
            to_compile
                .iter()
                .for_each(|(handle, (program, pass, desc))| {
                    s.spawn(Self::compile_pipeline(
                        &context,
                        *handle,
                        *program,
                        pass,
                        desc.clone(),
                    ))
                })
        });
        let mut pipelines = self.pipelines.write();
        for result in compiled {
            let (handle, pipeline, layout) = result?;
            pipelines[handle.0 as usize] = (pipeline, layout);
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
    let data = if let Ok(cache) = PipelineDiskCache::read(File::open(path)?) {
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
    let data = unsafe { device.get_pipeline_cache_data(cache) }.map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get pipeline cache data from device: {:?}", err),
        )
    })?;
    PipelineDiskCache::new(pdevice, &data).save(File::create(path)?)
}
