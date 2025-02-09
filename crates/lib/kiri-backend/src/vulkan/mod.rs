mod buffer;
mod descriptors;
mod drop_list;
mod frame;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod render_device;
mod staging;
mod swapchain;

pub use ash;
use ash::vk;
use buffer::*;
pub(crate) use descriptors::*;
use drop_list::*;
pub use frame::*;
use image::*;
pub use instance::*;
use parking_lot::{MutexGuard, RwLockWriteGuard};
pub use physical_device::*;
use pipeline::*;
pub use render_device::*;
use staging::*;
pub use swapchain::*;

use crate::Error;

pub(crate) type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
pub(crate) type GpuMemoryBlock = gpu_alloc::MemoryBlock<vk::DeviceMemory>;
pub(crate) type GpuDescriptorAllocator =
    gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>;
pub(crate) type GpuDescriptor = gpu_descriptor::DescriptorSet<vk::DescriptorSet>;

#[derive(Debug, Clone, Copy)]
pub struct ImageUploadData<'a> {
    pub data: &'a [u8],
}

impl<'a> ImageUploadData<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorDesc<'a> {
    pub name: &'a str,
    pub ty: vk::DescriptorType,
    pub count: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorLayoutDesc<'a> {
    pub layout: &'a [(usize, DescriptorDesc<'a>)],
    pub compute_groups_size: Option<(u32, u32, u32)>,
}

impl<'a> DescriptorLayoutDesc<'a> {
    pub fn has_slot(&self, index: usize) -> bool {
        self.layout.iter().any(|(x, _)| *x == index)
    }

    pub fn get_slot(&self, name: &str) -> Option<usize> {
        self.layout
            .iter()
            .find_map(|(slot, desc)| (desc.name == name).then_some(*slot))
    }

    pub fn get_desc(&self, slot: usize) -> Option<&DescriptorDesc> {
        self.layout
            .iter()
            .find_map(|(x, data)| if slot == *x { Some(data) } else { None })
    }

    pub fn get_layout(&self) -> &[(usize, DescriptorDesc<'a>)] {
        self.layout
    }

    pub fn by_types(
        &self,
        ty: &'a [vk::DescriptorType],
    ) -> impl Iterator<Item = (usize, DescriptorDesc)> {
        self.layout
            .iter()
            .copied()
            .filter(move |x| ty.contains(&x.1.ty))
    }
}

pub const EMPTY_DESCRIPTOR_LAYOUT: DescriptorLayoutDesc = DescriptorLayoutDesc {
    layout: &[],
    compute_groups_size: None,
};

#[derive(Debug, Clone, Copy)]
pub struct BufferCreateDesc<'a> {
    size: usize,
    usage: vk::BufferUsageFlags,
    memory_usage: gpu_alloc::UsageFlags,
    name: Option<&'a str>,
    dedicated: bool,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn host(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::HOST_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn upload(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::HOST_ACCESS | gpu_alloc::UsageFlags::UPLOAD,
            name: None,
            dedicated: false,
        }
    }

    pub fn shared(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                | gpu_alloc::UsageFlags::HOST_ACCESS,
            name: None,
            dedicated: false,
        }
    }

    pub fn index_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        self
    }

    pub fn veretex_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        self
    }

    pub fn storage_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        self
    }

    pub fn uniform_buffer(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        self
    }

    pub fn transfer_destination(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        self
    }

    pub fn transfer_source(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        self
    }

    pub fn indirect_draw(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        self
    }

    pub fn name(mut self, value: &'a str) -> Self {
        self.name = Some(value);
        self
    }

    pub fn dedicated(mut self) -> Self {
        self.dedicated = true;
        self
    }

    fn build(&self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo::default()
            .usage(self.usage)
            .size(self.size as _)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
    pub ty: Option<vk::ImageViewType>,
    pub format: Option<vk::Format>,
    pub aspect: vk::ImageAspectFlags,
    pub base_mip_level: usize,
    pub level_count: Option<usize>,
}

impl ImageViewDesc {
    pub fn new(aspect: vk::ImageAspectFlags) -> Self {
        Self {
            ty: None,
            format: None,
            aspect,
            base_mip_level: 0,
            level_count: None,
        }
    }

    pub fn color() -> Self {
        Self::new(vk::ImageAspectFlags::COLOR)
    }

    pub fn depth() -> Self {
        Self::new(vk::ImageAspectFlags::DEPTH)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageCreateDesc<'a> {
    pub dims: [usize; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub mip_levels: usize,
    pub array_elements: usize,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    pub flags: vk::ImageCreateFlags,
    pub tiling: vk::ImageTiling,
    pub initial_layout: Option<vk::ImageLayout>,
}

impl<'a> ImageCreateDesc<'a> {
    pub fn new(format: vk::Format, dims: [usize; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::empty(),
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
            initial_layout: None,
        }
    }

    pub fn texture(format: vk::Format, dims: [usize; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
            initial_layout: None,
        }
    }

    pub fn cubemap(format: vk::Format, dims: [usize; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 6,
            dedicated: false,
            name: None,
            initial_layout: None,
        }
    }

    pub fn color_target(format: vk::Format, dims: [usize; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
            initial_layout: Some(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        }
    }

    pub fn depth_stencil_target(format: vk::Format, dims: [usize; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
            initial_layout: Some(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        }
    }

    pub fn transfer_desitnation(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_DST;
        self
    }

    pub fn trasfer_source(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        self
    }

    pub fn storage(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::STORAGE;
        self
    }

    pub fn ty(mut self, value: vk::ImageType) -> Self {
        self.ty = value;
        self
    }

    pub fn usage(mut self, value: vk::ImageUsageFlags) -> Self {
        self.usage = value;
        self
    }

    pub fn sampled(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::SAMPLED;
        self
    }

    pub fn samples(mut self, value: vk::SampleCountFlags) -> Self {
        self.samples = value;
        self
    }

    pub fn mip_levels(mut self, value: usize) -> Self {
        self.mip_levels = value;
        self
    }

    pub fn array_elements(mut self, value: usize) -> Self {
        self.array_elements = value;
        self
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    pub fn transient(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        self
    }

    pub fn initial_layout(mut self, value: vk::ImageLayout) -> Self {
        self.initial_layout = Some(value);
        self
    }
}

pub const MAX_COLOR_ATTACHMENTS: usize = 8;
pub const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPassLayout<'a> {
    pub color: &'a [vk::Format],
    pub depth: Option<vk::Format>,
}

impl<'a> RenderPassLayout<'a> {
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
            depth_test: Some(vk::CompareOp::LESS_OR_EQUAL),
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
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC_ALPHA,
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

    pub fn depth_write(mut self, value: bool) -> Self {
        self.depth_write = value;

        self
    }

    pub fn depth_test(mut self, value: vk::CompareOp) -> Self {
        self.depth_test = Some(value);

        self
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexAttrubute {
    pub location: usize,
    pub format: vk::Format,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexStreamLayout<'a> {
    pub streams: &'a [InputVertexAttrubute],
    pub stride: usize,
}

pub const PASS_DESCRIPTOR_SLOT_INDEX: usize = 0;
pub const OBJECT_DESCRIPTOR_SLOT_INDEX: usize = 1;
pub const MATERIAL_DESCRIPTOR_SLOT_IDNEX: usize = 2;
pub const DYNAMIC_DESCRIPTOR_SLOT_INDEX: usize = 3;
pub const MAX_DESCRIPTOR_SETS: usize = 4;

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ShaderDesc<'a> {
    pub stage: vk::ShaderStageFlags,
    pub entry: &'a str,
    pub code: &'a [u8],
}

impl<'a> ShaderDesc<'a> {
    pub fn new(stage: vk::ShaderStageFlags, code: &'a [u8]) -> Self {
        Self {
            stage,
            entry: "main",
            code,
        }
    }

    pub fn vertex(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::VERTEX,
            entry: "main",
            code,
        }
    }

    pub fn fragment(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::FRAGMENT,
            entry: "main",
            code,
        }
    }

    pub fn compute(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::COMPUTE,
            entry: "main",
            code,
        }
    }

    pub fn entry(mut self, entry: &'a str) -> Self {
        self.entry = entry;
        self
    }
}

#[derive(Debug, Default)]
pub struct DescriptorSetCreateDesc<'a> {
    pub layout: DescriptorLayoutDesc<'static>,
    pub stages: vk::ShaderStageFlags,
    pub images: &'a [ImageHandle],
    pub unifoms: &'a [BufferSlice],
    pub storages: &'a [BufferSlice],
    pub dynamic_uniforms: &'a [(BufferHandle, usize)],
    pub dynamic_storage_buffers: &'a [(BufferHandle, usize)],
    pub name: Option<&'a str>,
}

pub struct DescriptorUpdateContext<'a> {
    device: &'a RenderDevice,
    descriptors: RwLockWriteGuard<'a, DescriptorPool>,
    dirty: MutexGuard<'a, Vec<DescriptorHandle>>,
    to_destroy: MutexGuard<'a, Vec<DescriptorHandle>>,
}

impl DescriptorUpdateContext<'_> {
    pub fn create_descriptor(
        &mut self,
        builder: DescriptorSetCreateDesc,
    ) -> Result<DescriptorHandle, Error> {
        let data = builder.build(self.device)?;
        let handle = self.descriptors.push(vk::DescriptorSet::null(), data);
        self.dirty.push(handle);
        Ok(handle)
    }

    pub fn destroy_descriptor(&mut self, handle: DescriptorHandle) {
        self.to_destroy.push(handle);
    }
}

impl RenderDevice {
    pub fn descriptors(&self) -> DescriptorUpdateContext {
        DescriptorUpdateContext {
            device: self,
            descriptors: self.descriptors.write(),
            dirty: self.dirty_descriptors.lock(),
            to_destroy: self.descriptors_to_destroy.lock(),
        }
    }
}

pub struct RenderResourceResolver<'a> {
    device: &'a ash::Device,
    buffers: &'a BufferPool,
    images: &'a ImagePool,
    raster_pipelines: &'a RasterPipelinePool,
    descriptors: &'a DescriptorPool,
    pub(crate) empty_descriptor_set: vk::DescriptorSet,
    pub(crate) backbuffer: &'a ImageData,
}

impl<'a> RenderResourceResolver<'a> {
    fn new(
        device: &'a ash::Device,
        backbuffer: &'a ImageData,
        buffers: &'a BufferPool,
        images: &'a ImagePool,
        raster_pipelines: &'a RasterPipelinePool,
        descriptors: &'a DescriptorPool,
        empty_descriptor_set: vk::DescriptorSet,
    ) -> Self {
        Self {
            device,
            buffers,
            images,
            raster_pipelines,
            descriptors,
            empty_descriptor_set,
            backbuffer,
        }
    }

    pub fn resolve_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, Error> {
        self.buffers
            .get(handle)
            .copied()
            .ok_or(Error::InvalidBufferHandle(handle))
    }

    pub fn resolve_image_view(
        &self,
        handle: ImageHandle,
        desc: ImageViewDesc,
    ) -> Result<vk::ImageView, Error> {
        Ok(self
            .images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .get_or_create_view(&self.device, desc)?)
    }

    pub fn resolve_image(&self, handle: ImageHandle) -> Result<vk::Image, Error> {
        Ok(self
            .images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .raw)
    }

    pub fn resolve_image_desc(&self, handle: ImageHandle) -> Result<&ImageDesc, Error> {
        Ok(&self
            .images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .desc)
    }

    pub fn resolve_raster_pipeline(
        &self,
        handle: RasterPipelineHandle,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
        let pipeline = self
            .raster_pipelines
            .get(handle.0 as usize)
            .ok_or(Error::InvalidRasterPipelineHandle(handle))?;
        match pipeline {
            Pipeline::Pending(_) => panic!("Pipeline isn't compiled yet"),
            Pipeline::Compiled(compiled_pipeline) => Ok((
                compiled_pipeline.pipeline,
                compiled_pipeline.pipeline_layout,
            )),
        }
    }

    pub fn resolve_descriptor_set(
        &self,
        handle: DescriptorHandle,
    ) -> Result<vk::DescriptorSet, Error> {
        self.descriptors
            .get(handle)
            .copied()
            .ok_or(Error::InvalidDescriptorHandle(handle))
    }
}
