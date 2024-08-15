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

mod vulkan;

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
#[allow(non_camel_case_types)]
pub enum Format {
    #[default]
    INVALID,
    R8_UNORM,
    R8_SNORM,
    R8_USCALED,
    R8_SSCALED,
    R8_UINT,
    R8_SINT,
    R8_SRGB,
    RG8_UNORM,
    RG8_SNORM,
    RG8_USCALED,
    RG8_SSCALED,
    RG8_UINT,
    RG8_SINT,
    RG8_SRGB,
    RGB8_UNORM,
    RGB8_SNORM,
    RGB8_USCALED,
    RGB8_SSCALED,
    RGB8_UINT,
    RGB8_SINT,
    RGB8_SRGB,
    RGBA8_UNORM,
    RGBA8_SNORM,
    RGBA8_USCALED,
    RGBA8_SSCALED,
    RGBA8_UINT,
    RGBA8_SINT,
    RGBA8_SRGB,
    BGR8_UNORM,
    BGR8_SNORM,
    BGR8_USCALED,
    BGR8_SSCALED,
    BGR8_UINT,
    BGR8_SINT,
    BGR8_SRGB,
    BGRA8_UNORM,
    BGRA8_SNORM,
    BGRA8_USCALED,
    BGRA8_SSCALED,
    BGRA8_UINT,
    BGRA8_SINT,
    BGRA8_SRGB,
    R16_UNORM,
    R16_SNORM,
    R16_USCALED,
    R16_SSCALED,
    R16_UINT,
    R16_SINT,
    R16_SFLOAT,
    RG16_UNORM,
    RG16_SNORM,
    RG16_USCALED,
    RG16_SSCALED,
    RG16_UINT,
    RG16_SINT,
    RG16_SFLOAT,
    RGB16_UNORM,
    RGB16_SNORM,
    RGB16_USCALED,
    RGB16_SSCALED,
    RGB16_UINT,
    RGB16_SINT,
    RGB16_SFLOAT,
    RGBA16_UNORM,
    RGBA16_SNORM,
    RGBA16_USCALED,
    RGBA16_SSCALED,
    RGBA16_UINT,
    RGBA16_SINT,
    RGBA16_SFLOAT,
    R32_UINT,
    R32_SINT,
    R32_SFLOAT,
    RG32_UINT,
    RG32_SINT,
    RG32_SFLOAT,
    RGB32_UINT,
    RGB32_SINT,
    RGB32_SFLOAT,
    RGBA32_UINT,
    RGBA32_SINT,
    RGBA32_SFLOAT,
    D16,
    D24,
    D32,
    D16_S8,
    D24_S8,
    BC1_RGB_UNORM,
    BC1_RGB_SRGB,
    BC1_RGBA_UNORM,
    BC1_RGBA_SRGB,
    BC2_UNORM,
    BC2_SRGB,
    BC3_UNORM,
    BC3_SRGB,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC6_UFLOAT,
    BC6_SFLOAT,
    BC7_UNORM,
    BC7_SRGB,
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
pub enum ImageType {
    Type1D,
    #[default]
    Type2D,
    Type3D,
}

bitflags! {
    #[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
    pub struct ImageUsage: u32 {
        const None = 0;
        const Sampled = 1;
        const Storage = 2;
        const ColorTarget = 4;
        const DepthStencilTarget = 8;
        const TransferDestination = 16;
        const TransferSource = 32;
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageLayout {
    Undefined,
    ShaderRead,
    ColorTarget,
    DepthStencilTarget,
    DepthStencilRead,
    TransferDestination,
    TransferSource,
    Present,
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
pub enum ImageViewType {
    Type1D,
    Type1DArray,
    #[default]
    Type2D,
    Type2DArray,
    Type3D,
}

bitflags! {
    #[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
    pub struct ImageAspect: u32 {
        const None = 0;
        const Color = 1;
        const Depth = 2;
        const Stencil = 4;
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
pub enum ImageMultisampling {
    #[default]
    None,
    Multisampling2,
    Multisampling4,
    Multisampling8,
}

bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        const Vertex = 1;
        const Index = 2;
        const Uniform = 4;
        const Storage = 8;
        const Destination = 16;
        const Source = 32;
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CullMode {
    Front,
    Back,
    FrontAndBack,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DepthCompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub enum RenderTargetLoadOp {
    Clear,
    Load,
    #[default]
    Discard,
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub enum RenderTargetStoreOp {
    Store,
    #[default]
    Discard,
}

bitflags! {
    #[derive(Debug, Default, Hash, Eq, PartialEq, Clone, Copy)]
    pub struct ShaderStage: u32 {
        const None = 0;
        const Vertex = 1;
        const Fragment = 2;
        const Graphics = 3;
        const Compute = 4;
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum BindType {
    Uniform,
    DynamicUniform,
    Storage,
    DynamicStorage,
    SampledImage,
    CombinedSampledImage,
    Sampler,
}

use bitflags::bitflags;
use speedy::{Readable, Writable};

pub use vulkan::Error;
pub use vulkan::{
    BindGroupDesc, BindGroupHandle, BindGroupSlotDesc, BindGroupUpdateContext, BufferCreateDesc,
    BufferHandle, BufferSlice, ClearRenderTarget, DrawStream, FrameState, ImageCreateDesc,
    ImageDesc, ImageHandle, InputVertexStreamAttrubute, InputVertexStreamDesc, Instance,
    InstanceBuilder, PhysicalDevice, PhysicalDeviceType, PipelineHandle, ProgramHandle,
    RasterPipelineCreateDesc, RenderContext, RenderDevice, RenderPassHandle, RenderPassLayout,
    RenderTarget, RenderTargetDesc, SkipMissingSlots, SubpassLayout, Surface, Swapchain,
    DYNAMIC_BINDING_SLOT, FRAME_BINDING_SLOT, MATERIAL_BINDING_SLOT, OBJECT_BINDING_SLOT,
};
