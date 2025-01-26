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

use std::collections::HashMap;

use kiri_backend::{ash::vk, PipelineBlendDesc, RasterPipelineCreateDesc};
use serde::{Deserialize, Serialize};
use speedy::{Readable, Writable};

use crate::Asset;

#[derive(Debug, Clone, Copy, Readable, Writable, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, Readable, Writable, Serialize, Deserialize)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSibtract,
    Min,
    Max,
}

impl From<BlendOp> for vk::BlendOp {
    fn from(value: BlendOp) -> Self {
        match value {
            BlendOp::Add => vk::BlendOp::ADD,
            BlendOp::Subtract => vk::BlendOp::SUBTRACT,
            BlendOp::ReverseSibtract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Min => vk::BlendOp::MIN,
            BlendOp::Max => vk::BlendOp::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable, Serialize, Deserialize)]
pub enum Cull {
    None,
    Front,
    Back,
}

impl From<Cull> for vk::CullModeFlags {
    fn from(value: Cull) -> Self {
        match value {
            Cull::None => vk::CullModeFlags::NONE,
            Cull::Front => vk::CullModeFlags::FRONT,
            Cull::Back => vk::CullModeFlags::BACK,
        }
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable, Serialize, Deserialize)]
pub enum DepthCompare {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Grater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

impl From<DepthCompare> for vk::CompareOp {
    fn from(value: DepthCompare) -> Self {
        match value {
            DepthCompare::Never => vk::CompareOp::NEVER,
            DepthCompare::Less => vk::CompareOp::LESS,
            DepthCompare::Equal => vk::CompareOp::EQUAL,
            DepthCompare::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            DepthCompare::Grater => vk::CompareOp::GREATER,
            DepthCompare::NotEqual => vk::CompareOp::NOT_EQUAL,
            DepthCompare::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            DepthCompare::Always => vk::CompareOp::ALWAYS,
        }
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable, Serialize, Deserialize)]
pub struct Blend {
    pub src: BlendFactor,
    pub dst: BlendFactor,
    pub op: BlendOp,
}

impl From<Blend> for PipelineBlendDesc {
    fn from(value: Blend) -> Self {
        Self {
            src: value.src.into(),
            dst: value.dst.into(),
            op: value.op.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable, Serialize, Deserialize)]
pub struct BlendDesc {
    pub color: Blend,
    pub alpha: Blend,
}

#[derive(Debug, Default, Clone, Copy, Readable, Writable, Serialize, Deserialize)]
pub struct PipelineDesc {
    pub blend: Option<BlendDesc>,
    pub cull: Option<Cull>,
    pub depth_test: Option<DepthCompare>,
    pub depth_write: Option<bool>,
}

impl From<PipelineDesc> for RasterPipelineCreateDesc {
    fn from(value: PipelineDesc) -> Self {
        Self {
            blend: value
                .blend
                .map(|desc| (desc.color.into(), desc.alpha.into())),
            cull: value.cull.map(|cull| cull.into()),
            depth_test: value.depth_test.map(|depth_test| depth_test.into()),
            depth_write: value.depth_write.unwrap_or(true),
        }
    }
}

#[derive(Debug, Default, Readable, Writable, Serialize, Deserialize)]
pub struct Technique {
    pub desc: PipelineDesc,
    pub spec: Option<HashMap<u32, u32>>,
}

#[derive(Debug, Readable, Writable)]
pub struct EffectAsset {
    pub vertex_shader: Vec<u8>,
    pub fragment_shader: Vec<u8>,
    pub techniques: HashMap<String, Technique>,
}

impl Asset for EffectAsset {
    const TYPE: uuid::Uuid = uuid::uuid!("aee3d3a5-f61f-41f3-9eb5-c9a27c12377f");

    fn deserialize<R: std::io::Read>(r: R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }

    fn serialize<W: std::io::Write>(&self, w: W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }
}
