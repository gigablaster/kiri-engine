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

use ash::vk::{self, ImageSubresourceRange};
use kiri_backend::Image;

use crate::{Error, ImageHandle, ImagePool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageBarrierType {
    SampledToWriteColor,
    WriteColorToWriteColor,
    WriteColorToSampled,
    DiscardToWriteColor,
    WriteDepthToWriteDepth,
    WriteDepthToSampled,
    DiscardToWriteDepth,
    WriteDepthToReadDepth,
    ReadDepthToSampled,
    ReadDepthToReadDepth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ImageBarrier(
    pub ImageHandle,
    pub ImageBarrierType,
    pub vk::ImageAspectFlags,
);

impl ImageBarrierType {
    pub fn build(self, image: &Image, aspect: vk::ImageAspectFlags) -> vk::ImageMemoryBarrier2 {
        let (old, new) = self.layouts();
        let (src_stage, dst_stage) = self.stages();
        let (src_access, dst_access) = self.access();
        vk::ImageMemoryBarrier2::default()
            .image(image.raw)
            .subresource_range(ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
            .old_layout(old)
            .new_layout(new)
            .src_stage_mask(src_stage)
            .dst_stage_mask(dst_stage)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
    }

    pub fn layouts(self) -> (vk::ImageLayout, vk::ImageLayout) {
        match self {
            ImageBarrierType::SampledToWriteColor => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ),
            ImageBarrierType::WriteColorToWriteColor => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ),
            ImageBarrierType::WriteColorToSampled => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            ImageBarrierType::DiscardToWriteColor => (
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ),
            ImageBarrierType::WriteDepthToWriteDepth => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
            ImageBarrierType::WriteDepthToSampled => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            ImageBarrierType::DiscardToWriteDepth => (
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
            ImageBarrierType::WriteDepthToReadDepth => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ),
            ImageBarrierType::ReadDepthToSampled => (
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            ImageBarrierType::ReadDepthToReadDepth => (
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ),
        }
    }

    pub fn stages(self) -> (vk::PipelineStageFlags2, vk::PipelineStageFlags2) {
        match self {
            ImageBarrierType::SampledToWriteColor => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            ),
            ImageBarrierType::WriteColorToWriteColor => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            ),
            ImageBarrierType::WriteColorToSampled => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ),
            ImageBarrierType::DiscardToWriteColor => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            ),
            ImageBarrierType::WriteDepthToWriteDepth => (
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            ),
            ImageBarrierType::WriteDepthToSampled => (
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ),
            ImageBarrierType::DiscardToWriteDepth => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            ),
            ImageBarrierType::WriteDepthToReadDepth => (
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            ),
            ImageBarrierType::ReadDepthToSampled => (
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ),
            ImageBarrierType::ReadDepthToReadDepth => (
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            ),
        }
    }

    fn access(self) -> (vk::AccessFlags2, vk::AccessFlags2) {
        match self {
            ImageBarrierType::SampledToWriteColor => (
                vk::AccessFlags2::SHADER_READ,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            ImageBarrierType::WriteColorToWriteColor => (
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            ImageBarrierType::WriteColorToSampled => (
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::SHADER_READ,
            ),
            ImageBarrierType::DiscardToWriteColor => (
                vk::AccessFlags2::SHADER_READ,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            ImageBarrierType::WriteDepthToWriteDepth => (
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            ImageBarrierType::WriteDepthToSampled => (
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::AccessFlags2::SHADER_READ,
            ),
            ImageBarrierType::DiscardToWriteDepth => (
                vk::AccessFlags2::SHADER_READ,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            ImageBarrierType::WriteDepthToReadDepth => (
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            ),
            ImageBarrierType::ReadDepthToSampled => (
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                vk::AccessFlags2::SHADER_READ,
            ),
            ImageBarrierType::ReadDepthToReadDepth => (
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            ),
        }
    }
}
pub(crate) fn record_barriers(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    images: &ImagePool,
    image_barriers: &[ImageBarrier],
) -> Result<(), Error> {
    let image_barriers = image_barriers
        .iter()
        .map(|x| {
            let image = images.get(x.0).unwrap();
            x.1.build(image, x.2)
        })
        .collect::<Vec<_>>();
    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .image_memory_barriers(&image_barriers)
                .dependency_flags(vk::DependencyFlags::BY_REGION),
        )
    };
    Ok(())
}
