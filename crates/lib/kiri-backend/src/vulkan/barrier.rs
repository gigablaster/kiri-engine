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

use ash::vk;

use super::{Image, ImageHandle, ImagePool};

#[derive(Debug, Clone, Copy)]
pub enum ImageBarrierType {
    ToDepthRenderTarget,
    ToColorRenderTarget,
    ToPresent,
    FromRenderTarget,
    FromDepthTarget,
    DiscardRenderTarget,
    DiscardDepthTarget,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageBarrier {
    pub image: ImageHandle,
    pub ty: ImageBarrierType,
}

impl ImageBarrier {
    pub fn new(image: ImageHandle, ty: ImageBarrierType) -> Self {
        Self { image, ty }
    }
}

impl ImageBarrierType {
    fn to_vk(self, image: &Image) -> vk::ImageMemoryBarrier2 {
        match self {
            ImageBarrierType::ToColorRenderTarget => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .subresource_range(image.subresource(vk::ImageAspectFlags::COLOR)),
            ImageBarrierType::ToPresent => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .subresource_range(image.subresource(vk::ImageAspectFlags::COLOR)),
            ImageBarrierType::ToDepthRenderTarget => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER) // fixme?
                .subresource_range(image.subresource(vk::ImageAspectFlags::DEPTH)),
            ImageBarrierType::DiscardRenderTarget => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .subresource_range(image.subresource(vk::ImageAspectFlags::COLOR)),
            ImageBarrierType::DiscardDepthTarget => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER) // fixme?
                .subresource_range(image.subresource(vk::ImageAspectFlags::DEPTH)),
            ImageBarrierType::FromRenderTarget => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .subresource_range(image.subresource(vk::ImageAspectFlags::COLOR)),
            ImageBarrierType::FromDepthTarget => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER) // fixme?
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .subresource_range(image.subresource(vk::ImageAspectFlags::DEPTH)),
        }
    }
}

pub(crate) fn image_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    images: &ImagePool,
    barriers: &[ImageBarrier],
) {
    let to_apply = barriers
        .iter()
        .map(|x| {
            let image = images.get_cold(x.image).unwrap();
            x.ty.to_vk(image)
        })
        .collect::<Vec<_>>();
    unsafe {
        device.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default()
                .image_memory_barriers(&to_apply)
                .dependency_flags(vk::DependencyFlags::BY_REGION),
        )
    }
}
