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

use super::{Error, Image, ImageSubresourceRange};

#[derive(Debug, Clone, Copy)]
pub enum Barrier<'a> {
    ToDepthRenderTarget(&'a Image),
    ToColorRenderTarget(&'a Image),
    ToPresent(&'a Image),
    FromRenderTarget(&'a Image),
    FromDepthTarget(&'a Image),
    DiscardRenderTarget(&'a Image),
    DiscardDepthTarget(&'a Image),
}

impl<'a> From<Barrier<'a>> for vk::ImageMemoryBarrier2<'a> {
    fn from(value: Barrier) -> Self {
        match value {
            Barrier::ToColorRenderTarget(image) => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .subresource_range(
                    image.subresource(ImageSubresourceRange::All, vk::ImageAspectFlags::COLOR),
                ),
            Barrier::ToPresent(image) => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .subresource_range(
                    image.subresource(ImageSubresourceRange::All, vk::ImageAspectFlags::COLOR),
                ),
            Barrier::ToDepthRenderTarget(image) => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER) // fixme?
                .subresource_range(
                    image.subresource(ImageSubresourceRange::All, vk::ImageAspectFlags::DEPTH),
                ),
            Barrier::DiscardRenderTarget(image) => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .subresource_range(
                    image.subresource(ImageSubresourceRange::All, vk::ImageAspectFlags::COLOR),
                ),
            Barrier::DiscardDepthTarget(image) => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::SHADER_READ)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER) // fixme?
                .subresource_range(
                    image.subresource(ImageSubresourceRange::All, vk::ImageAspectFlags::DEPTH),
                ),
            Barrier::FromRenderTarget(image) => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .subresource_range(
                    image.subresource(ImageSubresourceRange::All, vk::ImageAspectFlags::COLOR),
                ),
            Barrier::FromDepthTarget(image) => vk::ImageMemoryBarrier2::default()
                .image(image.raw)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER) // fixme?
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .subresource_range(
                    image.subresource(ImageSubresourceRange::All, vk::ImageAspectFlags::DEPTH),
                ),
        }
    }
}

pub fn image_barrier<'a>(device: &ash::Device, cb: vk::CommandBuffer, barriers: &[Barrier]) {
    let to_apply = barriers.iter().map(|x| (*x).into()).collect::<Vec<_>>();
    unsafe {
        device.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default()
                .image_memory_barriers(&to_apply)
                .dependency_flags(vk::DependencyFlags::BY_REGION),
        )
    }
}
