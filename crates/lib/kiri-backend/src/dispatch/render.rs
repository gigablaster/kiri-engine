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

use arrayvec::ArrayVec;
use ash::vk;

use crate::{vulkan::MAX_ATTACHMENTS, DrawStream, Error};

use super::{ImageDependency, PassDispatcher, RenderArea, RenderResourceResolver, RenderTarget};

pub struct RenderPassDispatcher {
    name: String,
    area: Option<vk::Rect2D>,
    reads: Vec<ImageDependency>,
    color_targets: Vec<RenderTarget>,
    depth_target: Option<RenderTarget>,
    streams: Vec<DrawStream>,
}

impl RenderPassDispatcher {
    pub fn new(
        name: impl AsRef<str>,
        area: RenderArea,
        color_targets: impl IntoIterator<Item = RenderTarget>,
        depth_target: Option<RenderTarget>,
        reads: impl IntoIterator<Item = ImageDependency>,
        streams: impl IntoIterator<Item = DrawStream>,
    ) -> Self {
        Self {
            name: name.as_ref().to_owned(),
            area: area.into(),
            reads: reads.into_iter().collect(),
            color_targets: color_targets.into_iter().collect(),
            depth_target,
            streams: streams.into_iter().collect(),
        }
    }
}

impl PassDispatcher for RenderPassDispatcher {
    fn name(&self) -> &str {
        &self.name
    }

    fn dispatch(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        resolver: &RenderResourceResolver,
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let mut color_attachments = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        for color in &self.color_targets {
            color_attachments.push(color.build(
                resolver,
                vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            )?);
        }
        let mut depth_attachment = None;
        if let Some(depth) = &self.depth_target {
            depth_attachment = Some(depth.build(
                resolver,
                vk::ImageAspectFlags::DEPTH,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            )?);
        }

        // From initial layout to attachments
        let mut barriers = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        for target in &self.color_targets {
            let initial_layout = target
                .initial_layout
                .unwrap_or(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            if initial_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(resolver.resolve_image(target.image)?)
                        .src_access_mask(vk::AccessFlags2::SHADER_READ)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .old_layout(initial_layout)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            } else {
                // Write-write barrier
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(resolver.resolve_image(target.image)?)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            }
        }
        self.reads
            .iter()
            .cloned()
            .try_for_each(|x| -> Result<(), Error> {
                barriers.push(x.build(resolver, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?);
                Ok(())
            })?;
        if let Some(target) = &self.depth_target {
            let initial_layout = target
                .initial_layout
                .unwrap_or(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            if initial_layout != vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(resolver.resolve_image(target.image)?)
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .old_layout(initial_layout)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .dst_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH
                                | vk::ImageAspectFlags::STENCIL,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            } else {
                // Write-write barrier
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(resolver.resolve_image(target.image)?)
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH
                                | vk::ImageAspectFlags::STENCIL,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            }
        }
        unsafe {
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&barriers)
                    .dependency_flags(vk::DependencyFlags::BY_REGION),
            )
        };

        let dims = self
            .color_targets
            .iter()
            .map(|x| resolver.resolve_image_desc(x.image).unwrap().dims)
            .chain(
                self.depth_target
                    .iter()
                    .map(|x| resolver.resolve_image_desc(x.image).unwrap().dims),
            )
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
        assert!(!dims.is_empty(), "Need at least one render target");
        assert!(
            dims.iter().all(|x| *x == dims[0]),
            "All attachments must be of same size"
        );
        let dims = dims[0];
        let render_area = self.area.unwrap_or(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: dims[0] as u32,
                height: dims[1] as u32,
            },
        });
        let mut rendering_info = vk::RenderingInfo::default()
            .color_attachments(&color_attachments)
            .layer_count(1)
            .render_area(render_area);
        if let Some(depth) = &depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth);
        }
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };
        for stream in &self.streams {
            stream.execute(device, command_buffer, render_area, resolver)?;
        }
        unsafe { device.cmd_end_rendering(command_buffer) };

        // From attachments to final layouts
        let mut barriers = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        for target in &self.color_targets {
            let final_layout = target
                .final_layout
                .unwrap_or(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            if final_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(resolver.resolve_image(target.image)?)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(final_layout)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            }
        }
        if let Some(target) = &self.depth_target {
            let final_layout = target
                .final_layout
                .unwrap_or(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            if final_layout != vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(resolver.resolve_image(target.image)?)
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
                        .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .new_layout(final_layout)
                        .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH
                                | vk::ImageAspectFlags::STENCIL,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            }
        }
        unsafe {
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&barriers)
                    .dependency_flags(vk::DependencyFlags::BY_REGION),
            )
        };
        Ok(())
    }
}
