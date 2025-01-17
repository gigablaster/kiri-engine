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

use std::mem;

use arrayvec::ArrayVec;
use kiri_backend::{
    ash::{
        self,
        vk::{self, Rect2D},
    },
    RasterPipeline, MAX_ATTACHMENTS, MAX_COLOR_ATTACHMENTS, MAX_DESCRIPTOR_SETS,
};

use crate::{
    BufferPointer, DescriptorHandle, Error, ImageHandle, PassDispatcher, RasterPipelineHandle,
    RenderResourceResolver,
};

const MAX_DYNAMIC_OFFSETS: usize = 2;

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub image: ImageHandle,
    pub initial_layout: Option<vk::ImageLayout>,
    pub final_layout: Option<vk::ImageLayout>,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub clear: Option<vk::ClearValue>,
}

impl RenderTarget {
    pub fn new(image: ImageHandle) -> Self {
        Self {
            image,
            initial_layout: None,
            final_layout: None,
            clear: None,
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::STORE,
        }
    }

    pub fn initial_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.initial_layout = Some(layout);
        self
    }

    pub fn final_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.final_layout = Some(layout);
        self
    }

    pub fn clear_color(mut self, value: [f32; 4]) -> Self {
        self.load = vk::AttachmentLoadOp::CLEAR;
        self.clear = Some(vk::ClearValue {
            color: vk::ClearColorValue { float32: value },
        });
        self
    }

    pub fn clear_depth_stencil(mut self, depth: f32, stencil: u32) -> Self {
        self.load = vk::AttachmentLoadOp::CLEAR;
        self.clear = Some(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
        });
        self
    }

    pub fn discard(mut self) -> Self {
        self.store = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    pub fn load(mut self) -> Self {
        self.load = vk::AttachmentLoadOp::LOAD;
        self
    }

    fn build(
        &self,
        resolver: &RenderResourceResolver,
        layout: vk::ImageLayout,
    ) -> Result<vk::RenderingAttachmentInfo, Error> {
        let view = resolver.resolve_image_view(self.image)?;
        let info = vk::RenderingAttachmentInfo::default()
            .image_layout(layout)
            .image_view(view)
            .load_op(self.load)
            .store_op(self.store)
            .clear_value(self.clear.unwrap_or_default());
        Ok(info)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageDependency {
    pub aspect: vk::ImageAspectFlags,
    pub image: ImageHandle,
    pub initial_layout: vk::ImageLayout,
    pub desired_layout: Option<vk::ImageLayout>,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
}

impl ImageDependency {
    pub fn color(image: ImageHandle) -> Self {
        Self {
            aspect: vk::ImageAspectFlags::COLOR,
            image,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            desired_layout: None,
            src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_access: vk::AccessFlags2::SHADER_READ,
            src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        }
    }

    pub fn depth(image: ImageHandle) -> Self {
        Self {
            aspect: vk::ImageAspectFlags::DEPTH,
            image,
            initial_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            desired_layout: None,
            src_access: vk::AccessFlags2::SHADER_READ,
            dst_access: vk::AccessFlags2::SHADER_READ,
            src_stage: vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        }
    }

    pub fn desired_layout(mut self, value: vk::ImageLayout) -> Self {
        self.desired_layout = Some(value);
        self
    }

    fn build<'a>(
        self,
        resolver: &RenderResourceResolver<'a>,
        desired_layout: vk::ImageLayout,
    ) -> Result<vk::ImageMemoryBarrier2<'a>, Error> {
        Ok(vk::ImageMemoryBarrier2::default()
            .old_layout(self.initial_layout)
            .new_layout(self.desired_layout.unwrap_or(desired_layout))
            .src_access_mask(self.src_access)
            .dst_access_mask(self.dst_access)
            .src_stage_mask(self.src_stage)
            .dst_stage_mask(self.dst_stage)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
            .image(resolver.resolve_image(self.image)?.raw))
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct DrawCommand {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

#[derive(Debug)]
pub enum DrawCount {
    Fixed(usize),
    Indirect(BufferPointer, usize),
}

#[derive(Debug)]
pub struct Draw {
    pub pipeline: RasterPipelineHandle,
    pub commands: BufferPointer,
    pub count: DrawCount,
    pub stride: usize,
    pub descriptors: [DescriptorHandle; MAX_DESCRIPTOR_SETS],
    pub dynamic_offsets: ArrayVec<u32, MAX_DYNAMIC_OFFSETS>,
}

impl Draw {
    pub fn direct(
        pipeline: RasterPipelineHandle,
        commands: BufferPointer,
        draw_count: usize,
    ) -> Self {
        Self {
            pipeline,
            commands,
            count: DrawCount::Fixed(draw_count),
            stride: mem::size_of::<DrawCommand>(),
            descriptors: [DescriptorHandle::invalid(); 4],
            dynamic_offsets: Default::default(),
        }
    }

    pub fn indirect(
        pipeline: RasterPipelineHandle,
        commands: BufferPointer,
        count: BufferPointer,
        max_draw_count: usize,
    ) -> Self {
        Self {
            pipeline,
            commands,
            count: DrawCount::Indirect(count, max_draw_count),
            stride: mem::size_of::<DrawCommand>(),
            descriptors: [DescriptorHandle::invalid(); 4],
            dynamic_offsets: Default::default(),
        }
    }

    pub fn descriptor(mut self, index: usize, descriptor: DescriptorHandle) -> Self {
        self.descriptors[index] = descriptor;
        self
    }

    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn push_dynamic_offset(mut self, offset: u32) -> Self {
        self.dynamic_offsets.push(offset);
        self
    }
}

pub struct RasterizerPassBuilder<'a> {
    color_targets: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth_target: Option<RenderTarget>,
    dependencies: Vec<ImageDependency>,
    draws: Vec<Draw>,
    area: Option<Rect2D>,
    name: &'a str,
}

impl<'a> RasterizerPassBuilder<'a> {
    pub fn new(
        name: &'a str,
        color_target: &[RenderTarget],
        depth_target: Option<RenderTarget>,
    ) -> Self {
        Self {
            color_targets: color_target
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>(),
            depth_target,
            dependencies: Default::default(),
            draws: Default::default(),
            area: None,
            name,
        }
    }

    pub fn read_image(mut self, image: ImageDependency) -> Self {
        self.dependencies.push(image);
        self
    }

    pub fn draw_area(mut self, area: Rect2D) -> Self {
        self.area = Some(area);
        self
    }

    pub fn draw(&mut self, stream: Draw) {
        self.draws.push(stream);
    }

    pub fn build(self) -> Box<dyn PassDispatcher> {
        Box::new(RasterizerPassDispatcher::new(
            self.name,
            &self.color_targets,
            self.depth_target,
            self.dependencies,
            self.draws,
            self.area,
        ))
    }
}

pub struct RasterizerPassDispatcher {
    color_targets: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth_target: Option<RenderTarget>,
    dependencies: Vec<ImageDependency>,
    draws: Vec<Draw>,
    area: Option<vk::Rect2D>,
    name: String,
}

impl RasterizerPassDispatcher {
    fn new(
        name: &str,
        color_targets: &[RenderTarget],
        depth_target: Option<RenderTarget>,
        dependencies: Vec<ImageDependency>,
        draws: Vec<Draw>,
        area: Option<vk::Rect2D>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            color_targets: color_targets.iter().copied().collect(),
            dependencies,
            depth_target,
            draws,
            area,
        }
    }
}

impl PassDispatcher for RasterizerPassDispatcher {
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
            color_attachments
                .push(color.build(resolver, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)?);
        }
        let mut depth_attachment = None;
        if let Some(depth) = &self.depth_target {
            depth_attachment =
                Some(depth.build(resolver, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)?);
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
                        .image(resolver.resolve_image(target.image)?.raw)
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
                        .image(resolver.resolve_image(target.image)?.raw)
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
        self.dependencies
            .iter()
            .copied()
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
                        .image(resolver.resolve_image(target.image)?.raw)
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
                        .image(resolver.resolve_image(target.image)?.raw)
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
            .map(|x| resolver.resolve_image(x.image).unwrap().desc.dims)
            .chain(
                self.depth_target
                    .iter()
                    .map(|x| resolver.resolve_image(x.image).unwrap().desc.dims),
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
                width: dims[0],
                height: dims[1],
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
        for draw in &self.draws {
            let mut descriptors = [vk::DescriptorSet::null(); MAX_DESCRIPTOR_SETS];
            for (index, descriptor) in draw.descriptors.iter().enumerate() {
                descriptors[index] = resolver
                    .resolve_descriptor_set(*descriptor)
                    .unwrap_or(resolver.empty_descriptor_set);
            }
            let draw_buffer = resolver.resolve_buffer(draw.commands.handle)?;
            let pipeline = resolver.resolve_raster_pipeline(draw.pipeline)?;
            unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.program.pipeline_layout,
                    0,
                    &descriptors,
                    &draw.dynamic_offsets,
                );
                match draw.count {
                    DrawCount::Fixed(count) => device.cmd_draw_indirect(
                        command_buffer,
                        draw_buffer,
                        draw.commands.offset,
                        count as _,
                        draw.stride as _,
                    ),
                    DrawCount::Indirect(count, max_draw_count) => {
                        let count_buffer = resolver.resolve_buffer(count.handle)?;
                        device.cmd_draw_indirect_count(
                            command_buffer,
                            draw_buffer,
                            draw.commands.offset,
                            count_buffer,
                            count.offset,
                            max_draw_count as _,
                            draw.stride as _,
                        );
                    }
                }
            }
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
                        .image(resolver.resolve_image(target.image)?.raw)
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
                        .image(resolver.resolve_image(target.image)?.raw)
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
