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

use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use parking_lot::Mutex;

use crate::{Error, Image, ImageViewDesc, RenderDevice};

pub const MAX_COLOR_ATTACHMENTS: usize = 8;
pub const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;
pub const MAX_SUBPASSES: usize = 16;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageAttachmentDesc {
    pub format: vk::Format,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub samples: vk::SampleCountFlags,
    pub initial_layout: Option<vk::ImageLayout>,
    pub final_layout: Option<vk::ImageLayout>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SubpassLayout<'a> {
    pub depth_write: bool,
    pub depth_read: bool,
    pub color_writes: &'a [usize],
    pub color_reads: &'a [usize],
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RenderPassLayout<'a> {
    pub color: &'a [ImageAttachmentDesc],
    pub depth: Option<ImageAttachmentDesc>,
    pub subpasses: &'a [SubpassLayout<'a>],
}

impl ImageAttachmentDesc {
    pub fn new(format: vk::Format) -> Self {
        Self {
            format,
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::STORE,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: None,
            final_layout: None,
        }
    }

    pub fn clear_input(mut self) -> Self {
        self.load = vk::AttachmentLoadOp::CLEAR;
        self
    }

    pub fn load_input(mut self) -> Self {
        self.load = vk::AttachmentLoadOp::LOAD;
        self
    }

    pub fn discard(mut self) -> Self {
        self.store = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    pub fn final_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.final_layout = Some(layout);
        self
    }

    pub fn initial_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.initial_layout = Some(layout);
        self
    }

    fn build(
        &self,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> vk::AttachmentDescription {
        vk::AttachmentDescription::default()
            .initial_layout(self.initial_layout.unwrap_or(initial_layout))
            .final_layout(self.final_layout.unwrap_or(final_layout))
            .format(self.format)
            .load_op(self.load)
            .store_op(self.store)
            .samples(self.samples)
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
struct FboDesc {
    pub dims: [u32; 2],
    pub attachments: ArrayVec<vk::ImageView, MAX_ATTACHMENTS>,
}

#[derive(Debug)]
pub struct ImageAttachment<'a> {
    pub image: &'a Image,
    pub aspect: vk::ImageAspectFlags,
}

impl FboDesc {
    pub fn new(attachments: &[ImageAttachment]) -> Result<Self, Error> {
        let dims = attachments
            .iter()
            .map(|x| x.image.desc.dims)
            .next()
            .expect("Need at least one attacment");
        let mut views = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        for attachment in attachments {
            views.push(
                attachment
                    .image
                    .view(ImageViewDesc::new(attachment.aspect))?,
            );
        }
        Ok(Self {
            dims,
            attachments: views,
        })
    }
}

#[derive(Debug)]
pub struct RenderPass {
    device: Arc<RenderDevice>,
    pub raw: vk::RenderPass,
    fbos: Mutex<HashMap<FboDesc, vk::Framebuffer>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Fbo {
    pub raw: vk::Framebuffer,
    pub dims: [u32; 2],
}

impl Fbo {
    pub fn area(self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: self.dims[0],
                height: self.dims[1],
            },
        }
    }
}

impl RenderPass {
    pub fn new(device: &Arc<RenderDevice>, layout: RenderPassLayout) -> Result<Self, Error> {
        let attachments = layout
            .color
            .iter()
            .map(|x| {
                x.build(
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )
            })
            .chain(layout.depth.iter().map(|x| {
                x.build(
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                )
            }))
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
        assert!(!attachments.is_empty(), "Need at least one attachment");
        let depth_index = attachments.len() - 1;
        let mut color_attachments_per_subpass = ArrayVec::<_, MAX_SUBPASSES>::new();
        let mut depth_attachments_per_subpass = ArrayVec::<_, MAX_SUBPASSES>::new();
        for subpass in layout.subpasses.iter() {
            let mut color_references = subpass
                .color_reads
                .iter()
                .chain(subpass.color_writes)
                .copied()
                .collect::<Vec<_>>();
            color_references.sort();
            color_references.dedup();
            let color_references = color_references
                .into_iter()
                .map(|x| {
                    vk::AttachmentReference::default()
                        .attachment(x as _)
                        .layout(attachments[x].initial_layout)
                })
                .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
            if subpass.depth_read || subpass.depth_write {
                assert!(
                    layout.depth.is_some(),
                    "Subpass wants to access depth attachment, but there's no depth attachment."
                );
                depth_attachments_per_subpass.push(Some(
                    vk::AttachmentReference::default()
                        .attachment(depth_index as _)
                        .layout(attachments[depth_index].initial_layout),
                ));
            } else {
                depth_attachments_per_subpass.push(None);
            }
            color_attachments_per_subpass.push(color_references);
        }
        let subpasses = layout
            .subpasses
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let mut subpass = vk::SubpassDescription::default()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&color_attachments_per_subpass[index]);
                if let Some(depth) = &depth_attachments_per_subpass[index] {
                    subpass = subpass.depth_stencil_attachment(depth);
                }
                subpass
            })
            .collect::<ArrayVec<_, MAX_SUBPASSES>>();
        let dependencies = build_dependencies(layout);
        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        let raw = unsafe { device.raw.create_render_pass(&create_info, None) }?;
        Ok(Self {
            device: device.clone(),
            raw,
            fbos: Default::default(),
        })
    }

    pub fn fbo(&self, attachments: &[ImageAttachment]) -> Result<Fbo, Error> {
        let mut cache = self.fbos.lock();
        let key = FboDesc::new(attachments)?;
        if let Some(fbo) = cache.get(&key) {
            Ok(Fbo {
                raw: *fbo,
                dims: key.dims,
            })
        } else {
            let fbo_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.raw)
                .attachments(&key.attachments)
                .width(key.dims[0])
                .height(key.dims[1])
                .layers(1);
            let fbo = unsafe { self.device.raw.create_framebuffer(&fbo_info, None) }?;
            let dims = key.dims;
            cache.insert(key, fbo);

            Ok(Fbo { raw: fbo, dims })
        }
    }

    pub fn clear_fbos(&self) {
        let mut cache = self.fbos.lock();
        for (_, fbo) in cache.iter() {
            unsafe { self.device.raw.destroy_framebuffer(*fbo, None) }
        }
        cache.clear();
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        self.clear_fbos();
        unsafe { self.device.raw.destroy_render_pass(self.raw, None) }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResourceState {
    pub last_read: u32,
    pub last_write: u32,
    pub load: bool,
    pub store: bool,
}

impl ResourceState {
    pub fn new(target: &ImageAttachmentDesc) -> Self {
        Self {
            last_read: vk::SUBPASS_EXTERNAL,
            last_write: vk::SUBPASS_EXTERNAL,
            load: target.load == vk::AttachmentLoadOp::LOAD,
            store: target.store == vk::AttachmentStoreOp::STORE,
        }
    }
}

fn subpass_sorter(lhs: &vk::SubpassDependency, rhs: &vk::SubpassDependency) -> Ordering {
    // For same src subpass order by dst subpass
    if lhs.src_subpass == rhs.src_subpass {
        if lhs.dst_subpass < rhs.dst_subpass {
            return Ordering::Less;
        }
        if lhs.dst_subpass > rhs.dst_subpass {
            return Ordering::Greater;
        }
        return Ordering::Equal;
    }
    // Element with SUBPASS_EXTERNAL is less
    if lhs.src_subpass == vk::SUBPASS_EXTERNAL {
        return Ordering::Less;
    }
    if rhs.src_subpass == vk::SUBPASS_EXTERNAL {
        return Ordering::Greater;
    }
    if lhs.src_subpass < rhs.src_subpass {
        return Ordering::Less;
    }
    if lhs.src_subpass > rhs.src_subpass {
        return Ordering::Greater;
    }
    Ordering::Equal
}

fn merge_subpasses(subpasses: Vec<vk::SubpassDependency>) -> Vec<vk::SubpassDependency> {
    let mut subpasses = subpasses;
    subpasses.retain(|x| x.src_subpass != x.dst_subpass);
    // Sort
    subpasses.sort_by(subpass_sorter);
    // Combine stages for same pairs
    let mut result = Vec::new();
    while !subpasses.is_empty() {
        let mut subpass = subpasses.remove(0);
        let mut index = 0;
        while index < subpasses.len() {
            let next = &subpasses[0];
            if next.src_subpass == subpass.src_subpass && next.dst_subpass == subpass.dst_subpass {
                let next = subpasses.remove(0);
                subpass.src_access_mask |= next.src_access_mask;
                subpass.dst_access_mask |= next.dst_access_mask;
                subpass.src_stage_mask |= next.src_stage_mask;
                subpass.dst_stage_mask |= next.dst_stage_mask;
            } else {
                index += 1;
            }
        }
        result.push(subpass);
    }

    result
}

fn build_dependencies(layout: RenderPassLayout) -> Vec<vk::SubpassDependency> {
    // For every pass track state of every resource and add dependencies if needed
    let mut state = layout
        .color
        .iter()
        .map(ResourceState::new)
        .chain(layout.depth.iter().map(ResourceState::new))
        .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
    let depth_index = state.len() - 1;
    let mut dependencies = Vec::new();
    // Add external dependencies at first subpass
    for index in 0..state.len() {
        if state[index].load {
            dependencies.push(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            });
            state[index].last_read = 0;
        }
        if state[index].store {
            dependencies.push(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            });
            state[index].last_write = 0;
        }
    }
    if layout.depth.is_some() {
        if state[depth_index].load {
            dependencies.push(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            });
            state[depth_index].last_read = 0;
        }
        if state[depth_index].store {
            dependencies.push(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            });
            state[depth_index].last_write = 0;
        }
    }
    for index in 0..layout.subpasses.len() {
        let subpass = layout.subpasses[index];
        // For all resources we read from - check last stage we wrote into them and add write-read dependency
        let index = index as u32;
        for read in subpass.color_reads.iter().copied() {
            let last_state = state[read];
            if last_state.last_write != index {
                dependencies.push(vk::SubpassDependency {
                    src_subpass: last_state.last_write,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                });
                state[read].last_read = index;
            }
        }
        if subpass.depth_read && layout.depth.is_some() {
            let last_state = state[depth_index];
            if last_state.last_write != index {
                dependencies.push(vk::SubpassDependency {
                    src_subpass: last_state.last_write,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                });
                state[depth_index].last_read = index;
            }
        }

        // For all resources we write into - check last stage we wrote into them and add write-write dependency
        for write in subpass.color_writes.iter().copied() {
            let last_state = state[write];
            if last_state.last_write != index {
                dependencies.push(vk::SubpassDependency {
                    src_subpass: last_state.last_write,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                });
                state[write].last_write = index;
            }
        }
        if subpass.depth_write && layout.depth.is_some() {
            let last_state = state[depth_index];
            if last_state.last_write != index {
                dependencies.push(vk::SubpassDependency {
                    src_subpass: last_state.last_write,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                });
                state[depth_index].last_write = index;
            }
        }
        // For all resources we write into - check last stage we read from them and add read-write dependenct
        for write in subpass.color_writes.iter().copied() {
            let last_state = state[write];
            if last_state.last_read != index {
                dependencies.push(vk::SubpassDependency {
                    src_subpass: last_state.last_read,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                });
                state[write].last_write = index;
            }
        }
        if subpass.depth_write && layout.depth.is_some() {
            let last_state = state[depth_index];
            if last_state.last_read != index {
                dependencies.push(vk::SubpassDependency {
                    src_subpass: last_state.last_write,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                });
                state[depth_index].last_write = index;
            }
        }
        // For all resources we read from - check last stage we read from them and just remember that
        for read in subpass.color_reads.iter().copied() {
            let last_state = state[read];
            if last_state.last_read != index {
                state[read].last_read = index;
            }
        }
        if subpass.depth_read && layout.depth.is_some() {
            let last_state = state[depth_index];
            if last_state.last_read != index {
                state[depth_index].last_read = index;
            }
        }
    }
    // For all targets that aren't discarded we add write->read dependecny to external pass
    for (index, target) in layout.color.iter().enumerate() {
        if target.store == vk::AttachmentStoreOp::STORE {
            let last_write = state[index].last_write;
            dependencies.push(vk::SubpassDependency {
                src_subpass: last_write,
                dst_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            })
        }
    }
    if let Some(depth) = layout.depth {
        if depth.store == vk::AttachmentStoreOp::STORE {
            let last_write = state[depth_index].last_write;
            dependencies.push(vk::SubpassDependency {
                src_subpass: last_write,
                dst_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            })
        }
    }
    merge_subpasses(dependencies)
}
