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
use ash::vk;
use parking_lot::Mutex;
use std::cmp::Ord;

use crate::{Error, Image, ImageViewDesc, RenderDevice};

pub const MAX_COLOR_ATTACHMENTS: usize = 8;
pub const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Copy)]
pub enum RenderTargetClearValue {
    Color([f32; 4]),
    DepthStencil(f32, u32),
}

impl From<RenderTargetClearValue> for vk::ClearValue {
    fn from(value: RenderTargetClearValue) -> Self {
        match value {
            RenderTargetClearValue::Color(color) => vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            },
            RenderTargetClearValue::DepthStencil(depth, stencil) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RenderTargetDesc {
    pub format: vk::Format,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub samples: vk::SampleCountFlags,
    pub inital_layout: Option<vk::ImageLayout>,
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
    pub color: &'a [RenderTargetDesc],
    pub depth: Option<RenderTargetDesc>,
    pub subpasses: &'a [SubpassLayout<'a>],
}

impl RenderTargetDesc {
    pub fn new(format: vk::Format) -> Self {
        Self {
            format,
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::STORE,
            samples: vk::SampleCountFlags::TYPE_1,
            inital_layout: None,
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
        self.inital_layout = Some(layout);
        self
    }

    fn build(
        &self,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> vk::AttachmentDescription {
        vk::AttachmentDescription::default()
            .initial_layout(self.inital_layout.unwrap_or(initial_layout))
            .final_layout(self.final_layout.unwrap_or(final_layout))
            .format(self.format)
            .load_op(self.load)
            .store_op(self.store)
            .samples(self.samples)
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
struct FramebufferDesc {
    pub dims: [u32; 2],
    pub attachments: ArrayVec<vk::ImageView, MAX_ATTACHMENTS>,
}

#[derive(Debug)]
pub struct RenderTarget<'a> {
    pub image: &'a Image,
    pub aspect: vk::ImageAspectFlags,
}

impl FramebufferDesc {
    pub fn new(attachments: &[RenderTarget]) -> Result<Self, Error> {
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
    framebuffers: Mutex<HashMap<FramebufferDesc, vk::Framebuffer>>,
}

impl RenderPass {
    pub fn framebuffer(&self, attachments: &[RenderTarget]) -> Result<vk::Framebuffer, Error> {
        let mut cache = self.framebuffers.lock();
        let key = FramebufferDesc::new(attachments)?;
        if let Some(fbo) = cache.get(&key) {
            Ok(*fbo)
        } else {
            let fbo_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.raw)
                .attachments(&key.attachments)
                .width(key.dims[0])
                .height(key.dims[1])
                .layers(1);
            let framebuffer = unsafe { self.device.raw.create_framebuffer(&fbo_info, None) }?;
            cache.insert(key, framebuffer);

            Ok(framebuffer)
        }
    }

    pub fn clear_framebuffers(&self) {
        let mut cache = self.framebuffers.lock();
        for (_, fbo) in cache.iter() {
            unsafe { self.device.raw.destroy_framebuffer(*fbo, None) }
        }
        cache.clear();
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        self.clear_framebuffers();
        unsafe { self.device.raw.destroy_render_pass(self.raw, None) }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResourceState {
    pub last_read: u32,
    pub last_written: u32,
}

impl Default for ResourceState {
    fn default() -> Self {
        Self {
            last_read: vk::SUBPASS_EXTERNAL,
            last_written: vk::SUBPASS_EXTERNAL,
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
    // First, we elimenate strange external-external dependencies
    subpasses
        .retain(|x| x.src_subpass != vk::SUBPASS_EXTERNAL && x.dst_subpass != vk::SUBPASS_EXTERNAL);
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

fn build_subpasses(layout: RenderPassLayout) -> Vec<vk::SubpassDependency> {
    // For every pass track state of every resource and add dependencies if needed
    let mut state = layout
        .color
        .iter()
        .map(|x| ResourceState::default())
        .chain(layout.depth.iter().map(|x| ResourceState::default()))
        .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
    let depth_index = layout.depth.map(|_| state.len() - 1);
    let mut subpasses = Vec::new();
    for index in 0..layout.subpasses.len() {
        let subpass = layout.subpasses[index];
        // For all resources we read from - check last stage we wrote into them and add write-read dependency
        let index = index as u32;
        for read in subpass.color_reads.iter().copied() {
            let last_state = state[read];
            if last_state.last_written != index {
                subpasses.push(vk::SubpassDependency {
                    src_subpass: last_state.last_written,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                })
            }
            state[read].last_read = index;
        }
        if subpass.depth_read {
            if let Some(depth_index) = depth_index {
                let last_state = state[depth_index];
                if last_state.last_written != index {
                    subpasses.push(vk::SubpassDependency {
                        src_subpass: last_state.last_written,
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
        }

        // For all resources we write into - check last stage we wrote into them and add write-write dependency
        for write in subpass.color_writes.iter().copied() {
            let last_state = state[write];
            if last_state.last_written != index {
                subpasses.push(vk::SubpassDependency {
                    src_subpass: last_state.last_written,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                })
            }
            state[write].last_written = index;
        }
        if subpass.depth_write {
            if let Some(depth_index) = depth_index {
                let last_state = state[depth_index];
                if last_state.last_written != index {
                    subpasses.push(vk::SubpassDependency {
                        src_subpass: last_state.last_written,
                        dst_subpass: index,
                        src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    });
                    state[depth_index].last_written = index;
                }
            }
        }
        // For all resources we write into - check last stage we read from them and add read-write dependenct
        for write in subpass.color_writes.iter().copied() {
            let last_state = state[write];
            if last_state.last_read != index {
                subpasses.push(vk::SubpassDependency {
                    src_subpass: last_state.last_read,
                    dst_subpass: index,
                    src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                });
                state[write].last_written = index;
            }
        }
        if subpass.depth_write {
            if let Some(depth_index) = depth_index {
                let last_state = state[depth_index];
                if last_state.last_read != index {
                    subpasses.push(vk::SubpassDependency {
                        src_subpass: last_state.last_written,
                        dst_subpass: index,
                        src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    });
                    state[depth_index].last_written = index;
                }
            }
        }
        // For all resources we read from - check last stage we read from them and just remember that
        for read in subpass.color_reads.iter().copied() {
            let last_state = state[read];
            if last_state.last_read != index {
                state[read].last_read = index;
            }
        }
        if subpass.depth_read {
            if let Some(depth_index) = depth_index {
                let last_state = state[depth_index];
                if last_state.last_read != index {
                    state[depth_index].last_read = index;
                }
            }
        }
    }
    // For all targets that aren't discarded we add write->read dependecny to external pass
    for (index, target) in layout.color.iter().enumerate() {
        if target.store == vk::AttachmentStoreOp::STORE {
            let last_write = state[index].last_written;
            subpasses.push(vk::SubpassDependency {
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
            let last_write = state[state.len() - 1].last_written;
            subpasses.push(vk::SubpassDependency {
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
    merge_subpasses(subpasses)
}
