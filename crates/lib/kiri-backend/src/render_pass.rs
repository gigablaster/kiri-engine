// Copyright (C) 2023-2024 gigablaster

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

use std::{collections::HashMap, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use crate::{AsVulkan, Image};

use super::{Error, ImageViewDesc, RenderDevice};

#[derive(Debug, Clone, Copy)]
pub enum AttachmentClearValue {
    None,
    Color([f32; 4]),
    DepthStencil(f32, u32),
}

impl From<AttachmentClearValue> for vk::ClearValue {
    fn from(value: AttachmentClearValue) -> Self {
        match value {
            AttachmentClearValue::Color(color) => vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            },
            AttachmentClearValue::DepthStencil(depth, stencil) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
            AttachmentClearValue::None => vk::ClearValue::default(),
        }
    }
}

pub(super) const MAX_COLOR_ATTACHMENTS: usize = 8;
pub(super) const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RenderPassAttachmentDesc {
    pub format: vk::Format,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub samples: vk::SampleCountFlags,
    pub inital_layout: Option<vk::ImageLayout>,
    pub final_layout: Option<vk::ImageLayout>,
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SubpassLayout<'a> {
    pub depth_write: bool,
    pub depth_read: bool,
    pub color_writes: &'a [usize],
    pub color_reads: &'a [usize],
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RenderPassLayout<'a> {
    pub color_targets: &'a [RenderPassAttachmentDesc],
    pub depth_target: Option<RenderPassAttachmentDesc>,
    pub subpasses: &'a [SubpassLayout<'a>],
}

impl RenderPassAttachmentDesc {
    fn build(
        &self,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> vk::AttachmentDescription {
        vk::AttachmentDescription::default()
            .initial_layout(self.inital_layout.unwrap_or(initial_layout).into())
            .final_layout(self.final_layout.unwrap_or(final_layout).into())
            .format(self.format)
            .load_op(self.load.into())
            .store_op(self.store.into())
            .samples(self.samples.into())
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct FramebufferDesc {
    dims: [u32; 2],
    attachments: ArrayVec<vk::ImageView, MAX_ATTACHMENTS>,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderPassAttachment<'a> {
    image: &'a Image,
    aspect: vk::ImageAspectFlags,
    clear: AttachmentClearValue,
}

impl<'a> RenderPassAttachment<'a> {
    pub fn color(image: &'a Image) -> Self {
        Self {
            image,
            aspect: vk::ImageAspectFlags::COLOR,
            clear: AttachmentClearValue::None,
        }
    }

    pub fn depth(image: &'a Image) -> Self {
        Self {
            image,
            aspect: vk::ImageAspectFlags::DEPTH,
            clear: AttachmentClearValue::None,
        }
    }

    pub fn clear(mut self, clear: AttachmentClearValue) -> Self {
        self.clear = clear;
        self
    }
}

impl FramebufferDesc {
    pub fn new(attachments: &[RenderPassAttachment]) -> Result<Self, Error> {
        let mut views = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        let mut dims = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        for attachment in attachments {
            views.push(
                attachment
                    .image
                    .view(ImageViewDesc::new(attachment.aspect))?,
            );
            dims.push(attachment.image.desc().dims);
        }
        assert!(!dims.is_empty(), "Need at least one render target");
        assert!(
            dims.iter().all(|x| *x == dims[0]),
            "All render targets must be of same size"
        );

        Ok(Self {
            dims: dims[0],
            attachments: views,
        })
    }
}

#[derive(Debug)]
pub struct RenderPass {
    device: Arc<RenderDevice>,
    raw: vk::RenderPass,
    framebuffers: RwLock<HashMap<FramebufferDesc, vk::Framebuffer>>,
}

fn add_or_merge_dependency(
    dependency: vk::SubpassDependency,
    cont: &mut Vec<vk::SubpassDependency>,
) {
    for index in 0..cont.len() {
        let dep = &mut cont[index];
        if dep.src_subpass == dependency.src_subpass && dep.dst_subpass == dependency.dst_subpass {
            dep.src_access_mask |= dependency.src_access_mask;
            dep.src_stage_mask |= dependency.src_stage_mask;
            dep.dst_access_mask |= dependency.dst_access_mask;
            dep.dst_stage_mask |= dependency.dst_stage_mask;
            dep.dependency_flags |= dependency.dependency_flags;
        } else {
            cont.push(dependency);
        }
    }
}

impl RenderPass {
    pub fn new(device: &Arc<RenderDevice>, layout: RenderPassLayout) -> Result<Self, Error> {
        // TODO:: fix this shit
        let render_pass_attachments = layout
            .color_targets
            .iter()
            .map(|attachment| {
                attachment.build(
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )
            })
            .chain(layout.depth_target.map(|attachment| {
                attachment.build(
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                )
            }))
            .collect::<Vec<_>>();
        if render_pass_attachments.is_empty() {
            panic!("Render pass must have at least one attachment");
        }
        let color_attachments_count = layout.color_targets.len();
        let depth_attachment_ref =
            layout
                .depth_target
                .is_some()
                .then_some(vk::AttachmentReference {
                    attachment: color_attachments_count as u32,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                });
        let mut subpasses = Vec::with_capacity(layout.subpasses.len());
        let mut last_modified = vec![vk::SUBPASS_EXTERNAL; render_pass_attachments.len()];
        let mut dependencies = Vec::new();

        for (subpass_index, subpass) in layout.subpasses.iter().enumerate() {
            let mut color_attachments_refs = subpass
                .color_writes
                .iter()
                .map(|index| vk::AttachmentReference {
                    attachment: *index as u32,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                })
                .chain(
                    subpass
                        .color_reads
                        .iter()
                        .map(|index| vk::AttachmentReference {
                            attachment: *index as u32,
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        }),
                )
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();
            color_attachments_refs.sort_by(|a, b| a.attachment.cmp(&b.attachment));
            let depth_ref = if subpass.depth_write {
                assert!(depth_attachment_ref.is_some());
                depth_attachment_ref
            } else {
                None
            };
            subpasses.push((color_attachments_refs, depth_ref));
            for index in subpass.color_reads.iter().copied() {
                let dependency = vk::SubpassDependency {
                    src_subpass: last_modified[index],
                    dst_subpass: subpass_index as u32,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                };
                add_or_merge_dependency(dependency, &mut dependencies);
            }
            for index in subpass.color_writes.iter().copied() {
                let dependency = vk::SubpassDependency {
                    src_subpass: last_modified[index],
                    dst_subpass: subpass_index as u32,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::BY_REGION,
                };
                add_or_merge_dependency(dependency, &mut dependencies);
                last_modified[index] = subpass_index as u32;
            }
            if depth_attachment_ref.is_some() {
                let depth = last_modified.last().copied().unwrap();
                if subpass.depth_read {
                    let dependency = vk::SubpassDependency {
                        src_subpass: depth,
                        dst_subpass: subpass_index as u32,
                        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    };
                    add_or_merge_dependency(dependency, &mut dependencies);
                }
                if subpass.depth_write {
                    let dependency = vk::SubpassDependency {
                        src_subpass: depth,
                        dst_subpass: subpass_index as u32,
                        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    };
                    add_or_merge_dependency(dependency, &mut dependencies);
                }
                *last_modified.last_mut().unwrap() = subpass_index as u32;
            }
            for index in 0..last_modified.len() {
                if last_modified[index] != vk::SUBPASS_EXTERNAL
                    && render_pass_attachments[index].store_op == vk::AttachmentStoreOp::STORE
                {
                    let dependency = vk::SubpassDependency {
                        src_subpass: last_modified[index],
                        dst_subpass: vk::SUBPASS_EXTERNAL,
                        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        dst_stage_mask: Default::default(),
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        dependency_flags: vk::DependencyFlags::BY_REGION,
                    };
                    add_or_merge_dependency(dependency, &mut dependencies);
                }
            }
        }

        let subpasses = subpasses
            .iter()
            .map(|x| {
                let mut desc = vk::SubpassDescription::default()
                    .color_attachments(&x.0)
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
                if let Some(depth) = &x.1 {
                    desc = desc.depth_stencil_attachment(depth)
                }
                desc
            })
            .collect::<Vec<_>>();
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&render_pass_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe { device.get().create_render_pass(&render_pass_info, None) }?;
        Ok(Self {
            device: device.clone(),
            raw: render_pass,
            framebuffers: Default::default(),
        })
    }

    pub fn clear_framebuffers(&self) {
        self.framebuffers
            .write()
            .drain()
            .for_each(|(_, fbo)| unsafe { self.device.get().destroy_framebuffer(fbo, None) });
    }

    pub fn framebuffer(
        &self,
        attachments: &[RenderPassAttachment],
    ) -> Result<vk::Framebuffer, Error> {
        let framebuffers = self.framebuffers.upgradable_read();
        let desc = FramebufferDesc::new(attachments)?;
        if let Some(fbo) = framebuffers.get(&desc) {
            Ok(*fbo)
        } else {
            let mut framebuffers = RwLockUpgradableReadGuard::upgrade(framebuffers);
            if let Some(fbo) = framebuffers.get(&desc) {
                Ok(*fbo)
            } else {
                let fbo_info = vk::FramebufferCreateInfo::default()
                    .render_pass(self.raw)
                    .attachments(&desc.attachments)
                    .width(desc.dims[0])
                    .height(desc.dims[1])
                    .layers(1);
                let fbo = unsafe { self.device.get().create_framebuffer(&fbo_info, None) }?;
                framebuffers.insert(desc, fbo);
                Ok(fbo)
            }
        }
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        self.clear_framebuffers();
        unsafe { self.device.get().destroy_render_pass(self.raw, None) };
    }
}

impl AsVulkan<vk::RenderPass> for RenderPass {
    fn as_vulkan(&self) -> vk::RenderPass {
        self.raw
    }
}
