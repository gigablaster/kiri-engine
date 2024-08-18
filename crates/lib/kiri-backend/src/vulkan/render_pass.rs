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

use std::collections::HashMap;

use arrayvec::ArrayVec;
use ash::vk::{self};
use log::debug;
use parking_lot::Mutex;

use crate::{
    Format, ImageAspect, ImageLayout, ImageMultisampling, RenderTargetLoadOp, RenderTargetStoreOp,
};

use super::{Error, ImageHandle, ImagePool, ImageViewDesc, RenderDevice, RenderPassHandle};

#[derive(Debug, Clone, Copy)]
pub enum ClearRenderTarget {
    None,
    Color([f32; 4]),
    DepthStencil(f32, u32),
}

impl From<ClearRenderTarget> for vk::ClearValue {
    fn from(value: ClearRenderTarget) -> Self {
        match value {
            ClearRenderTarget::Color(color) => vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            },
            ClearRenderTarget::DepthStencil(depth, stencil) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
            ClearRenderTarget::None => vk::ClearValue::default(),
        }
    }
}

pub(crate) const MAX_COLOR_ATTACHMENTS: usize = 8;
pub(crate) const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RenderTargetDesc {
    pub format: Format,
    pub load_op: RenderTargetLoadOp,
    pub store_op: RenderTargetStoreOp,
    pub samples: ImageMultisampling,
    pub inital_layout: Option<ImageLayout>,
    pub final_layout: Option<ImageLayout>,
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct SubpassLayout {
    depth_write: bool,
    depth_read: bool,
    color_writes: Vec<usize>,
    color_reads: Vec<usize>,
}

impl SubpassLayout {
    pub fn depth_write(mut self) -> Self {
        self.depth_write = true;
        self
    }

    pub fn depth_read(mut self) -> Self {
        self.depth_read = true;
        self
    }

    pub fn color_write(mut self, indices: &[usize]) -> Self {
        self.color_writes.extend(indices);
        self
    }

    pub fn color_read(mut self, indices: &[usize]) -> Self {
        self.color_reads.extend(indices);
        self
    }
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct RenderPassLayout {
    pub color_targets: Vec<RenderTargetDesc>,
    pub depth_target: Option<RenderTargetDesc>,
    pub subpasses: Vec<SubpassLayout>,
}

impl RenderPassLayout {
    pub fn color_target(mut self, target: RenderTargetDesc) -> Self {
        self.color_targets.push(target);
        self
    }

    pub fn depth_target(mut self, target: RenderTargetDesc) -> Self {
        self.depth_target = Some(target);
        self
    }

    pub fn subpass(mut self, subpass: SubpassLayout) -> Self {
        self.subpasses.push(subpass);
        self
    }
}

impl RenderTargetDesc {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            load_op: RenderTargetLoadOp::Discard,
            store_op: RenderTargetStoreOp::Discard,
            samples: ImageMultisampling::None,
            inital_layout: None,
            final_layout: None,
        }
    }

    pub fn clear_input(mut self) -> Self {
        self.load_op = RenderTargetLoadOp::Clear;
        self
    }

    pub fn load_input(mut self) -> Self {
        self.load_op = RenderTargetLoadOp::Load;
        self
    }

    pub fn store_output(mut self) -> Self {
        self.store_op = RenderTargetStoreOp::Store;
        self
    }

    pub fn final_layout(mut self, layout: ImageLayout) -> Self {
        self.final_layout = Some(layout);
        self
    }

    pub fn initial_layout(mut self, layout: ImageLayout) -> Self {
        self.inital_layout = Some(layout);
        self
    }

    fn build(
        &self,
        initial_layout: ImageLayout,
        final_layout: ImageLayout,
    ) -> vk::AttachmentDescription {
        vk::AttachmentDescription::default()
            .initial_layout(self.inital_layout.unwrap_or(initial_layout).into())
            .final_layout(self.final_layout.unwrap_or(final_layout).into())
            .format(self.format.into())
            .load_op(self.load_op.into())
            .store_op(self.store_op.into())
            .samples(self.samples.into())
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct FramebufferDesc {
    pub dims: [u32; 2],
    pub attachments: ArrayVec<vk::ImageView, MAX_ATTACHMENTS>,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderTarget {
    pub image: ImageHandle,
    pub aspect: ImageAspect,
    pub clear: ClearRenderTarget,
}

impl RenderTarget {
    pub fn color(image: ImageHandle) -> Self {
        Self {
            image,
            aspect: ImageAspect::Color,
            clear: ClearRenderTarget::None,
        }
    }

    pub fn depth(image: ImageHandle) -> Self {
        Self {
            image,
            aspect: ImageAspect::Depth,
            clear: ClearRenderTarget::None,
        }
    }

    pub fn clear(mut self, clear: ClearRenderTarget) -> Self {
        self.clear = clear;
        self
    }
}

impl FramebufferDesc {
    pub fn new(
        device: &ash::Device,
        images: &ImagePool,
        attachments: &[RenderTarget],
    ) -> Result<Self, Error> {
        let mut views = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        let mut dims = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        for attachment in attachments {
            let image = images
                .get_cold(attachment.image)
                .ok_or(Error::InvalidImageHandle(attachment.image))?;
            views.push(image.view(device, ImageViewDesc::new(attachment.aspect))?);
            dims.push(image.desc.dims);
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
    pub(crate) raw: vk::RenderPass,
    framebuffers: Mutex<HashMap<FramebufferDesc, vk::Framebuffer>>,
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

impl RenderDevice {
    pub fn create_render_pass(&self, layout: &RenderPassLayout) -> Result<RenderPassHandle, Error> {
        let render_pass_attachments = layout
            .color_targets
            .iter()
            .map(|attachment| attachment.build(ImageLayout::ColorTarget, ImageLayout::ColorTarget))
            .chain(layout.depth_target.map(|attachment| {
                attachment.build(
                    ImageLayout::DepthStencilTarget,
                    ImageLayout::DepthStencilTarget,
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
            let color_attachments_refs = subpass
                .color_writes
                .iter()
                .map(|index| vk::AttachmentReference {
                    attachment: *index as u32,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                })
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();
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

        let render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None) }?;

        let mut render_passes = self.render_passes.write();
        Ok(render_passes.push(RenderPass {
            raw: render_pass,
            framebuffers: Mutex::default(),
        }))
    }

    pub fn destroy_render_pass(&self, handle: RenderPassHandle) {
        if let Some(pass) = self.render_passes.write().remove(handle) {
            pass.free(&self.device);
        }
    }

    pub fn clear_framebuffers(&self, handle: RenderPassHandle) {
        if let Some(pass) = self.render_passes.read().get(handle) {
            pass.clear_framebuffers(&self.device);
        }
    }

    pub(crate) fn clear_swapchain_dependent_resources(&self) {
        debug!("Clear all framebuffers");
        self.render_passes
            .write()
            .iter()
            .for_each(|pass| pass.clear_framebuffers(&self.device));
    }
}

impl RenderPass {
    pub fn framebuffer(
        &self,
        device: &ash::Device,
        images: &ImagePool,
        attachments: &[RenderTarget],
    ) -> Result<(vk::Framebuffer, [u32; 2]), Error> {
        let mut cache = self.framebuffers.lock();
        let key = FramebufferDesc::new(device, images, attachments)?;
        if let Some(fbo) = cache.get(&key) {
            Ok((*fbo, key.dims))
        } else {
            let fbo_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.raw)
                .attachments(&key.attachments)
                .width(key.dims[0])
                .height(key.dims[1])
                .layers(1);
            let dims = key.dims;
            let framebuffer = unsafe { device.create_framebuffer(&fbo_info, None) }?;
            cache.insert(key, framebuffer);

            Ok((framebuffer, dims))
        }
    }

    pub fn clear_framebuffers(&self, device: &ash::Device) {
        let mut cache = self.framebuffers.lock();
        for (_, fbo) in cache.iter() {
            unsafe { device.destroy_framebuffer(*fbo, None) }
        }
        cache.clear();
    }

    pub fn free(&self, device: &ash::Device) {
        self.clear_framebuffers(device);
        unsafe { device.destroy_render_pass(self.raw, None) };
    }
}
