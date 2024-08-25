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

use std::{collections::HashMap, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk;
use parking_lot::Mutex;

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

#[derive(Debug, Default)]
pub struct SubpassLayout<'a> {
    pub depth_write: bool,
    pub depth_read: bool,
    pub color_writes: &'a [usize],
    pub color_reads: &'a [usize],
}

#[derive(Debug, Default)]
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
            store: vk::AttachmentStoreOp::DONT_CARE,
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

    pub fn store_output(mut self) -> Self {
        self.store = vk::AttachmentStoreOp::STORE;
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
