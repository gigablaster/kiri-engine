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

use kiri_backend::{Format, ImageCreateDesc, ImageHandle, ImageUsage, RenderContext, RenderDevice};
use log::debug;
use parking_lot::Mutex;

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderTargetSize {
    Backbuffer,
    Fixed([usize; 2]),
    ScaledDown([usize; 2]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporaryRenderTargetDesc {
    pub format: Format,
    pub usage: ImageUsage,
    pub size: RenderTargetSize,
}

impl TemporaryRenderTargetDesc {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            usage: ImageUsage::Sampled,
            size: RenderTargetSize::Backbuffer,
        }
    }

    pub fn color(mut self) -> Self {
        self.usage |= ImageUsage::ColorTarget;
        self
    }

    pub fn depth(mut self) -> Self {
        self.usage |= ImageUsage::DepthStencilTarget;
        self
    }

    pub fn fixed_size(mut self, size: [usize; 2]) -> Self {
        self.size = RenderTargetSize::Fixed(size);
        self
    }

    pub fn scaled_down(mut self, scale: [usize; 2]) -> Self {
        self.size = RenderTargetSize::ScaledDown(scale);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RenderTargetKey {
    format: Format,
    usage: ImageUsage,
    size: [u32; 2],
}

impl TemporaryRenderTargetDesc {
    fn to_key(self, size: [u32; 2]) -> RenderTargetKey {
        let size = match self.size {
            RenderTargetSize::Backbuffer => size,
            RenderTargetSize::Fixed(size) => [size[0] as u32, size[1] as u32],
            RenderTargetSize::ScaledDown(scale) => {
                [size[0] / scale[0] as u32, size[1] / scale[1] as u32]
            }
        };
        RenderTargetKey {
            format: self.format,
            usage: self.usage,
            size,
        }
    }
}
pub struct TemporaryRenderTarget<'a> {
    manager: &'a RenderTargetManager,
    key: RenderTargetKey,
    image: ImageHandle,
}

impl<'a> TemporaryRenderTarget<'a> {
    pub fn image(&self) -> ImageHandle {
        self.image
    }

    pub fn size(&self) -> [u32; 2] {
        self.key.size
    }
}

impl<'a> Drop for TemporaryRenderTarget<'a> {
    fn drop(&mut self) {
        self.manager.release(&self.key, self.image);
    }
}

#[derive(Debug)]
pub struct RenderTargetManager {
    device: Arc<RenderDevice>,
    allocated: Mutex<HashMap<RenderTargetKey, Vec<ImageHandle>>>,
    free: Mutex<HashMap<RenderTargetKey, Vec<ImageHandle>>>,
}

unsafe impl Send for RenderTargetManager {}
unsafe impl Sync for RenderTargetManager {}

impl RenderTargetManager {
    pub fn new(device: &Arc<RenderDevice>) -> Self {
        Self {
            device: device.clone(),
            allocated: Default::default(),
            free: Default::default(),
        }
    }

    pub fn allocate(
        &self,
        context: &RenderContext,
        desc: TemporaryRenderTargetDesc,
    ) -> Result<TemporaryRenderTarget, Error> {
        let mut free = self.free.lock();
        let key = desc.to_key(context.back_buffer_size);
        if let Some(targets) = free.get_mut(&key) {
            if let Some(image) = targets.pop() {
                return Ok(TemporaryRenderTarget {
                    manager: self,
                    key,
                    image,
                });
            }
        }
        drop(free);
        debug!("Create temporary render target {:?}", desc);
        let image = self.device.create_image(
            ImageCreateDesc::new(key.format, key.size)
                .usage(key.usage)
                .name(&format!("{:?}", desc)),
            None,
        )?;
        self.allocated.lock().entry(key).or_default().push(image);
        Ok(TemporaryRenderTarget {
            manager: self,
            key,
            image,
        })
    }

    fn release(&self, key: &RenderTargetKey, image: ImageHandle) {
        self.free.lock().entry(*key).or_default().push(image);
    }

    pub fn cleanup(&self) {
        debug!("Clear all temporary render targets");
        self.free.lock().clear();
        self.allocated.lock().drain().for_each(|(_, mut items)| {
            items
                .drain(..)
                .for_each(|handle| self.device.destroy_image(handle))
        });
    }
}

impl Drop for RenderTargetManager {
    fn drop(&mut self) {
        self.cleanup()
    }
}
