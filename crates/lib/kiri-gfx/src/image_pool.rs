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

use crate::{ImageHandle, Renderer};
use kiri_backend::{ash::vk, ImageCreateDesc};
use log::debug;
use parking_lot::Mutex;

use crate::Error;

pub trait ResolutionScale {
    fn scale_down(&self, scale: u32) -> [u32; 2];
}

impl ResolutionScale for [u32; 2] {
    fn scale_down(&self, scale: u32) -> [u32; 2] {
        [self[0] / scale, self[1] / scale]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TempImageKey {
    pub dims: [u32; 2],
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Debug)]
pub struct RenderTargetPool {
    renderer: Arc<Renderer>,
    images: Mutex<HashMap<TempImageKey, Vec<ImageHandle>>>,
}

#[derive(Debug)]
pub struct TransientImageGuard<'a> {
    pool: &'a RenderTargetPool,
    key: TempImageKey,
    pub handle: ImageHandle,
}

impl RenderTargetPool {
    pub fn new(renderer: &Arc<Renderer>) -> Self {
        Self {
            renderer: renderer.clone(),
            images: Default::default(),
        }
    }

    /// Get attachemnt that will return into pool automatically when
    /// when frame is rendererd
    pub fn get_image(
        &self,
        format: vk::Format,
        dims: [u32; 2],
        usage: vk::ImageUsageFlags,
    ) -> Result<TransientImageGuard, Error> {
        assert!(
            usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                || usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            "Must be an attachment"
        );
        let (key, image) = self.get_or_allocate_image(format, dims, usage)?;
        Ok(TransientImageGuard {
            pool: self,
            key,
            handle: image,
        })
    }

    fn get_or_allocate_image(
        &self,
        format: vk::Format,
        dims: [u32; 2],
        usage: vk::ImageUsageFlags,
    ) -> Result<(TempImageKey, ImageHandle), Error> {
        let mut images = self.images.lock();
        let key = TempImageKey {
            dims,
            format,
            usage,
        };
        let group = images.entry(key).or_default();
        if let Some(image) = group.pop() {
            Ok((key, image))
        } else {
            debug!(
                "Create render taget resolution: {:?} format: {:?} usage: {:?}",
                dims, format, usage,
            );
            let image = self.renderer.create_image(
                ImageCreateDesc::new(format, dims)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(usage),
                None,
            )?;
            Ok((key, image))
        }
    }

    pub fn purge(&self) {
        let mut images = self.images.lock();
        images.drain().for_each(|(_, mut group)| {
            group.drain(..).for_each(|x| self.renderer.destroy_image(x))
        });
    }

    fn recycle(&self, image: ImageHandle, key: TempImageKey) {
        self.images.lock().entry(key).or_default().push(image);
    }
}

impl<'a> Drop for TransientImageGuard<'a> {
    fn drop(&mut self) {
        self.pool.recycle(self.handle, self.key);
    }
}
