// Copyright (C) 2024-2025 gigablaster

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

use log::debug;
use parking_lot::Mutex;

use kiri_backend::{
    ash::vk,
    vulkan::{GraphicsDevice, ImageCreateDesc, ImageHandle},
    Error,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TempImageKey {
    pub dims: [usize; 2],
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Debug)]
pub struct RenderTargetPool {
    device: Arc<GraphicsDevice>,
    images: Mutex<HashMap<TempImageKey, Vec<ImageHandle>>>,
}

#[derive(Debug)]
pub struct TransientImage<'a> {
    pool: &'a RenderTargetPool,
    key: TempImageKey,
    pub handle: ImageHandle,
}

impl RenderTargetPool {
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            images: Default::default(),
        }
    }

    /// Get attachemnt that will return into pool automatically when
    /// when frame is rendererd.
    pub fn image(
        &self,
        format: vk::Format,
        dims: [usize; 2],
        usage: vk::ImageUsageFlags,
    ) -> Result<TransientImage, Error> {
        assert!(
            usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                || usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            "Must be an attachment"
        );
        let (key, handle) = self.get_or_allocate_image(format, dims, usage)?;
        Ok(TransientImage {
            pool: self,
            key,
            handle,
        })
    }

    fn get_or_allocate_image(
        &self,
        format: vk::Format,
        dims: [usize; 2],
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
            let image = self.device.create_image(
                ImageCreateDesc::new(format, dims)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(usage),
            )?;
            Ok((key, image))
        }
    }

    /// Clears pool
    ///
    /// Should be called when backbuffer resolution changed.
    pub fn purge(&self) {
        self.images.lock().clear();
    }

    fn recycle(&self, image: ImageHandle, key: TempImageKey) {
        self.images.lock().entry(key).or_default().push(image);
    }
}

impl Drop for TransientImage<'_> {
    fn drop(&mut self) {
        self.pool.recycle(self.handle, self.key);
    }
}
