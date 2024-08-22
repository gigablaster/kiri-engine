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

use std::{borrow::Borrow, collections::HashMap, ops::Deref};

use ash::vk;
use kiri_backend::{Image, ImageCreateDesc};
use parking_lot::Mutex;

use crate::{Error, ImageHandle, Renderer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resolution {
    Full,
    ScaledDown(u32),
    Fixed(u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TempImageKey {
    pub resolution: Resolution,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Debug, Default)]
pub(super) struct TempImagePool {
    images: Mutex<HashMap<TempImageKey, Vec<ImageHandle>>>,
}

#[derive(Debug)]
pub struct TempImageGuard<'a> {
    pool: &'a TempImagePool,
    key: TempImageKey,
    pub handle: ImageHandle,
}

impl Resolution {
    pub fn get_dims(self, relative: [u32; 2]) -> [u32; 2] {
        match self {
            Resolution::Full => relative,
            Resolution::ScaledDown(scale) => [relative[0] / scale, relative[1] / scale],
            Resolution::Fixed(width, height) => [width, height],
        }
    }
}

impl TempImagePool {
    pub fn get(
        &self,
        renderer: &Renderer,
        resolution: Resolution,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        backbuffer: &Image,
    ) -> Result<TempImageGuard, Error> {
        let mut images = self.images.lock();
        let key = TempImageKey {
            resolution,
            format,
            usage,
        };
        let group = images.entry(key).or_default();
        if let Some(image) = group.pop() {
            Ok(TempImageGuard {
                pool: self,
                key,
                handle: image,
            })
        } else {
            let image = renderer.create_image(
                ImageCreateDesc::new(format, resolution.get_dims(backbuffer.desc.dims))
                    .usage(usage),
                None,
            )?;
            Ok(TempImageGuard {
                pool: self,
                key,
                handle: image,
            })
        }
    }

    pub fn purge(&self, renderer: &Renderer) {
        let mut images = self.images.lock();
        images
            .drain()
            .for_each(|(_, mut group)| group.drain(..).for_each(|x| renderer.destroy_image(x)))
    }

    fn recycle(&self, image: ImageHandle, key: TempImageKey) {
        let mut images = self.images.lock();
        let group = images.entry(key).or_default();
        group.push(image);
    }
}

impl<'a> Drop for TempImageGuard<'a> {
    fn drop(&mut self) {
        self.pool.recycle(self.handle, self.key);
    }
}
