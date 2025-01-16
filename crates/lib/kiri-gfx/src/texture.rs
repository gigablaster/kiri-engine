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

use std::sync::Arc;

use crate::{Error, ImageHandle, Renderer};
use kiri_backend::{ash::vk, Image, ImageCreateDesc, ImageUploadData, ImageViewDesc};

#[derive(Debug)]
pub struct Texture {
    renderer: Arc<Renderer>,
    pub handle: ImageHandle,
}

impl Drop for Texture {
    fn drop(&mut self) {
        self.renderer.remove_image(self.handle);
    }
}

pub struct TextureBuilder<'a> {
    pub format: vk::Format,
    pub dims: [u32; 2],
    pub data: Option<&'a [ImageUploadData<'a>]>,
    pub name: Option<&'a str>,
}

impl<'a> TextureBuilder<'a> {
    pub fn new(format: vk::Format, dims: [u32; 2]) -> Self {
        Self {
            format,
            dims,
            data: None,
            name: None,
        }
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    pub fn data(mut self, data: &'a [ImageUploadData]) -> Self {
        self.data = Some(data);
        self
    }

    pub fn build(self, renderer: &Arc<Renderer>) -> Result<Texture, Error> {
        let image = self.create_image(renderer)?;
        Ok(Texture {
            renderer: renderer.clone(),
            handle: renderer
                .register_image(image, ImageViewDesc::new(vk::ImageAspectFlags::COLOR))?,
        })
    }

    fn create_image(self, renderer: &Arc<Renderer>) -> Result<Arc<Image>, Error> {
        let mut desc = ImageCreateDesc::texture(self.format, self.dims)
            .sampled()
            .mip_levels(self.data.map(|x| x.len() as u32).unwrap_or(1))
            .transfer_desitnation();
        if let Some(name) = self.name {
            desc = desc.name(name);
        }
        Ok(Arc::new(Image::new(&renderer.device, desc, self.data)?))
    }
}

impl Texture {
    pub fn update(&self, builder: TextureBuilder) -> Result<(), Error> {
        self.renderer
            .replace_image(self.handle, builder.create_image(&self.renderer)?)?;
        Ok(())
    }
}
