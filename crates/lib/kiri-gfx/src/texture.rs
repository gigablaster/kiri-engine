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

use std::sync::Arc;

use crate::{Error, ImageHandle, ImageUploadData, Renderer};
use kiri_backend::{ash::vk, ImageCreateDesc};

#[derive(Debug)]
pub struct Texture {
    renderer: Arc<Renderer>,
    pub image: ImageHandle,
}

impl Drop for Texture {
    fn drop(&mut self) {
        self.renderer.destroy_image(self.image);
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
        let mut desc = ImageCreateDesc::texture(self.format, self.dims)
            .sampled()
            .transfer_desitnation();
        if let Some(name) = self.name {
            desc = desc.name(name);
        }
        Ok(Texture {
            renderer: renderer.clone(),
            image: renderer.create_image(desc, self.data)?,
        })
    }
}

impl Texture {
    pub fn update(&self, builder: TextureBuilder) -> Result<(), Error> {
        let mut desc = ImageCreateDesc::texture(builder.format, builder.dims)
            .sampled()
            .transfer_desitnation();
        if let Some(name) = builder.name {
            desc = desc.name(name);
        }
        self.renderer.update_image(self.image, desc, builder.data)?;
        Ok(())
    }
}
