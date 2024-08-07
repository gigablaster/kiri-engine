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

use ash::vk;

use crate::{ImageHandle, RenderContext};

use super::{error::Error, DropList, GpuMemory};

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub dims: [u32; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub array_elements: u32,
    pub tiling: vk::ImageTiling,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
    pub ty: Option<vk::ImageViewType>,
    pub format: Option<vk::Format>,
    pub aspect: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: Option<u32>,
}

impl ImageViewDesc {
    pub fn new(aspect: vk::ImageAspectFlags) -> Self {
        Self {
            ty: None,
            format: None,
            aspect,
            base_mip_level: 0,
            level_count: None,
        }
    }
    pub fn color() -> Self {
        Self {
            ty: None,
            format: None,
            aspect: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: None,
        }
    }

    pub fn depth() -> Self {
        Self {
            ty: None,
            format: None,
            aspect: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: None,
        }
    }

    pub fn view_type(mut self, view_type: vk::ImageViewType) -> Self {
        self.ty = Some(view_type);
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = Some(format);
        self
    }

    pub fn base_mip_level(mut self, base_mip_level: u32) -> Self {
        self.base_mip_level = base_mip_level;
        self
    }

    pub fn level_count(mut self, level_count: u32) -> Self {
        self.level_count = Some(level_count);
        self
    }

    pub(crate) fn build(&self, image: &Image) -> vk::ImageViewCreateInfo {
        vk::ImageViewCreateInfo::default()
            .format(self.format.unwrap_or(image.desc.format))
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .view_type(
                self.ty
                    .unwrap_or_else(|| Self::convert_image_type_to_view_type(image)),
            )
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect,
                base_mip_level: self.base_mip_level,
                level_count: self.level_count.unwrap_or(image.desc.mip_levels),
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(image.raw)
    }

    fn convert_image_type_to_view_type(image: &Image) -> vk::ImageViewType {
        match image.desc.ty {
            vk::ImageType::TYPE_1D if image.desc.array_elements == 1 => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D_ARRAY,
            vk::ImageType::TYPE_2D if image.desc.array_elements == 1 => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageCreateDesc<'a> {
    pub dims: [u32; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub mip_levels: usize,
    pub array_elements: usize,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    pub(crate) flags: vk::ImageCreateFlags,
    pub(crate) tiling: vk::ImageTiling,
}

impl<'a> ImageCreateDesc<'a> {
    pub fn new(format: vk::Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::empty(),
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn texture(format: vk::Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn cubemap(format: vk::Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 6,
            dedicated: false,
            name: None,
        }
    }

    pub fn color_attachment(format: vk::Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn depth_stencil_attachment(format: vk::Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn ty(mut self, value: vk::ImageType) -> Self {
        self.ty = value;
        self
    }

    pub fn usage(mut self, value: vk::ImageUsageFlags) -> Self {
        self.usage = value;
        self
    }

    pub fn sampled(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::SAMPLED;
        self
    }

    pub fn samples(mut self, value: vk::SampleCountFlags) -> Self {
        self.samples = value;
        self
    }

    pub fn mip_levels(mut self, value: usize) -> Self {
        self.mip_levels = value;
        self
    }

    pub fn array_elements(mut self, value: usize) -> Self {
        self.array_elements = value;
        self
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    fn build(&self) -> vk::ImageCreateInfo {
        vk::ImageCreateInfo::default()
            .array_layers(self.array_elements as _)
            .mip_levels(self.mip_levels as _)
            .usage(self.usage)
            .flags(self.flags)
            .format(self.format)
            .samples(self.samples)
            .image_type(self.ty)
            .tiling(self.tiling)
            .extent(self.create_dims())
    }

    fn create_dims(&self) -> vk::Extent3D {
        match self.ty {
            vk::ImageType::TYPE_1D => vk::Extent3D {
                width: self.dims[0],
                height: 1,
                depth: 1,
            },
            vk::ImageType::TYPE_2D => vk::Extent3D {
                width: self.dims[0],
                height: self.dims[1],
                depth: 1,
            },
            vk::ImageType::TYPE_3D => vk::Extent3D {
                width: self.dims[0],
                height: self.dims[1],
                depth: self.array_elements as u32,
            },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ImageSubresourceData<'a> {
    pub data: &'a [u8],
    pub row_pitch: usize,
}

#[derive(Debug)]
pub(crate) struct Image {
    pub raw: vk::Image,
    pub desc: ImageDesc,
    memory: Option<GpuMemory>,
}

pub(crate) enum ImageSubresourceRange {
    All,
    Level(u32),
    LevelAndMip(u32, u32),
}

impl Image {
    pub(crate) fn internal(image: vk::Image, desc: ImageDesc) -> Self {
        Self {
            raw: image,
            desc,
            memory: None,
        }
    }

    pub fn new(context: &RenderContext, desc: ImageCreateDesc) -> Result<Self, Error> {
        let image = unsafe { context.device.create_image(&desc.build(), None) }?;
        if let Some(name) = desc.name {
            context.set_object_name(image, name);
        }
        let mut requirements = unsafe { context.device.get_image_memory_requirements(image) };
        // Workaround - gpu_alloc returns wrong offset when size < aligment.
        requirements.size = requirements.size.max(requirements.alignment);
        let memory = context.allocate(
            requirements,
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            desc.dedicated,
        )?;
        unsafe {
            context
                .device
                .bind_image_memory(image, *memory.memory(), memory.offset())
        }?;

        Ok(Self {
            raw: image,
            desc: ImageDesc {
                dims: desc.dims,
                ty: desc.ty,
                usage: desc.usage,
                format: desc.format,
                tiling: desc.tiling,
                mip_levels: desc.mip_levels as u32,
                array_elements: desc.array_elements as u32,
            },
            memory: Some(memory),
        })
    }

    pub(crate) fn free(mut self, drop_list: &mut DropList) {
        if let Some(memory) = self.memory.take() {
            drop_list.drop_memory(memory);
            drop_list.drop_image(self.raw);
        }
    }

    pub(crate) fn subresource(
        &self,
        range: ImageSubresourceRange,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageSubresourceRange {
        match range {
            ImageSubresourceRange::All => vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_array_layer(0)
                .base_mip_level(0)
                .layer_count(self.desc.array_elements)
                .level_count(self.desc.mip_levels),
            ImageSubresourceRange::Level(level) => vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_array_layer(level)
                .base_mip_level(0)
                .layer_count(1)
                .level_count(self.desc.mip_levels),
            ImageSubresourceRange::LevelAndMip(level, mip) => vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_array_layer(level)
                .base_mip_level(mip)
                .layer_count(1)
                .level_count(1),
        }
    }
}

impl RenderContext {
    pub fn create_image(
        &self,
        desc: ImageCreateDesc,
        aspect: vk::ImageAspectFlags,
        data: Option<&[ImageSubresourceData]>,
    ) -> Result<ImageHandle, Error> {
        let image = Image::new(&self, desc)?;
        if let Some(data) = data {
            self.staging.lock().upload_image(&self, &image, data)?;
        }
        self.insert_image(image, aspect)
    }

    pub(crate) fn register_image(
        &self,
        image: vk::Image,
        desc: ImageDesc,
        aspect: vk::ImageAspectFlags,
    ) -> Result<ImageHandle, Error> {
        let image = Image::internal(image, desc);
        self.insert_image(image, aspect)
    }

    fn insert_image(
        &self,
        image: Image,
        aspect: vk::ImageAspectFlags,
    ) -> Result<ImageHandle, Error> {
        let view = unsafe {
            self.device
                .create_image_view(&ImageViewDesc::new(aspect).build(&image), None)
        }?;
        Ok(self.images.write().push(view, image))
    }

    pub fn destroy_image(&self, handle: ImageHandle) {
        if let Some((view, image)) = self.images.write().remove(handle) {
            self.with_drop_list(|drop_list| {
                drop_list.drop_view(view);
                image.free(drop_list);
            })
        }
    }
}
