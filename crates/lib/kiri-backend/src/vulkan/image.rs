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

use crate::{
    Format, ImageAspect, ImageHandle, ImageLayout, ImageMultisampling, ImageType, ImageUsage,
    ImageViewType, RenderContext,
};

use super::{error::Error, DropList, GpuMemory};

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub dims: [u32; 2],
    pub ty: ImageType,
    pub usage: ImageUsage,
    pub format: Format,
    pub mip_levels: u32,
    pub array_elements: u32,
}

impl From<ImageType> for vk::ImageType {
    fn from(value: ImageType) -> Self {
        match value {
            ImageType::Type1D => vk::ImageType::TYPE_1D,
            ImageType::Type2D => vk::ImageType::TYPE_2D,
            ImageType::Type3D => vk::ImageType::TYPE_3D,
        }
    }
}

impl From<ImageViewType> for vk::ImageViewType {
    fn from(value: ImageViewType) -> Self {
        match value {
            ImageViewType::Type1D => vk::ImageViewType::TYPE_1D,
            ImageViewType::Type1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
            ImageViewType::Type2D => vk::ImageViewType::TYPE_2D,
            ImageViewType::Type2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
            ImageViewType::Type3D => vk::ImageViewType::TYPE_3D,
        }
    }
}

impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(value: ImageUsage) -> Self {
        let mut result = vk::ImageUsageFlags::empty();
        if value.contains(ImageUsage::Sampled) {
            result |= vk::ImageUsageFlags::SAMPLED;
        }
        if value.contains(ImageUsage::Storage) {
            result |= vk::ImageUsageFlags::STORAGE;
        }
        if value.contains(ImageUsage::ColorTarget) {
            result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if value.contains(ImageUsage::DepthStencilTarget) {
            result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if value.contains(ImageUsage::TransferDestination) {
            result |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if value.contains(ImageUsage::Source) {
            result |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        result
    }
}

impl From<ImageAspect> for vk::ImageAspectFlags {
    fn from(value: ImageAspect) -> Self {
        let mut result = vk::ImageAspectFlags::empty();
        if value.contains(ImageAspect::Color) {
            result |= vk::ImageAspectFlags::COLOR;
        }
        if value.contains(ImageAspect::Depth) {
            result |= vk::ImageAspectFlags::DEPTH;
        }
        if value.contains(ImageAspect::Stencil) {
            result |= vk::ImageAspectFlags::STENCIL;
        }

        result
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
    pub ty: Option<ImageViewType>,
    pub format: Option<Format>,
    pub aspect: ImageAspect,
    pub base_mip_level: u32,
    pub level_count: Option<u32>,
}

impl ImageViewDesc {
    pub fn new(aspect: ImageAspect) -> Self {
        Self {
            ty: None,
            format: None,
            aspect,
            base_mip_level: 0,
            level_count: None,
        }
    }

    pub(crate) fn build(&self, image: &Image) -> vk::ImageViewCreateInfo {
        vk::ImageViewCreateInfo::default()
            .format(self.format.unwrap_or(image.desc.format).into())
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .view_type(
                self.ty
                    .unwrap_or_else(|| Self::convert_image_type_to_view_type(image))
                    .into(),
            )
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect.into(),
                base_mip_level: self.base_mip_level,
                level_count: self.level_count.unwrap_or(image.desc.mip_levels),
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(image.raw)
    }

    fn convert_image_type_to_view_type(image: &Image) -> ImageViewType {
        match image.desc.ty {
            ImageType::Type1D if image.desc.array_elements == 1 => ImageViewType::Type1D,
            ImageType::Type1D => ImageViewType::Type1DArray,
            ImageType::Type2D if image.desc.array_elements == 1 => ImageViewType::Type2D,
            ImageType::Type2D => ImageViewType::Type2DArray,
            ImageType::Type3D => ImageViewType::Type3D,
        }
    }
}

impl From<ImageMultisampling> for vk::SampleCountFlags {
    fn from(value: ImageMultisampling) -> Self {
        match value {
            ImageMultisampling::None => vk::SampleCountFlags::TYPE_1,
            ImageMultisampling::Multisampling2 => vk::SampleCountFlags::TYPE_2,
            ImageMultisampling::Multisampling4 => vk::SampleCountFlags::TYPE_4,
            ImageMultisampling::Multisampling8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

impl From<ImageLayout> for vk::ImageLayout {
    fn from(value: ImageLayout) -> Self {
        match value {
            ImageLayout::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageLayout::ColorTarget => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilTarget => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilRead => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ImageLayout::Destination => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::Source => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageCreateDesc<'a> {
    pub dims: [u32; 2],
    pub ty: ImageType,
    pub usage: ImageUsage,
    pub format: Format,
    pub samples: ImageMultisampling,
    pub mip_levels: usize,
    pub array_elements: usize,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    pub(crate) flags: vk::ImageCreateFlags,
    pub(crate) tiling: vk::ImageTiling,
}

impl<'a> ImageCreateDesc<'a> {
    pub fn new(format: Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: ImageType::Type2D,
            usage: ImageUsage::empty(),
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::None,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn texture(format: Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: ImageType::Type2D,
            usage: ImageUsage::Sampled | ImageUsage::TransferDestination,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::None,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn cubemap(format: Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: ImageType::Type2D,
            usage: ImageUsage::Sampled | ImageUsage::TransferDestination,
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::None,
            mip_levels: 1,
            array_elements: 6,
            dedicated: false,
            name: None,
        }
    }

    pub fn color_target(format: Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: ImageType::Type2D,
            usage: ImageUsage::ColorTarget,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::None,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn depth_stencil_target(format: Format, dims: [u32; 2]) -> Self {
        Self {
            dims,
            ty: ImageType::Type2D,
            usage: ImageUsage::DepthStencilTarget,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::None,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn ty(mut self, value: ImageType) -> Self {
        self.ty = value;
        self
    }

    pub fn usage(mut self, value: ImageUsage) -> Self {
        self.usage = value;
        self
    }

    pub fn sampled(mut self) -> Self {
        self.usage |= ImageUsage::Sampled;
        self
    }

    pub fn samples(mut self, value: ImageMultisampling) -> Self {
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
            .usage(self.usage.into())
            .flags(self.flags)
            .format(self.format.into())
            .samples(self.samples.into())
            .image_type(self.ty.into())
            .tiling(self.tiling)
            .extent(self.create_dims())
    }

    fn create_dims(&self) -> vk::Extent3D {
        match self.ty {
            ImageType::Type1D => vk::Extent3D {
                width: self.dims[0],
                height: 1,
                depth: 1,
            },
            ImageType::Type2D => vk::Extent3D {
                width: self.dims[0],
                height: self.dims[1],
                depth: 1,
            },
            ImageType::Type3D => vk::Extent3D {
                width: self.dims[0],
                height: self.dims[1],
                depth: self.array_elements as u32,
            },
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

    pub(crate) fn subresource(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_array_layer(0)
            .base_mip_level(0)
            .layer_count(self.desc.array_elements)
            .level_count(self.desc.mip_levels)
    }
}

impl<'game> RenderContext<'game> {
    pub fn create_image(
        &self,
        desc: ImageCreateDesc,
        aspect: ImageAspect,
        data: Option<&[ImageSubresourceData]>,
    ) -> Result<ImageHandle, Error> {
        let image = Image::new(self, desc)?;
        if let Some(data) = data {
            self.staging.lock().upload_image(self, &image, data)?;
        }
        self.insert_image(image, aspect)
    }

    pub fn update_image(
        &self,
        handle: ImageHandle,
        desc: ImageCreateDesc,
        aspect: ImageAspect,
        data: Option<&[ImageSubresourceData]>,
    ) -> Result<(), Error> {
        let image = Image::new(self, desc)?;
        if let Some(data) = data {
            self.staging.lock().upload_image(self, &image, data)?;
        }
        let view = unsafe {
            self.device
                .create_image_view(&ImageViewDesc::new(aspect).build(&image), None)
        }?;
        let desc = image.desc;
        let (old_view, old_image) = self
            .images
            .write()
            .replace_hot_cold(handle, view, image)
            .ok_or(Error::InvalidImageHandle(handle))?;
        self.with_drop_list(|drop_list| {
            drop_list.drop_view(old_view);
            old_image.free(drop_list);
        });
        if desc.usage.contains(ImageUsage::Sampled) {
            self.sampled_images_to_update.lock().insert(handle);
        }
        if desc.usage.contains(ImageUsage::Storage) {
            self.storage_images_to_update.lock().insert(handle);
        }
        Ok(())
    }

    pub(crate) fn register_image(
        &self,
        image: vk::Image,
        desc: ImageDesc,
        aspect: ImageAspect,
    ) -> Result<ImageHandle, Error> {
        let image = Image::internal(image, desc);
        self.insert_image(image, aspect)
    }

    fn insert_image(&self, image: Image, aspect: ImageAspect) -> Result<ImageHandle, Error> {
        let view = unsafe {
            self.device
                .create_image_view(&ImageViewDesc::new(aspect).build(&image), None)
        }?;
        let desc = image.desc;
        let handle = self.images.write().push(view, image);
        if desc.usage.contains(ImageUsage::Sampled) {
            self.sampled_images_to_update.lock().insert(handle);
        }
        if desc.usage.contains(ImageUsage::Storage) {
            self.storage_images_to_update.lock().insert(handle);
        }
        Ok(handle)
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
