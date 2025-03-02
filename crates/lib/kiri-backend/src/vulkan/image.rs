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

use std::collections::HashMap;

use crate::Error;
use ash::vk::{self};
use kiri_common::Pool;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use super::{drop_list::DropList, GpuMemoryBlock, GraphicsDevice, ImageHandle};

pub(crate) type ImagePool = Pool<Image>;

#[derive(Debug, Clone, Copy)]
pub struct ImageUploadData<'a> {
    pub data: &'a [u8],
}

impl<'a> ImageUploadData<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub dims: [usize; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub mip_levels: usize,
    pub array_elements: usize,
}

impl ImageDesc {
    pub fn aspect(&self) -> f32 {
        self.dims[0] as f32 / self.dims[1] as f32
    }
}

impl ImageViewDesc {
    pub fn new(aspect: vk::ImageAspectFlags) -> Self {
        Self {
            ty: None,
            format: None,
            aspect,
            base_mip_level: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
        }
    }

    pub fn color() -> Self {
        Self::new(vk::ImageAspectFlags::COLOR)
    }

    pub fn depth() -> Self {
        Self::new(vk::ImageAspectFlags::DEPTH)
    }

    pub fn base_mip_level(mut self, value: usize) -> Self {
        self.base_mip_level = value;
        self
    }

    pub fn mip_count(mut self, value: usize) -> Self {
        self.mip_count = Some(value);
        self
    }

    pub fn base_layer(mut self, value: usize) -> Self {
        self.base_layer = value;
        self
    }

    pub fn layer_count(mut self, value: usize) -> Self {
        self.layer_count = Some(value);
        self
    }

    fn build(&self, image: &Image) -> vk::ImageViewCreateInfo {
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
                base_mip_level: self.base_mip_level as u32,
                level_count: self.mip_count.unwrap_or(image.desc.mip_levels) as u32,
                base_array_layer: self.base_layer as u32,
                layer_count: self.layer_count.unwrap_or(1) as u32,
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
            ty => panic!("Unknown image type {:?}", ty),
        }
    }
}

impl ImageCreateDesc<'_> {
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
            .extent(self.to_extent())
            .initial_layout(self.initial_layout.unwrap_or(vk::ImageLayout::UNDEFINED))
    }

    fn to_extent(self) -> vk::Extent3D {
        match self.ty {
            vk::ImageType::TYPE_1D => vk::Extent3D {
                width: self.dims[0] as u32,
                height: 1,
                depth: 1,
            },
            vk::ImageType::TYPE_2D => vk::Extent3D {
                width: self.dims[0] as u32,
                height: self.dims[1] as u32,
                depth: 1,
            },
            vk::ImageType::TYPE_3D => vk::Extent3D {
                width: self.dims[0] as u32,
                height: self.dims[1] as u32,
                depth: self.array_elements as u32,
            },
            ty => panic!("Unknown image type {:?}", ty),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
    pub ty: Option<vk::ImageViewType>,
    pub format: Option<vk::Format>,
    pub aspect: vk::ImageAspectFlags,
    pub base_mip_level: usize,
    pub mip_count: Option<usize>,
    pub base_layer: usize,
    pub layer_count: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageCreateDesc<'a> {
    pub dims: [usize; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub mip_levels: usize,
    pub array_elements: usize,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    pub flags: vk::ImageCreateFlags,
    pub tiling: vk::ImageTiling,
    pub initial_layout: Option<vk::ImageLayout>,
}

impl<'a> ImageCreateDesc<'a> {
    pub fn new(format: vk::Format, dims: [usize; 2]) -> Self {
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
            initial_layout: None,
        }
    }

    pub fn texture(format: vk::Format, dims: [usize; 2]) -> Self {
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
            initial_layout: None,
        }
    }

    pub fn cubemap(format: vk::Format, dims: [usize; 2]) -> Self {
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
            initial_layout: None,
        }
    }

    pub fn color_target(format: vk::Format, dims: [usize; 2]) -> Self {
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
            initial_layout: Some(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        }
    }

    pub fn depth_stencil_target(format: vk::Format, dims: [usize; 2]) -> Self {
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
            initial_layout: Some(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        }
    }

    pub fn transfer_desitnation(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_DST;
        self
    }

    pub fn trasfer_source(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        self
    }

    pub fn storage(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::STORAGE;
        self
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

    pub fn transient(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        self
    }

    pub fn initial_layout(mut self, value: vk::ImageLayout) -> Self {
        self.initial_layout = Some(value);
        self
    }
}

#[derive(Debug)]
pub struct Image {
    raw: vk::Image,
    pub desc: ImageDesc,
    memory: Option<GpuMemoryBlock>,
    views: RwLock<HashMap<ImageViewDesc, vk::ImageView>>,
}

impl Image {
    pub(crate) fn external(raw: vk::Image, desc: ImageDesc) -> Self {
        Self {
            raw,
            desc,
            memory: None,
            views: Default::default(),
        }
    }

    pub(crate) fn free(mut self, drop_list: &mut DropList) {
        if let Some(memory) = self.memory.take() {
            self.free_views(drop_list);
            drop_list.drop_image(self.raw);
            drop_list.drop_memory(memory);
        }
    }

    pub(crate) fn free_views(&self, drop_list: &mut DropList) {
        self.views
            .write()
            .drain()
            .for_each(|(_, view)| drop_list.drop_view(view));
    }

    pub fn get_or_create_view(
        &self,
        device: &ash::Device,
        desc: ImageViewDesc,
    ) -> Result<vk::ImageView, Error> {
        let views = self.views.upgradable_read();
        if let Some(view) = views.get(&desc) {
            Ok(*view)
        } else {
            let mut views = RwLockUpgradableReadGuard::upgrade(views);
            if let Some(view) = views.get(&desc) {
                Ok(*view)
            } else {
                let create_info = desc.build(self);
                let view = unsafe { device.create_image_view(&create_info, None) }?;
                views.insert(desc, view);
                Ok(view)
            }
        }
    }

    pub(crate) fn raw(&self) -> vk::Image {
        self.raw
    }
}

impl GraphicsDevice {
    /// Creates new image
    ///
    /// Including memory allocation. All resources will be freed when instance
    /// is dropped.    
    pub fn create_image(&self, desc: ImageCreateDesc) -> Result<ImageHandle, Error> {
        let image = unsafe { self.raw.create_image(&desc.build(), None) }?;
        if let Some(name) = desc.name {
            self.set_object_name(image, name);
        }
        let mut requirements = unsafe { self.raw.get_image_memory_requirements(image) };
        // Workaround - gpu_alloc returns wrong offset when size < aligment.
        requirements.size = requirements.size.max(requirements.alignment);

        let mut memory_usage = gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS;
        if desc
            .usage
            .contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
        {
            memory_usage |= gpu_alloc::UsageFlags::TRANSIENT;
        }
        let memory = self.allocate(requirements, memory_usage, false)?;
        unsafe {
            self.raw
                .bind_image_memory(image, *memory.memory(), memory.offset())
        }?;
        let desc = ImageDesc {
            dims: desc.dims,
            ty: desc.ty,
            usage: desc.usage,
            format: desc.format,
            mip_levels: desc.mip_levels,
            array_elements: desc.array_elements,
        };
        let image = Image {
            raw: image,
            desc,
            views: Default::default(),
            memory: Some(memory),
        };
        Ok(self.images.write().push(image))
    }

    pub fn upload_image_data<'a>(
        &self,
        handle: ImageHandle,
        layer: usize,
        data: impl IntoIterator<Item = ImageUploadData<'a>>,
    ) -> Result<(), Error> {
        let images = self.images.read();
        let image = images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?;
        let raw = image.raw();
        let desc = image.desc;
        drop(images);
        self.staging
            .lock()
            .upload_image(&self.raw, raw, layer, desc, data)
    }

    pub fn clear_image_views(&self, handle: ImageHandle) {
        let images = self.images.read();
        let mut drop_list = self.current_drop_list.lock();
        if let Some(image) = images.get(handle) {
            image.free_views(&mut drop_list);
        };
    }

    /// Gets or creates image view
    ///
    /// Image views are managed by image itself.
    pub fn get_or_create_image_view(
        &self,
        handle: ImageHandle,
        desc: ImageViewDesc,
    ) -> Result<vk::ImageView, Error> {
        self.images
            .read()
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .get_or_create_view(&self.raw, desc)
    }

    pub fn destroy_image(&self, handle: ImageHandle) {
        self.images_to_destroy.lock().push(handle);
    }
}
