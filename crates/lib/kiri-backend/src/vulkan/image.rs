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
use ash::vk::{self, ImageView};
use kiri_common::Pool;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use super::{
    drop_list::DropList, GpuMemoryBlock, ImageCreateDesc, ImageHandle, ImageUploadData,
    ImageViewDesc, RenderDevice,
};

pub type ImagePool = Pool<ImageData>;

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
    fn build(&self, image: &ImageData) -> vk::ImageViewCreateInfo {
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
                level_count: self.level_count.unwrap_or(image.desc.mip_levels) as u32,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(image.raw)
    }

    fn convert_image_type_to_view_type(image: &ImageData) -> vk::ImageViewType {
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

impl<'a> ImageCreateDesc<'a> {
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

#[derive(Debug)]
pub struct ImageData {
    pub raw: vk::Image,
    pub desc: ImageDesc,
    pub memory: Option<GpuMemoryBlock>,
    pub views: RwLock<HashMap<ImageViewDesc, vk::ImageView>>,
}

impl ImageData {
    pub fn free(mut self, drop_list: &mut DropList) {
        if let Some(memory) = self.memory.take() {
            self.free_views(drop_list);
            drop_list.drop_image(self.raw);
            drop_list.drop_memory(memory);
        }
    }

    pub fn free_views(&self, drop_list: &mut DropList) {
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
}

impl RenderDevice {
    /// Wraps external image
    ///
    /// Image won't be destroyed when instance is dropped. But views will be freed.
    pub(crate) fn crate_external_image(
        &self,
        image: vk::Image,
        desc: ImageDesc,
        name: Option<&str>,
    ) -> ImageHandle {
        if let Some(name) = name {
            self.set_object_name(image, name);
        }

        let image = ImageData {
            raw: image,
            desc,
            views: Default::default(),
            memory: None,
        };
        self.images.write().push(image)
    }

    /// Creates new image
    ///
    /// Including memory allocation. All resources will be freed when instance
    /// is dropped.    
    pub fn create_image(
        &self,
        desc: ImageCreateDesc,
        data: Option<&[ImageUploadData]>,
    ) -> Result<ImageHandle, Error> {
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
        if let Some(data) = data {
            self.staging
                .lock()
                .upload_image(&self.raw, image, desc, data)?;
        }
        let image = ImageData {
            raw: image,
            desc,
            views: Default::default(),
            memory: Some(memory),
        };
        Ok(self.images.write().push(image))
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
