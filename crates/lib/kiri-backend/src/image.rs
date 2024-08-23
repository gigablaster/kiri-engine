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

use ash::vk;
use log::warn;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use crate::RenderDevice;

use super::{error::Error, DropList, GpuMemory};

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub dims: [u32; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub array_elements: u32,
}

trait IsRenderTarget {
    fn is_render_target(&self) -> bool;
}

impl IsRenderTarget for vk::ImageUsageFlags {
    fn is_render_target(&self) -> bool {
        self.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            || self.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
    }
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
            ty => panic!("Unknown image type {:?}", ty),
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
    pub flags: vk::ImageCreateFlags,
    pub tiling: vk::ImageTiling,
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
            samples: vk::SampleCountFlags::empty(),
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

    pub fn color_target(format: vk::Format, dims: [u32; 2]) -> Self {
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

    pub fn depth_stencil_target(format: vk::Format, dims: [u32; 2]) -> Self {
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

    pub fn transfer_desitnation(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_DST;
        self
    }

    pub fn trasfer_source(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
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
            ty => panic!("Unknown image type {:?}", ty),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageSubresource {
    All,
    Level(usize),
    LevelAndMip(usize, usize),
}

#[derive(Debug)]
pub struct Image {
    device: Arc<RenderDevice>,
    pub raw: vk::Image,
    pub desc: ImageDesc,
    memory: Option<GpuMemory>,
    views: RwLock<HashMap<ImageViewDesc, vk::ImageView>>,
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            self.device.with_drop_list(|drop_list| {
                drop_list.drop_memory(memory);
                drop_list.drop_image(self.raw);
                self.clear_views_impl(drop_list);
            })
        } else {
            if self.desc.usage.is_render_target() {
                self.device.with_drop_list(|drop_list| {
                    drop_list.drop_image(self.raw);
                })
            }
            self.clear_views();
        }
    }
}

/// Wraps vulkan image
///
/// Keep all resources, everything will be freed as soon as image is dropped.
/// Keeps tracking for associated image views.
impl Image {
    /// Wraps external image
    ///
    /// Image won't be destroyed when instance is dropped. But views will be freed.
    pub fn external(
        device: &Arc<RenderDevice>,
        image: vk::Image,
        desc: ImageDesc,
        name: Option<&str>,
    ) -> Self {
        assert!(!desc.usage.is_render_target());
        if let Some(name) = name {
            device.set_object_name(image, name);
        }
        Self {
            device: device.clone(),
            raw: image,
            desc,
            memory: None,
            views: Default::default(),
        }
    }

    /// Creates new image
    ///
    /// Including memory allocation. All resources will be freed when instance
    /// is dropped.    
    pub fn new(device: &Arc<RenderDevice>, desc: ImageCreateDesc) -> Result<Self, Error> {
        let image = unsafe { device.raw.create_image(&desc.build(), None) }?;
        if let Some(name) = desc.name {
            device.set_object_name(image, name);
        }
        let mut requirements = unsafe { device.raw.get_image_memory_requirements(image) };
        // Workaround - gpu_alloc returns wrong offset when size < aligment.
        requirements.size = requirements.size.max(requirements.alignment);

        let memory = if desc.usage.is_render_target() {
            match device.allocate_render_target(requirements) {
                Ok((memory, offset)) => {
                    unsafe { device.raw.bind_image_memory(image, memory, offset) }?;
                    None
                }
                Err(Error::OutOfDeviceMemory) => {
                    warn!("Failed to allocate from main render target pool - fallback to normal allocator");
                    let memory = device.allocate_memory(
                        requirements,
                        gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                        desc.dedicated,
                    )?;
                    unsafe {
                        device
                            .raw
                            .bind_image_memory(image, *memory.memory(), memory.offset())
                    }?;
                    Some(memory)
                }
                Err(other) => return Err(other),
            }
        } else {
            let memory = device.allocate_memory(
                requirements,
                gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                desc.dedicated,
            )?;
            unsafe {
                device
                    .raw
                    .bind_image_memory(image, *memory.memory(), memory.offset())
            }?;
            Some(memory)
        };
        Ok(Self {
            device: device.clone(),
            raw: image,
            desc: ImageDesc {
                dims: desc.dims,
                ty: desc.ty,
                usage: desc.usage,
                format: desc.format,
                mip_levels: desc.mip_levels as u32,
                array_elements: desc.array_elements as u32,
            },
            memory,
            views: Default::default(),
        })
    }

    fn clear_views_impl(&self, drop_list: &mut DropList) {
        self.views
            .write()
            .drain()
            .for_each(|(_, view)| drop_list.drop_view(view))
    }

    /// Gets or creates image view
    ///
    /// Image views are managed by image itself.
    pub fn view(&self, desc: ImageViewDesc) -> Result<vk::ImageView, Error> {
        let views = self.views.upgradable_read();
        if let Some(view) = views.get(&desc) {
            Ok(*view)
        } else {
            let mut views = RwLockUpgradableReadGuard::upgrade(views);
            if let Some(view) = views.get(&desc) {
                Ok(*view)
            } else {
                let view = self.create_view(desc)?;
                views.insert(desc, view);
                Ok(view)
            }
        }
    }

    /// Clear all views created for this image
    pub fn clear_views(&self) {
        self.device.with_drop_list(|drop_list| {
            self.clear_views_impl(drop_list);
        })
    }

    fn create_view(&self, desc: ImageViewDesc) -> Result<vk::ImageView, Error> {
        let create_info = desc.build(self);
        let view = unsafe { self.device.raw.create_image_view(&create_info, None) }?;
        Ok(view)
    }

    pub fn subresource(
        &self,
        aspect: vk::ImageAspectFlags,
        range: ImageSubresource,
    ) -> vk::ImageSubresourceRange {
        match range {
            ImageSubresource::All => vk::ImageSubresourceRange::default().aspect_mask(aspect),
            ImageSubresource::Level(level) => vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_array_layer(level as _)
                .layer_count(1),
            ImageSubresource::LevelAndMip(level, mip) => vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_array_layer(level as _)
                .layer_count(1)
                .base_mip_level(mip as _)
                .level_count(1),
        }
    }
}
