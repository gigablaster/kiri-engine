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

use std::{
    mem::{self},
    slice,
    sync::Arc,
};

use ash::vk::{self};
use kiri_backend::{
    create_descriptor_layout, DescriptorSetDesc, Image, ImageViewDesc, RenderDevice,
};
use kiri_common::{DefaultPoolLimits, Handle, Pool, PoolLimits, TempList};
use lazy_static::lazy_static;

use crate::Error;

pub type ImageHandle = Handle<(Arc<Image>, vk::ImageAspectFlags, vk::ImageLayout)>;

type ImagePool = Pool<(Arc<Image>, vk::ImageAspectFlags, vk::ImageLayout)>;

const SAMPLED_IMAGES_BINDING: usize = 0;
const STORAGE_IMAGES_BINDINGS: usize = 1;

lazy_static! {
    static ref BINDLESS_LAYOUT_DESC: DescriptorSetDesc = [
        (
            SAMPLED_IMAGES_BINDING,
            (
                "sampled".to_owned(),
                vk::DescriptorType::SAMPLED_IMAGE,
                DefaultPoolLimits::max_index() as usize
            ),
        ),
        (
            STORAGE_IMAGES_BINDINGS,
            (
                "storage".to_owned(),
                vk::DescriptorType::STORAGE_IMAGE,
                DefaultPoolLimits::max_index() as usize
            ),
        ),
    ]
    .iter()
    .cloned()
    .collect();
}

#[derive(Debug)]
pub struct BindlessManager {
    pub device: Arc<RenderDevice>,
    images: ImagePool,
    sampled_images_to_update: Vec<ImageHandle>,
    storage_images_to_update: Vec<ImageHandle>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

pub enum FrameState {
    Rendered,
    NeedRecreateSwapchain,
}

impl Drop for BindlessManager {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_descriptor_pool(self.pool, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

impl BindlessManager {
    pub fn new(device: &Arc<RenderDevice>) -> Result<Self, Error> {
        let layout = create_descriptor_layout(
            device,
            vk::ShaderStageFlags::ALL,
            &BINDLESS_LAYOUT_DESC,
            true,
        )?;
        let bindings = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: DefaultPoolLimits::max_index(),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: DefaultPoolLimits::max_index(),
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&bindings)
            .max_sets(1);
        let pool = unsafe { device.raw.create_descriptor_pool(&pool_info, None) }?;
        let mut allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(slice::from_ref(&layout));
        allocate_info.descriptor_set_count = 1;
        let bindless = unsafe { device.raw.allocate_descriptor_sets(&allocate_info) }?.remove(0);
        Ok(Self {
            device: device.clone(),
            images: Default::default(),
            sampled_images_to_update: Default::default(),
            storage_images_to_update: Default::default(),
            layout,
            pool,
            descriptor_set: bindless,
        })
    }

    pub fn import_image(
        &mut self,
        image: Arc<Image>,
        aspect: vk::ImageAspectFlags,
        layout: vk::ImageLayout,
    ) -> ImageHandle {
        let usage = image.desc.usage;
        let handle = self.images.push((image, aspect, layout));
        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            self.sampled_images_to_update.push(handle);
        }
        if usage.contains(vk::ImageUsageFlags::STORAGE) {
            self.storage_images_to_update.push(handle);
        }
        handle
    }

    pub fn update_image(
        &mut self,
        handle: ImageHandle,
        image: Arc<Image>,
        aspect: vk::ImageAspectFlags,
        layout: vk::ImageLayout,
    ) {
        let usage = image.desc.usage;
        if let Some(_) = self.images.replace(handle, (image, aspect, layout)) {
            if usage.contains(vk::ImageUsageFlags::SAMPLED) {
                self.sampled_images_to_update.push(handle);
            }
            if usage.contains(vk::ImageUsageFlags::STORAGE) {
                self.storage_images_to_update.push(handle);
            }
        };
    }

    pub fn remove_image(&mut self, handle: ImageHandle) {
        self.images.remove(handle);
    }

    pub fn resolve_image(&self, handle: ImageHandle) -> Result<&Arc<Image>, Error> {
        Ok(&self
            .images
            .get(handle)
            .ok_or(Error::InvalidImageHandle(handle))?
            .0)
    }

    pub fn update_descriptors(&mut self) -> Result<(), Error> {
        self.sampled_images_to_update.sort();
        self.sampled_images_to_update.dedup();
        self.storage_images_to_update.sort();
        self.storage_images_to_update.dedup();

        let images_to_write = TempList::new();
        let mut writes = Vec::new();
        self.sampled_images_to_update.drain(..).try_for_each(
            |handle| -> Result<(), kiri_backend::Error> {
                if let Some((image, aspect, layout)) = self.images.get(handle) {
                    let view = image.view(ImageViewDesc::new(*aspect))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .image_info(slice::from_ref(
                                images_to_write.add(
                                    vk::DescriptorImageInfo::default()
                                        .image_layout(*layout)
                                        .image_view(view),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .dst_binding(SAMPLED_IMAGES_BINDING as _)
                            .dst_array_element(handle.index() as _)
                            .dst_set(self.descriptor_set),
                    );
                }
                Ok(())
            },
        )?;
        self.storage_images_to_update.drain(..).try_for_each(
            |handle| -> Result<(), kiri_backend::Error> {
                if let Some((image, aspect, layout)) = self.images.get(handle) {
                    let view = image.view(ImageViewDesc::new(*aspect))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .image_info(slice::from_ref(
                                images_to_write.add(
                                    vk::DescriptorImageInfo::default()
                                        .image_view(view)
                                        .image_layout(*layout),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .dst_binding(SAMPLED_IMAGES_BINDING as _)
                            .dst_array_element(handle.index() as _)
                            .dst_set(self.descriptor_set),
                    )
                }
                Ok(())
            },
        )?;

        unsafe { self.device.raw.update_descriptor_sets(&writes, &[]) };

        Ok(())
    }
}
