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

use std::{collections::HashMap, mem, sync::Arc};

use crate::{ImageHandle, PassDispatcher, RenderContext, Renderer};
use ash::vk::{self, ImageUsageFlags};
use kiri_backend::ImageCreateDesc;
use log::debug;
use parking_lot::Mutex;

use crate::Error;

pub trait ResolutionScale {
    fn scale_down(&self, scale: u32) -> [u32; 2];
}

impl ResolutionScale for [u32; 2] {
    fn scale_down(&self, scale: u32) -> [u32; 2] {
        [self[0] / scale, self[1] / scale]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TempImageKey {
    pub dims: [u32; 2],
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Debug)]
pub struct RenderTargetPool {
    renderer: Arc<Renderer>,
    images: Mutex<HashMap<TempImageKey, Vec<ImageHandle>>>,
    images_in_use: Mutex<Vec<(ImageHandle, ImageUsageFlags)>>,
}

#[derive(Debug)]
pub struct RenderTargetGuard<'a> {
    pool: &'a RenderTargetPool,
    key: TempImageKey,
    pub handle: ImageHandle,
}

impl RenderTargetPool {
    pub fn new(renderer: &Arc<Renderer>) -> Self {
        Self {
            renderer: renderer.clone(),
            images: Default::default(),
            images_in_use: Default::default(),
        }
    }

    pub fn get(
        &self,
        format: vk::Format,
        dims: [u32; 2],
        usage: vk::ImageUsageFlags,
    ) -> Result<RenderTargetGuard, Error> {
        assert!(
            usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                || usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            "Must be an attachment"
        );
        let mut images = self.images.lock();
        let key = TempImageKey {
            dims,
            format,
            usage,
        };
        let group = images.entry(key).or_default();
        if let Some(image) = group.pop() {
            self.mark_as_used(image, usage);
            Ok(RenderTargetGuard {
                pool: self,
                key,
                handle: image,
            })
        } else {
            debug!(
                "Create render taget resolution: {:?} format: {:?} usage: {:?}",
                dims, format, usage,
            );
            let image = self.renderer.create_image(
                ImageCreateDesc::new(format, dims)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(usage),
                None,
            )?;
            self.mark_as_used(image, usage);
            Ok(RenderTargetGuard {
                pool: self,
                key,
                handle: image,
            })
        }
    }

    fn mark_as_used(&self, image: ImageHandle, usage: ImageUsageFlags) {
        self.images_in_use.lock().push((image, usage));
    }

    pub fn purge(&self) {
        let mut images = self.images.lock();
        images.drain().for_each(|(_, mut group)| {
            group.drain(..).for_each(|x| self.renderer.destroy_image(x))
        });
    }

    /// Inserts necessary barriers
    ///
    /// Should be called before any pass that used allocated render targets.
    pub fn insert_barriers(&self, context: &RenderContext) {
        let images: Vec<_> = mem::take(&mut self.images_in_use.lock());
        let color = images
            .iter()
            .copied()
            .filter_map(|(image, usage)| {
                if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
                    Some(image)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let depth = images
            .iter()
            .copied()
            .filter_map(|(image, usage)| {
                if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
                    Some(image)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        context.submit(Box::new(TempRenderTargetsBarrierDispatcher {
            color,
            depth,
        }))
    }

    fn recycle(&self, image: ImageHandle, key: TempImageKey) {
        let mut images = self.images.lock();
        let group = images.entry(key).or_default();
        group.push(image);
    }
}

impl<'a> Drop for RenderTargetGuard<'a> {
    fn drop(&mut self) {
        self.pool.recycle(self.handle, self.key);
    }
}

struct TempRenderTargetsBarrierDispatcher {
    color: Vec<ImageHandle>,
    depth: Vec<ImageHandle>,
}

impl PassDispatcher for TempRenderTargetsBarrierDispatcher {
    fn name(&self) -> &str {
        "Barriers for temporary render targets"
    }

    fn dispatch(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        resolver: &crate::RenderResourceResolver,
    ) -> Result<(), Error> {
        let mut color_barriers = Vec::with_capacity(self.color.len());
        for image in self.color.iter().copied() {
            if let Ok(image) = resolver.resolve_image(image) {
                color_barriers.push(
                    vk::ImageMemoryBarrier::default()
                        .image(image.raw)
                        .src_access_mask(vk::AccessFlags::SHADER_READ)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR, // fixme
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            }
        }
        let mut depth_barriers = Vec::with_capacity(self.depth.len());
        for image in self.depth.iter().copied() {
            if let Ok(image) = resolver.resolve_image(image) {
                depth_barriers.push(
                    vk::ImageMemoryBarrier::default()
                        .image(image.raw)
                        .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                        .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH
                                | vk::ImageAspectFlags::STENCIL,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }),
                )
            }
        }
        unsafe {
            if !color_barriers.is_empty() {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &[],
                    &color_barriers,
                )
            }
            if !depth_barriers.is_empty() {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &[],
                    &depth_barriers,
                )
            }
        }
        Ok(())
    }
}
