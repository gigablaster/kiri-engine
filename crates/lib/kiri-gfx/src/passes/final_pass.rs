use std::sync::Arc;

use kiri_backend::{
    ash::{self, vk},
    Image,
};

use crate::PassDispatcher;

/// Copy result image to backbuffer and prepare it for presentation
pub struct FinalCompositionPassDispatcher {
    image: Arc<Image>,
}

impl FinalCompositionPassDispatcher {
    pub fn new(image: Arc<Image>) -> Self {
        Self { image }
    }
}

impl PassDispatcher for FinalCompositionPassDispatcher {
    fn dispatch(
        &self,
        device: &ash::Device,
        command_buffer: ash::vk::CommandBuffer,
        resolver: &crate::RenderResourceResolver,
    ) -> Result<(), crate::Error> {
        unsafe {
            let barriers = [
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(resolver.backbuffer.raw)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(self.image.raw)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            ];
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &barriers,
            );
            device.cmd_blit_image(
                command_buffer,
                self.image.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                resolver.backbuffer.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D::default()
                            .x(self.image.desc.dims[0] as _)
                            .y(self.image.desc.dims[1] as _)
                            .z(1),
                    ])
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D::default()
                            .x(resolver.backbuffer.desc.dims[0] as _)
                            .y(resolver.backbuffer.desc.dims[1] as _)
                            .z(1),
                    ])
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
                vk::Filter::LINEAR,
            );
            let barriers = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE) // ?
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(resolver.backbuffer.raw)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &barriers,
            );
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "Copy target to backbuffer"
    }
}
