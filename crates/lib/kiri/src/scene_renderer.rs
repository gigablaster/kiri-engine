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

use std::{cmp::Ordering, mem, sync::Arc};

use ash::vk::{self};
use kiri_backend::{
    DescriptorSetLayoutDesc, ImageAttachmentDesc, RenderPassLayout, SubpassLayout,
    DYNAMIC_BINDING_SLOT, MATERIAL_BINDING_SLOT, PASS_BINDING_SLOT,
};
use kiri_gfx::{
    BindingSlot, BufferPointer, DescriptorHandle, DescriptorSetBuilder, DrawStreamBuilder,
    ImageHandle, PipelineHandle, RenderContext, RenderPassHandle, RenderTarget, RenderTargetPool,
};
use lazy_static::lazy_static;

use crate::{
    gpu::{GpuInstanceData, GpuStaticVertex, RenderPassGpuData},
    Error, PipelineCache, RasterPipelineDesc, ResourceCache, Scene, SceneCuller,
};

lazy_static! {
    static ref RENDER_PASS_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc =
        DescriptorSetLayoutDesc::default().slot(0, "pass", vk::DescriptorType::UNIFORM_BUFFER, 1);
    static ref INSTANCE_DATA_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc =
        DescriptorSetLayoutDesc::default().slot(
            0,
            "instance",
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            1
        );
}

#[derive(Debug)]
pub struct SceneRenderer {
    target_pool: RenderTargetPool,
    resources: Arc<ResourceCache>,
    pipelines: Arc<PipelineCache>,
    main_pass: RenderPassHandle,
}

struct NullCuller {}

impl SceneCuller for NullCuller {
    fn cull(&self, _bounds: crate::Bounds) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Camera {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
}

const DRAWS_PER_STREAM: usize = 256;

#[derive(Debug, Clone, Copy)]
struct RenderOp {
    pipeline: PipelineHandle,
    model: glam::Mat4,
    vertex_buffer: BufferPointer,
    index_buffer: BufferPointer,
    material: DescriptorHandle,
    first_index: u32,
    index_count: u32,
}

impl PartialEq for RenderOp {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
            && self.model == other.model
            && self.vertex_buffer == other.vertex_buffer
            && self.index_buffer == other.index_buffer
            && self.material == other.material
            && self.first_index == other.first_index
            && self.index_count == other.index_count
    }
}

impl Eq for RenderOp {}

impl PartialOrd for RenderOp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RenderOp {
    fn cmp(&self, other: &Self) -> Ordering {
        let pipeline = self.pipeline.cmp(&other.pipeline);
        let material = self.material.cmp(&other.material);
        if pipeline == Ordering::Equal {
            material
        } else {
            pipeline
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]

pub struct DirectionalLight {
    pub direction: glam::Vec3A,
    pub color: glam::Vec3A,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]

pub struct HemisphericalAmbient {
    pub top: glam::Vec3A,
    pub middle: glam::Vec3A,
    pub bottom: glam::Vec3A,
}

pub struct RenderEnviroment {
    pub camera: Camera,
    pub lights: [DirectionalLight; 3],
    pub ambient: HemisphericalAmbient,
}

impl SceneRenderer {
    pub fn new(
        resource_cache: &Arc<ResourceCache>,
        pipeline_cache: &Arc<PipelineCache>,
    ) -> Result<Self, Error> {
        let renderer = &resource_cache.renderer;
        let layout = RenderPassLayout {
            color: &[ImageAttachmentDesc::new(vk::Format::A2R10G10B10_UNORM_PACK32).clear_input()],
            depth: Some(
                ImageAttachmentDesc::new(vk::Format::D24_UNORM_S8_UINT)
                    .clear_input()
                    .discard(),
            ),
            subpasses: &[SubpassLayout {
                depth_write: true,
                depth_read: false,
                color_writes: &[0],
                color_reads: &[],
            }],
        };
        Ok(Self {
            target_pool: RenderTargetPool::new(renderer),
            resources: resource_cache.clone(),
            pipelines: pipeline_cache.clone(),
            main_pass: renderer.create_render_pass(layout)?,
        })
    }

    pub fn render(
        &self,
        scene: &Scene,
        env: RenderEnviroment,
        context: &RenderContext,
    ) -> Result<ImageHandle, Error> {
        puffin::profile_function!();
        self.target_pool.recycle();
        let resolver = self.resources.resolve();
        let color_target = self.target_pool.get_image(
            vk::Format::A2R10G10B10_UNORM_PACK32,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let depth_target = self.target_pool.get_transient_image(
            vk::Format::D24_UNORM_S8_UINT,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        )?;
        let pass_data = context.push_dynamic_data(&[RenderPassGpuData {
            view: env.camera.view,
            projection: env.camera.projection,
            view_projection: env.camera.projection * env.camera.view,
            eye_position: env.camera.view.transform_point3(glam::Vec3::default()),
            lights: env.lights,
            ambient: env.ambient,
        }])?;
        let pass_ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                &RENDER_PASS_DESCRIPTOR_LAYOUT,
            )
            .bind_uniform_buffer(BindingSlot::Index(0), pass_data)?,
        )?;
        let instance_ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                &INSTANCE_DATA_DESCRIPTOR_LAYOUT,
            )
            .bind_dynamic_storage_buffer(
                BindingSlot::Index(0),
                context.get_temprary_buffer(),
                (mem::size_of::<GpuInstanceData>() * DRAWS_PER_STREAM) as _,
            )?,
        )?;
        let visible = scene.cull(NullCuller {}, &resolver);
        let mut pass = context.create_rasterizer_pass(
            "Main pass",
            self.main_pass,
            &[RenderTarget::new(color_target).clear_color([0.0, 0.0, 0.0, 1.0])],
            Some(RenderTarget::new(depth_target.handle).clear_depth_stencil(1.0, 0)),
            None,
        );
        let pipeline = self
            .pipelines
            .get_or_create_raster_pipeline(RasterPipelineDesc::new::<GpuStaticVertex>(
                "shaders/main.vert",
                "shaders/main.frag",
                self.main_pass,
                0,
            ))?;
        let mut render_ops = Vec::new();
        for (model, mesh) in &visible.static_meshes {
            for surface in &mesh.surfaces {
                render_ops.push(RenderOp {
                    pipeline,
                    model: (*model).into(),
                    vertex_buffer: mesh.vertex_buffer,
                    index_buffer: mesh.index_buffer,
                    material: surface.material.ds,
                    first_index: surface.first_index,
                    index_count: surface.index_count,
                })
            }
        }
        render_ops.sort();

        let mut index = 0;
        while index < render_ops.len() {
            let mut instance = 0;
            let mut data = context.write_dynamic_data::<GpuInstanceData>(DRAWS_PER_STREAM)?;
            let mut stream = DrawStreamBuilder::new(0);

            while instance < DRAWS_PER_STREAM && index < render_ops.len() {
                let op = &render_ops[index];
                // debug!("{:?}", op.model.to_scale_rotation_translation());
                data.write(GpuInstanceData { model: op.model })?;
                stream.set_pipeline(op.pipeline);
                stream.set_descriptor(PASS_BINDING_SLOT, Some(pass_ds));
                stream.set_descriptor(DYNAMIC_BINDING_SLOT, Some(instance_ds));
                stream.set_vertex_buffer(0, Some(op.vertex_buffer));
                stream.set_index_buffer(op.index_buffer);
                stream.set_descriptor(MATERIAL_BINDING_SLOT, Some(op.material));
                stream.set_dynamic_offset(0, Some(data.offset as _));
                stream.draw(op.first_index, op.index_count, instance as _, 1);
                instance += 1;
                index += 1;
            }
            pass.draw(stream.build());
        }
        self.target_pool.insert_barriers(context);
        context.submit(pass.build());
        Ok(color_target)
    }

    pub fn swapchain_changed(&self) {
        self.target_pool.purge();
    }
}
