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

use crate::{
    gpu::{GpuInstanceData, RenderPassGpuData},
    Error,
};
use kiri_assets::STATIC_MESH_INPUT_LAYOUT;
use kiri_backend::{
    ash::vk, DescriptorSetDesc, DescriptorSetLayoutDesc, InputVertexAttrubute,
    InputVertexStreamLayout, RenderPassLayout, DYNAMIC_BINDING_SLOT, EMPTY_DESCRIPTOR_LAYOUT,
    MATERIAL_BINDING_SLOT, PASS_BINDING_SLOT,
};
use kiri_gfx::{
    passes::{
        FinalCompositionPassDispatcher, ImageDependency, RasterizerPassBuilder, RenderTarget,
    },
    BufferPointer, DescriptorHandle, DescriptorSetBuilder, DrawStream, DrawStreamBuilder,
    ImageHandle, PipelineHandle, RenderContext, RenderTargetPool, TransientImageGuard,
};
use kiri_math::{vec3, vec4, BoundingBox, Bounds, Camera, Mat4, Plane, Vec3, Vec3A};
use kiri_resources::{
    PipelineCache, RasterPipelineDesc, ResourceManager, MATERIAL_DESCRIPTOR_LAYOUT,
};
use kiri_scene::{Scene, SceneCuller};

const RENDER_PASS_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[(
        0,
        DescriptorSetDesc {
            name: "per_pass",
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            count: 1,
        },
    )],
    update_after_bind: false,
};

const INSTANCE_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[(
        0,
        DescriptorSetDesc {
            name: "instance",
            ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            count: 1,
        },
    )],
    update_after_bind: false,
};

const POSTPROCESS_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[
        (
            0,
            DescriptorSetDesc {
                name: "main",
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ),
        (
            1,
            DescriptorSetDesc {
                name: "params",
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
            },
        ),
    ],
    update_after_bind: false,
};

const PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[vk::Format::R16G16B16A16_SFLOAT],
    depth: Some(vk::Format::D24_UNORM_S8_UINT),
};

const POSTPROCESS_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[vk::Format::A2R10G10B10_UNORM_PACK32],
    depth: None,
};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PostprocessVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

const POSTPROCESS_INPUT_LAYOUT: [InputVertexStreamLayout; 1] = [InputVertexStreamLayout {
    streams: &[
        InputVertexAttrubute {
            location: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0,
        },
        InputVertexAttrubute {
            location: 1,
            format: vk::Format::R32G32_SFLOAT,
            offset: 8,
        },
    ],
    stride: 16,
}];

#[derive(Debug)]
pub struct SceneRenderer {
    target_pool: RenderTargetPool,
    resources: Arc<ResourceManager>,
    _pipelines: Arc<PipelineCache>,
    main_material: PipelineHandle,
    tonemapping: PipelineHandle,
}

struct FrustrumCuller {
    planes: [Plane; 6],
}

impl FrustrumCuller {
    pub fn new(planes: [Plane; 6]) -> Self {
        Self { planes }
    }
}

impl SceneCuller for FrustrumCuller {
    fn visible(&self, bounds: BoundingBox) -> bool {
        bounds.is_visible(&self.planes)
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuCamera {
    pub view: Mat4,
    pub projection: Mat4,
}

const DRAWS_PER_STREAM: usize = 256;

#[derive(Debug, Clone, Copy)]
struct RenderOp {
    pipeline: PipelineHandle,
    material: DescriptorHandle,
    index: usize,
}
#[derive(Debug, Clone, Copy)]
struct RenderOpData {
    model: Mat4,
    uv_scale: f32,
    vertex_positions: BufferPointer,
    vertex_attributes: BufferPointer,
    index_buffer: BufferPointer,
    first_index: u32,
    index_count: u32,
    vertex_offset: u32,
}

impl PartialEq for RenderOp {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
            && self.material == other.material
            && self.index == other.index
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
    pub direction: Vec3A,
    pub color: Vec3A,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]

pub struct HemisphericalAmbient {
    pub top: Vec3A,
    pub middle: Vec3A,
    pub bottom: Vec3A,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderEnviroment {
    pub lights: [DirectionalLight; 3],
    pub ambient: HemisphericalAmbient,
    pub expouse: f32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct TonemappingParams {
    pub expouse: f32,
}

impl SceneRenderer {
    pub fn new(
        resource_cache: &Arc<ResourceManager>,
        pipeline_cache: &Arc<PipelineCache>,
    ) -> Result<Self, Error> {
        let renderer: &Arc<kiri_gfx::Renderer> = &resource_cache.renderer;
        let main_material =
            pipeline_cache.get_or_create_raster_pipeline(RasterPipelineDesc::new(
                "shaders/main.vert",
                "shaders/main.frag",
                &PASS_LAYOUT,
                &STATIC_MESH_INPUT_LAYOUT,
                &[
                    RENDER_PASS_DESCRIPTOR_LAYOUT,
                    EMPTY_DESCRIPTOR_LAYOUT,
                    MATERIAL_DESCRIPTOR_LAYOUT,
                    INSTANCE_DESCRIPTOR_LAYOUT,
                ],
            ))?;
        let tonemapping = pipeline_cache.get_or_create_raster_pipeline(RasterPipelineDesc::new(
            "shaders/fullscreen.vert",
            "shaders/tonemapping.frag",
            &POSTPROCESS_PASS_LAYOUT,
            &POSTPROCESS_INPUT_LAYOUT,
            &[
                POSTPROCESS_DESCRIPTOR_LAYOUT,
                EMPTY_DESCRIPTOR_LAYOUT,
                EMPTY_DESCRIPTOR_LAYOUT,
                EMPTY_DESCRIPTOR_LAYOUT,
            ],
        ))?;
        Ok(Self {
            target_pool: RenderTargetPool::new(renderer),
            resources: resource_cache.clone(),
            _pipelines: pipeline_cache.clone(),
            main_material,
            tonemapping,
        })
    }

    pub fn render(
        &self,
        scene: &Scene,
        camera: impl Camera,
        env: RenderEnviroment,
        context: &RenderContext,
    ) -> Result<(), Error> {
        puffin::profile_function!();
        let color_target = self.render_geometry(scene, camera, &env, context)?;
        let post = self.tonemapping(color_target.handle, &env, context)?;

        context.submit(Box::new(FinalCompositionPassDispatcher::new(post.handle)));

        Ok(())
    }

    fn tonemapping(
        &self,
        color_target: ImageHandle,
        env: &RenderEnviroment,
        context: &RenderContext,
    ) -> Result<TransientImageGuard, Error> {
        let post = self.target_pool.get_image(
            vk::Format::A2R10G10B10_UNORM_PACK32,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let mut pass = RasterizerPassBuilder::new(
            "Tonemapping",
            &[RenderTarget::new(post.handle).initial_layout(vk::ImageLayout::UNDEFINED)],
            None,
        )
        .read_image(ImageDependency::color(color_target));

        let ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                POSTPROCESS_DESCRIPTOR_LAYOUT,
            )
            .bind_image(0, color_target, vk::ImageAspectFlags::COLOR)
            .bind_uniform_buffer(
                1,
                context.push_dynamic_data(&[TonemappingParams {
                    expouse: env.expouse,
                }])?,
            ),
        )?;
        pass.draw(self.postprocess(context, self.tonemapping, ds)?);

        context.submit(pass.build());
        Ok(post)
    }

    fn render_geometry(
        &self,
        scene: &Scene,
        camera: impl Camera,
        env: &RenderEnviroment,
        context: &RenderContext,
    ) -> Result<TransientImageGuard, Error> {
        let resolver = self.resources.resolve();
        let color_target = self.target_pool.get_image(
            vk::Format::R16G16B16A16_SFLOAT,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?;
        let view = camera.view();
        let projection = camera.projection();
        let depth_target = self.target_pool.get_image(
            vk::Format::D24_UNORM_S8_UINT,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        )?;
        let projection = projection
            * Mat4::from_cols(
                vec4(1.0, 0.0, 0.0, 0.0),
                vec4(0.0, -1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 0.5, 0.0),
                vec4(0.0, 0.0, 0.5, 1.0),
            );
        let pass_data = context.push_dynamic_data(&[RenderPassGpuData {
            view,
            projection,
            view_projection: projection * view,
            eye_position: view.transform_point3(Vec3::default()),
            lights: env.lights,
            ambient: env.ambient,
        }])?;
        let pass_ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                RENDER_PASS_DESCRIPTOR_LAYOUT,
            )
            .bind_uniform_buffer(0, pass_data),
        )?;
        let instance_ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                INSTANCE_DESCRIPTOR_LAYOUT,
            )
            .bind_dynamic_storage_buffer(
                0,
                context.get_temprary_buffer(),
                (mem::size_of::<GpuInstanceData>() * DRAWS_PER_STREAM) as _,
            ),
        )?;
        let mut pass = RasterizerPassBuilder::new(
            "Main pass",
            &[RenderTarget::new(color_target.handle)
                .clear_color([0.0, 0.0, 0.0, 1.0])
                .initial_layout(vk::ImageLayout::UNDEFINED)],
            Some(
                RenderTarget::new(depth_target.handle)
                    .clear_depth_stencil(1.0, 0)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .discard(),
            ),
        );
        let mut render_data = Vec::with_capacity(64536);
        let mut render_ops = Vec::with_capacity(64536);
        let frustum = camera.frustum();
        {
            puffin::profile_scope!("Generate renderops");
            for (tr, mesh) in scene.cull(&FrustrumCuller::new(frustum), &resolver) {
                let decompress_mat = Mat4::from_scale(vec3(
                    mesh.position_scale,
                    mesh.position_scale,
                    mesh.position_scale,
                ));
                let model: Mat4 = tr.into();
                for surface in &mesh.surfaces {
                    let index = render_data.len();
                    render_data.push(RenderOpData {
                        model: model * decompress_mat,
                        uv_scale: mesh.uv_scale,
                        vertex_positions: mesh.vertex_positions,
                        vertex_attributes: mesh.vertex_attributes,
                        index_buffer: mesh.index_buffer,
                        first_index: surface.first_index,
                        index_count: surface.index_count,
                        vertex_offset: surface.vertex_offset,
                    });
                    render_ops.push(RenderOp {
                        pipeline: self.main_material,
                        material: surface.material.ds,
                        index,
                    });
                    debug_assert!(render_data.len() == render_ops.len(), "Sanity check failed");
                }
            }
        }
        {
            puffin::profile_scope!("Sorting renderops");
            radsort::sort_by_key(&mut render_ops, |x| {
                (Into::<u64>::into(x.pipeline), Into::<u64>::into(x.material))
            });
        }
        {
            puffin::profile_scope!("Generate draw streams");
            let mut index = 0;
            while index < render_data.len() {
                let mut instance = 0;
                let mut data = context.write_dynamic_data::<GpuInstanceData>(DRAWS_PER_STREAM)?;
                let mut stream = DrawStreamBuilder::default();

                while instance < DRAWS_PER_STREAM && index < render_data.len() {
                    let op = render_ops[index];
                    let op_data = &render_data[op.index];
                    data.write(GpuInstanceData {
                        model: op_data.model,
                        uv_scale: op_data.uv_scale,
                    })?;
                    stream.set_pipeline(op.pipeline);
                    stream.set_descriptor(PASS_BINDING_SLOT, Some(pass_ds));
                    stream.set_descriptor(DYNAMIC_BINDING_SLOT, Some(instance_ds));
                    stream.set_vertex_buffer(0, op_data.vertex_positions);
                    stream.set_vertex_buffer(1, op_data.vertex_attributes);
                    stream.set_index_buffer(op_data.index_buffer);
                    stream.set_vertex_offset(op_data.vertex_offset as _);
                    stream.set_descriptor(MATERIAL_BINDING_SLOT, Some(op.material));
                    stream.set_dynamic_offset(0, Some(data.offset as _));
                    stream.draw(op_data.first_index, op_data.index_count, instance as _, 1);
                    instance += 1;
                    index += 1;
                }
                pass.draw(stream.build());
            }
            context.submit(pass.build());
        }
        Ok(color_target)
    }

    pub fn swapchain_changed(&self) {
        self.target_pool.purge();
    }

    fn postprocess(
        &self,
        context: &RenderContext,
        pipeline: PipelineHandle,
        ds: DescriptorHandle,
    ) -> Result<DrawStream, Error> {
        let mut stream = DrawStreamBuilder::default();
        let vb = context.push_dynamic_data(&[
            PostprocessVertex {
                position: [-1.0, -1.0],
                uv: [0.0, 0.0],
            },
            PostprocessVertex {
                position: [1.0, -1.0],
                uv: [1.0, 0.0],
            },
            PostprocessVertex {
                position: [1.0, 1.0],
                uv: [1.0, 1.0],
            },
            PostprocessVertex {
                position: [-1.0, 1.0],
                uv: [0.0, 1.0],
            },
        ])?;
        let ib = context.push_dynamic_data(&[2u16, 1u16, 0u16, 3u16, 2u16, 0u16])?;
        stream.set_pipeline(pipeline);
        stream.set_descriptor(0, Some(ds));
        stream.set_vertex_buffer(0, vb.into());
        stream.set_index_buffer(ib.into());
        stream.draw(0, 6, 0, 1);
        Ok(stream.build())
    }
}
