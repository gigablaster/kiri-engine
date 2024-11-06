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

use std::mem;

use bevy_ecs::{
    entity::Entity,
    system::{Commands, Query, Res},
};
use kiri_backend::{
    ash::vk, DescriptorSetDesc, DescriptorSetLayoutDesc, InputVertexAttrubute,
    InputVertexStreamLayout, RenderPassLayout, DYNAMIC_BINDING_SLOT, EMPTY_DESCRIPTOR_LAYOUT,
    MATERIAL_BINDING_SLOT, PASS_BINDING_SLOT,
};
use kiri_gfx::{
    effects::{INSTANCE_DESCRIPTOR_LAYOUT, RENDER_PASS_DESCRIPTOR_LAYOUT},
    passes::{
        FinalCompositionPassDispatcher, ImageDependency, RasterizerPassBuilder, RenderTarget,
    },
    BufferPointer, DescriptorHandle, DescriptorSetBuilder, DrawStream, DrawStreamBuilder,
    FrameState, ImageHandle, PipelineCache, PipelineHandle, RasterPipelineDesc, RenderContext,
    RenderMesh, RenderTargetPool, TransientImageGuard,
};
use kiri_math::{Affine3A, Bounds, Camera, Mat4, Vec3, Vec3A};
use kiri_resources::Resource;

use crate::Transform;

use super::{
    DirectionalLight, HemisphericalLight, Model, PendingModel, PerspectiveCamera, Postprocess,
    RenderTargetPoolWrapper, RendererWrapper, ResourceManagerWrapper, SwapchainWrapper,
};

pub fn process_model_loading(
    mut commands: Commands,
    resources: Res<ResourceManagerWrapper>,
    query: Query<(Entity, &PendingModel)>,
) {
    for (entity, pending) in query.iter() {
        if let Some(model) = resources.0.get_model(pending.0) {
            match model {
                Resource::Loading => {}
                Resource::Failed => {
                    commands.entity(entity).remove::<PendingModel>();
                }
                Resource::Loaded(model) => {
                    commands
                        .entity(entity)
                        .remove::<PendingModel>()
                        .insert(Model(model));
                }
            }
        } else {
            commands.entity(entity).remove::<PendingModel>();
        }
    }
}

// Enough for most if not all scenes
const DEFAULT_RENDER_OP_CAPACITY: usize = 65536;

pub fn render_world<'a>(
    mut commands: Commands,
    renderer: Res<RendererWrapper>,
    swpachain: Res<SwapchainWrapper>,
    pool: Res<RenderTargetPoolWrapper>,
    ambient: Res<HemisphericalLight>,
    postprocess: Res<Postprocess>,
    resource_manager: Res<ResourceManagerWrapper>,
    models: Query<(&Transform, &Model)>,
    directional_lights: Query<&DirectionalLight>,
    camera: Query<(&Transform, &PerspectiveCamera)>,
) {
    let mut to_render = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let (camera_transform, camera_props) = camera.single();
    let camera = kiri_math::PerspectiveCamera::new(
        camera_transform.0.translation.into(),
        camera_transform.0.transform_vector3(Vec3::Z),
        camera_transform.0.transform_vector3(Vec3::Y),
        camera_props.fov,
        camera_props.aspect,
        camera_props.znear,
        camera_props.zfar,
    );
    let frustum = camera.frustum();
    let ambient = ambient.into_inner();
    let light = directional_lights.single();
    {
        puffin::profile_scope!("Culling");
        for (transform, model) in models.iter() {
            let bbox = model.0.bounds.transform(transform.0);
            if bbox.is_visible(&frustum) {
                for (node, mesh) in model.0.node_to_mesh.iter().copied() {
                    let transform = transform.0 * model.0.world_transforms[node as usize];
                    let bbox = model.0.bounds_per_mesh[mesh as usize].transform(transform);
                    if bbox.is_visible(&frustum) {
                        to_render.push((transform, &model.0.meshes[mesh as usize]));
                    }
                }
            }
        }
    }

    if FrameState::NeedRecreateSwapchain
        == renderer
            .0
            .render(&swpachain.0, |context| {
                render_frame(
                    to_render,
                    context,
                    &pool.0,
                    &camera,
                    light,
                    ambient,
                    postprocess.into_inner(),
                    &resource_manager.0.pipeline_cache,
                )
                .unwrap()
            })
            .unwrap()
    {
        commands.remove_resource::<SwapchainWrapper>();
    }
}

trait RenderOp {
    fn render_data(&self) -> (PipelineHandle, DescriptorHandle);
}

#[derive(Debug, Clone, Copy)]
struct OpaqueOp {
    pipeline: PipelineHandle,
    ds: DescriptorHandle,
}

impl RenderOp for OpaqueOp {
    fn render_data(&self) -> (PipelineHandle, DescriptorHandle) {
        (self.pipeline, self.ds)
    }
}

#[derive(Debug, Clone, Copy)]
struct TransparentOp {
    pipeline: PipelineHandle,
    ds: DescriptorHandle,
    depth: u64,
}

impl RenderOp for TransparentOp {
    fn render_data(&self) -> (PipelineHandle, DescriptorHandle) {
        (self.pipeline, self.ds)
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderOpData {
    model: Mat4,
    vertex_positions: BufferPointer,
    vertex_attributes: BufferPointer,
    index_buffer: BufferPointer,
    first_index: u32,
    index_count: u32,
    vertex_offset: u32,
    uv_scale: f32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct DirectionalLightGpuData {
    direction: Vec3A,
    color: Vec3A,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct HemisphericalAmbientGpuData {
    pub top: Vec3A,
    pub middle: Vec3A,
    pub bottom: Vec3A,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct GpuPassData {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
    pub eye_position: Vec3A,
    pub light: DirectionalLightGpuData,
    pub ambient: HemisphericalAmbientGpuData,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct GpuInstanceData {
    pub model: Mat4,
    pub uv_scale: f32,
}

const DRAWS_PER_STREAM: usize = 512;

fn render_frame(
    to_render: Vec<(Affine3A, &RenderMesh)>,
    context: &RenderContext,
    pool: &RenderTargetPool,
    camera: &impl Camera,
    light: &DirectionalLight,
    ambient: &HemisphericalLight,
    postprocess: &Postprocess,
    pipeline_cache: &PipelineCache,
) -> Result<(), kiri_gfx::Error> {
    // Prepare data
    // It can be multithreaded with relative ease, do it once it became a bottleneck
    puffin::profile_function!();
    let mut render_ops = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let mut prepass_bin = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let mut opaque_bin = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let mut transparent_bin = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    {
        puffin::profile_scope!("Generate render ops");
        for (transform, mesh) in to_render {
            for surface in &mesh.surfaces {
                let index = render_ops.len();
                render_ops.push(RenderOpData {
                    model: transform.into(),
                    vertex_positions: mesh.positions,
                    vertex_attributes: mesh.attributes,
                    index_buffer: mesh.index_buffer,
                    first_index: surface.first_index,
                    index_count: surface.index_count,
                    vertex_offset: surface.vertex_offset,
                    uv_scale: mesh.uv_scale,
                });
                let depth = surface.material.depth;
                if depth.is_valid() {
                    prepass_bin.push((
                        index,
                        OpaqueOp {
                            pipeline: depth,
                            ds: surface.material.instance.ds,
                        },
                    ));
                }
                let opaque = surface.material.main;
                if opaque.is_valid() {
                    opaque_bin.push((
                        index,
                        OpaqueOp {
                            pipeline: opaque,
                            ds: surface.material.instance.ds,
                        },
                    ));
                }
                let transparent = surface.material.transparent;
                if transparent.is_valid() {
                    // Fixme: depth
                    transparent_bin.push((
                        index,
                        TransparentOp {
                            pipeline: transparent,
                            ds: surface.material.instance.ds,
                            depth: 0,
                        },
                    ));
                }
            }
        }
    }
    // Sort
    {
        puffin::profile_scope!("Sort render ops");
        radsort::sort_by_cached_key(&mut prepass_bin, |op| {
            (op.1.pipeline.index() as u64) << 32 | op.1.ds.index() as u64
        });
        radsort::sort_by_cached_key(&mut opaque_bin, |op| {
            (op.1.pipeline.index() as u64) << 32 | op.1.ds.index() as u64
        });
        radsort::sort_by_cached_key(&mut transparent_bin, |op| op.1.depth);
    }
    // Execute rendering
    let color = pool.get_image(
        vk::Format::R16G16B16A16_SFLOAT,
        context.backbuffer.desc.dims,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
    )?;
    {
        puffin::profile_scope!("Generate and submit command buffers");
        let depth = pool.get_image(
            vk::Format::X8_D24_UNORM_PACK32,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        let mut depth_pass = RasterizerPassBuilder::new(
            "Depth",
            &[],
            Some(
                RenderTarget::new(depth.handle)
                    .clear_depth_stencil(1.0, 0)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
            ),
        );
        let view = camera.view();
        let projection = camera.projection();
        let view_projection = projection * view;
        let eye_position = view.transform_point3a(Vec3A::ZERO);
        let pass_data = context.push_dynamic_data(&[GpuPassData {
            view,
            projection,
            view_projection,
            eye_position,
            light: DirectionalLightGpuData {
                direction: light.direction.into(),
                color: light.color.into(),
            },
            ambient: HemisphericalAmbientGpuData {
                top: ambient.top.into(),
                middle: ambient.middle.into(),
                bottom: ambient.bottom.into(),
            },
        }])?;
        let pass_ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                RENDER_PASS_DESCRIPTOR_LAYOUT,
            )
            .bind_uniform_buffer(0, pass_data),
        )?;
        generate_commands(context, &mut depth_pass, &render_ops, &prepass_bin, pass_ds)?;
        let mut main_pass = RasterizerPassBuilder::new(
            "Main",
            &[RenderTarget::new(color.handle)
                .clear_color([0.0, 0.0, 0.0, 1.0])
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
            Some(RenderTarget::new(depth.handle).load().discard()),
        );
        generate_commands(context, &mut main_pass, &render_ops, &opaque_bin, pass_ds)?;
        generate_commands(
            context,
            &mut main_pass,
            &render_ops,
            &transparent_bin,
            pass_ds,
        )?;
        context.submit(depth_pass.build());
        context.submit(main_pass.build());
    }
    let post = tonemapping(color.handle, &postprocess, context, &pool, &pipeline_cache)?;

    context.submit(Box::new(FinalCompositionPassDispatcher::new(post.handle)));

    Ok(())
}

fn generate_commands<T: RenderOp + Copy>(
    context: &RenderContext,
    pass: &mut RasterizerPassBuilder,
    ops: &[RenderOpData],
    order: &[(usize, T)],
    pass_ds: DescriptorHandle,
) -> Result<(), kiri_gfx::Error> {
    puffin::profile_function!();
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
    let mut index = 0;
    while index < order.len() {
        let mut instance = 0;
        let mut writer = context.write_dynamic_data::<GpuInstanceData>(DRAWS_PER_STREAM)?;
        let mut stream = DrawStreamBuilder::default();
        while instance < DRAWS_PER_STREAM && index < order.len() {
            let (op_index, per_instance) = order[index];
            let op = &ops[op_index];
            writer.write(GpuInstanceData {
                model: op.model,
                uv_scale: op.uv_scale,
            })?;
            let (pipeline, material_ds) = per_instance.render_data();
            stream.set_pipeline(pipeline);
            stream.set_descriptor(PASS_BINDING_SLOT, Some(pass_ds));
            stream.set_descriptor(MATERIAL_BINDING_SLOT, Some(material_ds));
            stream.set_descriptor(DYNAMIC_BINDING_SLOT, Some(instance_ds));
            stream.set_dynamic_offset(0, Some(writer.offset as _));
            stream.set_vertex_buffer(0, op.vertex_positions);
            stream.set_vertex_buffer(1, op.vertex_attributes);
            stream.set_vertex_offset(op.vertex_offset as _);
            stream.set_index_buffer(op.index_buffer);
            stream.draw(op.first_index, op.index_count, instance as _, 1);
            index += 1;
            instance += 1;
        }
        pass.draw(stream.build());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct TonemappingGpuData {
    pub expouse: f32,
}

fn tonemapping<'a>(
    color_target: ImageHandle,
    env: &Postprocess,
    context: &RenderContext,
    pool: &'a RenderTargetPool,
    pipeline_cache: &PipelineCache,
) -> Result<TransientImageGuard<'a>, kiri_gfx::Error> {
    let post = pool.get_image(
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
            context.push_dynamic_data(&[TonemappingGpuData {
                expouse: env.expouse,
            }])?,
        ),
    )?;
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
    pass.draw(postprocess(context, tonemapping, ds)?);

    context.submit(pass.build());
    Ok(post)
}

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

fn postprocess(
    context: &RenderContext,
    pipeline: PipelineHandle,
    ds: DescriptorHandle,
) -> Result<DrawStream, kiri_gfx::Error> {
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
