// Copyright (C) 2025 gigablaster

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

use std::{hash::Hasher, mem, sync::Arc};

use arrayvec::ArrayVec;
use kiri_backend::{
    ash::vk,
    vulkan::{
        BufferPointer, DescriptorHandle, DescriptorSetCreateData, RasterPipelineHandle,
        DYNAMIC_DESCRIPTOR_SLOT_INDEX, MATERIAL_DESCRIPTOR_SLOT_IDNEX, PASS_DESCRIPTOR_SLOT_INDEX,
    },
    DrawStream, DrawStreamBuilder, FrameDispatcher, RenderTarget,
};
use kiri_gfx::{
    material::{INSTANCE_DESCRIPTOR_LAYOUT, SCENE_DESCRIPTOR_LAYOUT},
    RenderTargetPool, TransientImage,
};
use kiri_math::{vec3, vec4, Bounds, Camera, Mat4, PerspectiveCamera, Plane, Vec3, Vec3A};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use seahash::SeaHasher;

use crate::{Culler, Scene};

#[derive(Debug, Clone, Copy)]
pub struct RenderView {
    pub camera: PerspectiveCamera,
    pub expouse: f32,
    pub ambient: (Vec3, Vec3, Vec3),
}

struct FrustumCuller {
    frustum: [Plane; 6],
}

impl Culler for FrustumCuller {
    fn cull(&self, bounds: kiri_math::BoundingBox) -> bool {
        bounds.is_visible(&self.frustum)
    }
}

impl FrustumCuller {
    pub fn new(camera: PerspectiveCamera) -> Self {
        Self {
            frustum: camera.frustum(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct GpuInstanceData {
    pub model: Mat4,
    pub uv_scale: f32,
}

const DEFAULT_RENDER_OP_CAPACITY: usize = 100000;
const DRAWS_PER_STREAM: usize = 256;
const MAX_DIRECTIONAL_LIGHTS: usize = 3;

#[derive(Debug, Clone, Copy)]
#[repr(align(16))]
struct RenderOp {
    pipeline: RasterPipelineHandle,
    ds: DescriptorHandle,
    index_buffer: BufferPointer,
    depth_normalized: u32,
    op_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct RenderOpData {
    model: Mat4,
    vertices: BufferPointer,
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
    pub lights: [DirectionalLightGpuData; MAX_DIRECTIONAL_LIGHTS],
    pub ambient: HemisphericalAmbientGpuData,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct GpuZPassData {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
}

/// Renders scene from given point of view into HDR image
pub fn render_scene<'a>(
    dispatcher: &FrameDispatcher<'a>,
    pool: &'a RenderTargetPool,
    scene: &Scene,
    render_view: RenderView,
) -> Result<TransientImage<'a>, kiri_gfx::Error> {
    puffin::profile_function!();
    let culled = scene.cull(FrustumCuller::new(render_view.camera));
    let mut ops = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let mut zpass_opaque_bin = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let mut zpass_masked_bin = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let mut opaque_bin = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    let mut transparent_bin = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
    {
        puffin::profile_scope!("Prepare render OPs");
        for (mesh, transform) in culled.meshes {
            let decompress_mat = Mat4::from_scale(vec3(
                mesh.position_scale,
                mesh.position_scale,
                mesh.position_scale,
            ));
            for surface in &mesh.surfaces {
                let index = ops.len();
                ops.push(RenderOpData {
                    model: decompress_mat * transform,
                    vertices: mesh.vertex_buffer,
                    index_buffer: mesh.index_buffer,
                    first_index: surface.first_index,
                    index_count: surface.index_count,
                    vertex_offset: surface.vertex_offset,
                    uv_scale: mesh.uv_scale,
                });
                match surface.material.group {
                    kiri_gfx::material::RenderGroup::Opaque => {
                        if surface.material.depth.is_valid() {
                            zpass_opaque_bin.push(RenderOp {
                                pipeline: surface.material.depth,
                                ds: surface.material.descriptor,
                                index_buffer: mesh.index_buffer,
                                depth_normalized: 0,
                                op_index: index,
                            });
                        }
                        if surface.material.main.is_valid() {
                            opaque_bin.push(RenderOp {
                                pipeline: surface.material.main,
                                ds: surface.material.descriptor,
                                index_buffer: mesh.index_buffer,
                                depth_normalized: 0,
                                op_index: index,
                            });
                        }
                    }
                    kiri_gfx::material::RenderGroup::Masked => {
                        if surface.material.depth.is_valid() {
                            zpass_masked_bin.push(RenderOp {
                                pipeline: surface.material.depth,
                                ds: surface.material.descriptor,
                                index_buffer: mesh.index_buffer,
                                depth_normalized: 0,
                                op_index: index,
                            });
                        }
                        if surface.material.main.is_valid() {
                            opaque_bin.push(RenderOp {
                                pipeline: surface.material.main,
                                ds: surface.material.descriptor,
                                index_buffer: mesh.index_buffer,
                                depth_normalized: 0,
                                op_index: index,
                            });
                        }
                    }
                    kiri_gfx::material::RenderGroup::Transparent => {
                        if surface.material.main.is_valid() {
                            transparent_bin.push(RenderOp {
                                pipeline: surface.material.main,
                                ds: surface.material.descriptor,
                                index_buffer: mesh.index_buffer,
                                depth_normalized: 0,
                                op_index: index,
                            })
                        }
                    }
                }
            }
        }
        {
            puffin::profile_scope!("Sort render ops");
            sort_opaque_render_ops(&mut zpass_opaque_bin);
            sort_opaque_render_ops(&mut zpass_masked_bin);
            sort_opaque_render_ops(&mut opaque_bin);
            sort_transparent_ops(&mut transparent_bin);
        }
        {
            puffin::profile_scope!("Submit");
            let view = render_view.camera.view();
            let projection = Mat4::from_cols(
                vec4(1.0, 0.0, 0.0, 0.0),
                vec4(0.0, -1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 0.5, 0.0),
                vec4(0.0, 0.0, 0.5, 1.0),
            ) * render_view.camera.projection();
            let view_projection = projection * view;
            let zpass_ds = dispatcher.temp_descriptor(DescriptorSetCreateData {
                layout: SCENE_DESCRIPTOR_LAYOUT,
                stages: vk::ShaderStageFlags::ALL_GRAPHICS,
                unifoms: &[dispatcher.push_dynamic_data(&[GpuZPassData {
                    view,
                    projection,
                    view_projection,
                }])?],
                ..Default::default()
            })?;
            let eye_position = view.transform_point3a(Vec3A::ZERO);
            let mut directional_lights = culled.directional_lights.clone();
            directional_lights.sort_by(|lhs, rhs| lhs.0.power.total_cmp(&rhs.0.power));
            directional_lights.resize(MAX_DIRECTIONAL_LIGHTS, Default::default());
            let directional_lights_gpu = directional_lights
                .into_iter()
                .map(|(light, transform)| DirectionalLightGpuData {
                    direction: transform.transform_vector3a(Vec3A::X),
                    color: (light.color * light.power).into(),
                })
                .collect::<ArrayVec<_, MAX_DIRECTIONAL_LIGHTS>>();
            let main_pass_ds = dispatcher.temp_descriptor(DescriptorSetCreateData {
                layout: SCENE_DESCRIPTOR_LAYOUT,
                stages: vk::ShaderStageFlags::ALL_GRAPHICS,
                unifoms: &[dispatcher.push_dynamic_data(&[GpuPassData {
                    view,
                    projection,
                    view_projection,
                    eye_position,
                    lights: directional_lights_gpu.into_inner().unwrap(),
                    ambient: HemisphericalAmbientGpuData {
                        top: render_view.ambient.0.into(),
                        middle: render_view.ambient.1.into(),
                        bottom: render_view.ambient.2.into(),
                    },
                }])?],
                ..Default::default()
            })?;
            let zpass_opaque = generate_commands(dispatcher, &ops, &zpass_opaque_bin, zpass_ds)?;
            let zpass_masked = generate_commands(dispatcher, &ops, &zpass_masked_bin, zpass_ds)?;
            let main_opaque = generate_commands(dispatcher, &ops, &opaque_bin, main_pass_ds)?;
            let main_transparent =
                generate_commands(dispatcher, &ops, &transparent_bin, main_pass_ds)?;
            let depth = pool.image(
                vk::Format::D24_UNORM_S8_UINT,
                dispatcher.backbuffer_desc().dims,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            )?;
            let hdr = pool.image(
                vk::Format::R16G16B16A16_SFLOAT,
                dispatcher.backbuffer_desc().dims,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            )?;
            dispatcher.render_pass(
                "zpass",
                kiri_backend::RenderArea::AllTarget,
                &[],
                Some(
                    RenderTarget::new(depth.handle)
                        .clear_depth_stencil(1.0, 0)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                ),
                &[],
                zpass_opaque.into_iter().chain(zpass_masked),
            );
            dispatcher.render_pass(
                "main",
                kiri_backend::RenderArea::AllTarget,
                &[RenderTarget::new(hdr.handle)
                    .clear_color([0.0, 0.0, 0.0, 1.0])
                    .initial_layout(vk::ImageLayout::UNDEFINED)],
                None,
                &[],
                main_opaque.into_iter().chain(main_transparent),
            );
            Ok(hdr)
        }
    }
}

fn generate_commands(
    dispatcher: &FrameDispatcher,
    ops: &[RenderOpData],
    order: &[RenderOp],
    pass_ds: DescriptorHandle,
) -> Result<Vec<DrawStream>, kiri_gfx::Error> {
    let streams = order
        .par_chunks(DRAWS_PER_STREAM)
        .map(|chunk| {
            let mut instances = ArrayVec::<_, DRAWS_PER_STREAM>::new();
            for it in chunk {
                instances.push(GpuInstanceData {
                    model: ops[it.op_index].model,
                    uv_scale: ops[it.op_index].uv_scale,
                });
            }
            let instances_ds = dispatcher
                .temp_descriptor(DescriptorSetCreateData {
                    layout: INSTANCE_DESCRIPTOR_LAYOUT,
                    stages: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    storages: &[dispatcher.push_dynamic_data(&instances).unwrap()],
                    ..Default::default()
                })
                .unwrap();
            let mut stream = DrawStreamBuilder::default();
            chunk
                .iter()
                .enumerate()
                .for_each(|(draw_index, render_op)| {
                    let op = &ops[render_op.op_index];
                    stream.set_pipeline(render_op.pipeline);
                    stream.set_descriptor(PASS_DESCRIPTOR_SLOT_INDEX, Some(pass_ds));
                    stream.set_descriptor(DYNAMIC_DESCRIPTOR_SLOT_INDEX, Some(instances_ds));
                    stream.set_descriptor(MATERIAL_DESCRIPTOR_SLOT_IDNEX, Some(render_op.ds));
                    stream.set_vertex_buffer(0, op.vertices);
                    stream.set_vertex_offset(op.vertex_offset as _);
                    stream.set_index_buffer(op.index_buffer);
                    stream.draw(op.first_index, op.index_count, draw_index as _, 1);
                });
            stream.build()
        })
        .collect::<Vec<_>>();
    Ok(streams)
}

fn sort_opaque_render_ops(ops: &mut [RenderOp]) {
    radsort::sort_by_key(ops, |op| {
        let mut hasher = SeaHasher::default();
        hasher.write_u32(op.pipeline.into());
        hasher.write_u32(op.ds.into());
        hasher.write_u32(op.index_buffer.handle.into());
        hasher.write_u32(op.index_buffer.offset);
        hasher.finish()
    });
}

fn sort_transparent_ops(ops: &mut [RenderOp]) {
    radsort::sort_by_key(ops, |op| (op.depth_normalized) as u64);
}
