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

use std::{mem, sync::Arc};

use crossbeam::queue::SegQueue;
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
    PipelineHandle, RasterPipelineDesc, RenderContext, RenderModel, RenderTargetPool,
    TransientImage,
};
use kiri_math::{vec4, Affine3A, Bounds, Camera, Mat4, PerspectiveCamera, Plane, Vec3, Vec3A};
use kiri_resources::{ModelHandle, ResourceManager};
use log::warn;
use parking_lot::Mutex;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
const DRAWS_PER_STREAM: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderWorldHandle {
    index: u32,
    generation: u32,
}

#[derive(Debug, Default)]
enum WorldObject {
    #[default]
    Empty,
    PendingModel(ModelHandle),
    Model(Arc<RenderModel>),
    DirectionalLight(DirectionalLight),
    PointLight(PointLight),
}

#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    pub color: Vec3,
    pub power: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub color: Vec3,
    pub power: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderView {
    pub camera: PerspectiveCamera,
    pub expouse: f32,
    pub ambient: (Vec3, Vec3, Vec3),
}

#[derive(Debug)]
enum Operation {
    Remove(RenderWorldHandle),
    UpdateTransform(RenderWorldHandle, Affine3A),
    LoadModel(RenderWorldHandle, ModelHandle),
    SetModel(RenderWorldHandle, Arc<RenderModel>),
    UpdaetLightColor(RenderWorldHandle, Vec3),
    UpdateLightPower(RenderWorldHandle, f32),
    UpdateLightRadius(RenderWorldHandle, f32),
}

#[derive(Debug)]
pub struct RenderWorld {
    resources: Arc<ResourceManager>,
    inner: Mutex<RenderWorldInner>,
    operations: SegQueue<Operation>,
    tonemapping: PipelineHandle,
}

pub struct WorldSpawnContext<'a> {
    world: &'a mut RenderWorldInner,
}

impl<'a> WorldSpawnContext<'a> {
    pub fn model(&mut self, transform: &Affine3A, model: &Arc<RenderModel>) -> RenderWorldHandle {
        self.world
            .spawn(transform, WorldObject::Model(model.clone()))
    }

    pub fn load_model(&mut self, transform: &Affine3A, model: ModelHandle) -> RenderWorldHandle {
        self.world
            .spawn(transform, WorldObject::PendingModel(model))
    }

    pub fn directional_light(
        &mut self,
        transform: &Affine3A,
        light: DirectionalLight,
    ) -> RenderWorldHandle {
        self.world
            .spawn(transform, WorldObject::DirectionalLight(light))
    }

    pub fn point_light(&mut self, transform: &Affine3A, light: PointLight) -> RenderWorldHandle {
        self.world.spawn(transform, WorldObject::PointLight(light))
    }
}

const DEFAULT_RENDER_OP_CAPACITY: usize = 100000;

#[derive(Debug, Clone, Copy)]
struct RenderOp {
    pipeline: PipelineHandle,
    ds: DescriptorHandle,
    op_index: usize,
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
struct GpuZPassData {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct TonemappingGpuData {
    pub expouse: f32,
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

struct CullData {
    ops: Vec<RenderOpData>,
    depth: Vec<RenderOp>,
    opaque: Vec<RenderOp>,
    transparent: Vec<RenderOp>,
    directional_lights: Vec<(Affine3A, DirectionalLight)>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct GpuInstanceData {
    pub model: Mat4,
    pub uv_scale: f32,
}

impl RenderWorld {
    pub fn new(resource_manager: &Arc<ResourceManager>) -> Result<Self, kiri_gfx::Error> {
        let tonemapping = resource_manager
            .pipeline_cache
            .get_or_create_raster_pipeline(RasterPipelineDesc::new(
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
            resources: resource_manager.clone(),
            inner: Default::default(),
            operations: Default::default(),
            tonemapping,
        })
    }

    pub fn spawn<CB: FnOnce(&mut WorldSpawnContext)>(&self, cb: CB) {
        let mut world = self.inner.lock();
        cb(&mut WorldSpawnContext { world: &mut world });
    }

    pub fn remove(&self, handle: RenderWorldHandle) {
        self.operations.push(Operation::Remove(handle));
    }

    pub fn update_transform(&self, handle: RenderWorldHandle, tranform: Affine3A) {
        self.operations
            .push(Operation::UpdateTransform(handle, tranform));
    }

    pub fn set_model(&self, handle: RenderWorldHandle, model: &Arc<RenderModel>) {
        self.operations
            .push(Operation::SetModel(handle, model.clone()));
    }

    pub fn load_model(&self, handle: RenderWorldHandle, model: ModelHandle) {
        self.operations.push(Operation::LoadModel(handle, model));
    }

    pub fn change_light_color(&self, handle: RenderWorldHandle, color: Vec3) {
        self.operations
            .push(Operation::UpdaetLightColor(handle, color));
    }

    pub fn change_light_power(&self, handle: RenderWorldHandle, power: f32) {
        self.operations
            .push(Operation::UpdateLightPower(handle, power));
    }

    pub fn change_light_radius(&self, handle: RenderWorldHandle, radius: f32) {
        self.operations
            .push(Operation::UpdateLightRadius(handle, radius));
    }

    fn process_loading(&self, world: &mut RenderWorldInner) {
        self.resources.resolve(|context| {
            for object in world.objects.iter_mut() {
                match object {
                    WorldObject::PendingModel(handle) => match context.resolve_model(*handle) {
                        Ok(model) => {
                            if let Some(model) = model {
                                *object = WorldObject::Model(model);
                            }
                        }
                        Err(err) => {
                            warn!("Failed to load model for node {}: {}", handle, err);
                            *object = WorldObject::Empty;
                        }
                    },
                    _ => {}
                }
            }
        });
    }

    fn commit(&self, world: &mut RenderWorldInner) {
        puffin::profile_function!();
        while let Some(op) = self.operations.pop() {
            match op {
                Operation::Remove(handle) => world.remove(handle),
                Operation::UpdateTransform(handle, transform) => {
                    world.update_transform(handle, &transform);
                }
                Operation::UpdaetLightColor(handle, color) => {
                    world.update_light_color(handle, color)
                }
                Operation::UpdateLightPower(handle, power) => {
                    world.update_light_power(handle, power)
                }
                Operation::UpdateLightRadius(handle, radius) => {
                    world.update_light_radius(handle, radius)
                }
                Operation::SetModel(handle, model) => world.update_render_model(handle, model),
                Operation::LoadModel(handle, model) => world.load_model(handle, model),
            }
        }
    }

    pub fn render(
        &self,
        camera: RenderView,
        context: &RenderContext,
        pool: &RenderTargetPool,
    ) -> Result<(), kiri_gfx::Error> {
        puffin::profile_function!();
        let mut world = self.inner.lock();
        self.commit(&mut world);
        self.process_loading(&mut world);
        let culled = world.cull(&camera.camera.frustum());
        drop(world);
        let depth = Self::render_zprepass(context, pool, &camera.camera, &culled)?;
        let hdr = Self::render_hdr(
            context,
            pool,
            depth,
            &camera.camera,
            camera.ambient,
            &culled,
        )?;
        let ldr = self.postprocess(context, pool, hdr, camera.expouse)?;
        self.copy_to_backbuffer(context, ldr);
        Ok(())
    }

    fn render_zprepass<'a>(
        context: &'a RenderContext,
        pool: &'a RenderTargetPool,
        camera: &impl Camera,
        culled: &CullData,
    ) -> Result<TransientImage<'a>, kiri_gfx::Error> {
        puffin::profile_function!();
        let depth = pool.get_image(
            vk::Format::D24_UNORM_S8_UINT,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        let mut pass = RasterizerPassBuilder::new(
            "zpass",
            &[],
            Some(
                RenderTarget::new(depth.handle)
                    .clear_depth_stencil(1.0, 0)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
            ),
        );
        let view = camera.view();
        let projection = Mat4::from_cols(
            vec4(1.0, 0.0, 0.0, 0.0),
            vec4(0.0, -1.0, 0.0, 0.0),
            vec4(0.0, 0.0, 0.5, 0.0),
            vec4(0.0, 0.0, 0.5, 1.0),
        ) * camera.projection();
        let view_projection = projection * view;
        let pass_data = context.push_dynamic_data(&[GpuZPassData {
            view,
            projection,
            view_projection,
        }])?;
        let pass_ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                RENDER_PASS_DESCRIPTOR_LAYOUT,
            )
            .bind_uniform_buffer(0, pass_data),
        )?;
        Self::generate_commands::<true>(context, &mut pass, &culled.ops, &culled.depth, pass_ds);
        context.submit(pass.build());
        Ok(depth)
    }

    fn render_hdr<'a>(
        context: &'a RenderContext,
        pool: &'a RenderTargetPool,
        depth: TransientImage<'a>,
        camera: &impl Camera,
        ambient: (Vec3, Vec3, Vec3),
        culled: &CullData,
    ) -> Result<TransientImage<'a>, kiri_gfx::Error> {
        puffin::profile_function!();
        let hdr = pool.get_image(
            vk::Format::R16G16B16A16_SFLOAT,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?;
        let view = camera.view();
        let projection = Mat4::from_cols(
            vec4(1.0, 0.0, 0.0, 0.0),
            vec4(0.0, -1.0, 0.0, 0.0),
            vec4(0.0, 0.0, 0.5, 0.0),
            vec4(0.0, 0.0, 0.5, 1.0),
        ) * camera.projection();
        let (light_transform, light) = culled
            .directional_lights
            .iter()
            .max_by(|x, y| x.1.power.total_cmp(&y.1.power))
            .unwrap_or(&(
                Affine3A::IDENTITY,
                DirectionalLight {
                    color: Vec3::ZERO,
                    power: 0.0,
                },
            ));
        let view_projection = projection * view;
        let eye_position = view.transform_point3a(Vec3A::ZERO);
        let pass_data = context.push_dynamic_data(&[GpuPassData {
            view,
            projection,
            view_projection,
            eye_position,
            light: DirectionalLightGpuData {
                direction: light_transform.transform_vector3a(Vec3A::X),
                color: light.color.into(),
            },
            ambient: HemisphericalAmbientGpuData {
                top: ambient.0.into(),
                middle: ambient.1.into(),
                bottom: ambient.2.into(),
            },
        }])?;
        let pass_ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                RENDER_PASS_DESCRIPTOR_LAYOUT,
            )
            .bind_uniform_buffer(0, pass_data),
        )?;
        let mut pass = RasterizerPassBuilder::new(
            "main",
            &[RenderTarget::new(hdr.handle)
                .clear_color([0.0, 0.0, 0.0, 1.0])
                .initial_layout(vk::ImageLayout::UNDEFINED)],
            Some(RenderTarget::new(depth.handle).load().discard()),
        );
        Self::generate_commands::<false>(context, &mut pass, &culled.ops, &culled.opaque, pass_ds);
        Self::generate_commands::<false>(
            context,
            &mut pass,
            &culled.ops,
            &culled.transparent,
            pass_ds,
        );
        context.submit(pass.build());
        Ok(hdr)
    }

    fn postprocess<'a>(
        &self,
        context: &'a RenderContext,
        pool: &'a RenderTargetPool,
        hdr: TransientImage,
        expouse: f32,
    ) -> Result<TransientImage<'a>, kiri_gfx::Error> {
        // TODO:: HDR bloom
        let ldr = self.tonemapping(context, pool, hdr, expouse)?;
        Ok(ldr)
    }

    fn tonemapping<'a>(
        &self,
        context: &'a RenderContext,
        pool: &'a RenderTargetPool,
        hdr: TransientImage,
        expouse: f32,
    ) -> Result<TransientImage<'a>, kiri_gfx::Error> {
        let post = pool.get_image(
            vk::Format::A2R10G10B10_UNORM_PACK32,
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let mut pass = RasterizerPassBuilder::new(
            "tonemapping",
            &[RenderTarget::new(post.handle).initial_layout(vk::ImageLayout::UNDEFINED)],
            None,
        )
        .read_image(ImageDependency::color(hdr.handle));

        let ds = context.get_descriptor_set(
            DescriptorSetBuilder::new(
                vk::ShaderStageFlags::ALL_GRAPHICS,
                POSTPROCESS_DESCRIPTOR_LAYOUT,
            )
            .bind_image(0, hdr.handle, vk::ImageAspectFlags::COLOR)
            .bind_uniform_buffer(
                1,
                context.push_dynamic_data(&[TonemappingGpuData { expouse: expouse }])?,
            ),
        )?;

        pass.draw(Self::fullscreen_quad(context, self.tonemapping, ds)?);

        context.submit(pass.build());
        Ok(post)
    }

    fn fullscreen_quad(
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

    fn copy_to_backbuffer(&self, context: &RenderContext, ldr: TransientImage) {
        context.submit(Box::new(FinalCompositionPassDispatcher::new(ldr.handle)));
    }

    fn generate_commands<const DEPTH_ONLY: bool>(
        context: &RenderContext,
        pass: &mut RasterizerPassBuilder,
        ops: &[RenderOpData],
        order: &[RenderOp],
        pass_ds: DescriptorHandle,
    ) {
        puffin::profile_function!();
        order
            .par_chunks(DRAWS_PER_STREAM)
            .map(|chunk| {
                let instance_ds = context
                    .get_descriptor_set(
                        DescriptorSetBuilder::new(
                            vk::ShaderStageFlags::ALL_GRAPHICS,
                            INSTANCE_DESCRIPTOR_LAYOUT,
                        )
                        .bind_dynamic_storage_buffer(
                            0,
                            context.get_temprary_buffer(),
                            (mem::size_of::<GpuInstanceData>() * DRAWS_PER_STREAM) as _,
                        ),
                    )
                    .unwrap();
                let mut writer = context
                    .write_dynamic_data::<GpuInstanceData>(DRAWS_PER_STREAM)
                    .unwrap();
                let mut stream = DrawStreamBuilder::default();
                stream.set_descriptor(PASS_BINDING_SLOT, Some(pass_ds));
                stream.set_descriptor(DYNAMIC_BINDING_SLOT, Some(instance_ds));
                chunk
                    .iter()
                    .enumerate()
                    .for_each(|(draw_index, render_op)| {
                        let op = &ops[render_op.op_index];
                        writer
                            .write(GpuInstanceData {
                                model: op.model,
                                uv_scale: op.uv_scale,
                            })
                            .unwrap();
                        stream.set_pipeline(render_op.pipeline);
                        stream.set_descriptor(MATERIAL_BINDING_SLOT, Some(render_op.ds));
                        stream.set_dynamic_offset(0, Some(writer.offset as _));
                        stream.set_vertex_buffer(0, op.vertex_positions);
                        if !DEPTH_ONLY {
                            stream.set_vertex_buffer(1, op.vertex_attributes);
                        }
                        stream.set_vertex_offset(op.vertex_offset as _);
                        stream.set_index_buffer(op.index_buffer);
                        stream.draw(op.first_index, op.index_count, draw_index as _, 1);
                    });
                stream.build()
            })
            .collect::<Vec<_>>()
            .drain(..)
            .for_each(|stream| pass.draw(stream));
    }
}

#[derive(Debug, Default)]
struct RenderWorldInner {
    objects: Vec<WorldObject>,
    empty: Vec<u32>,
    generations: Vec<u32>,
    transforms: Vec<Affine3A>,
}

impl RenderWorldInner {
    fn spawn(&mut self, transform: &Affine3A, object: WorldObject) -> RenderWorldHandle {
        let handle = self.allocate(transform);
        self.objects[handle.index as usize] = object;
        handle
    }

    fn allocate(&mut self, transform: &Affine3A) -> RenderWorldHandle {
        if let Some(index) = self.empty.pop() {
            self.transforms[index as usize] = *transform;
            RenderWorldHandle {
                index,
                generation: self.generations[index as usize],
            }
        } else {
            let index = self.generations.len();
            assert!(index < u32::MAX as usize, "Far too many world objects");
            self.generations.push(0);
            self.objects.push(WorldObject::Empty);
            self.transforms.push(*transform);
            RenderWorldHandle {
                index: index as u32,
                generation: 0,
            }
        }
    }

    fn remove(&mut self, handle: RenderWorldHandle) {
        if !self.is_valid(handle) {
            return;
        }
        self.invalidate(handle);
    }

    fn update_transform(&mut self, handle: RenderWorldHandle, transform: &Affine3A) {
        if !self.is_valid(handle) {
            return;
        }
        // TODO: Invalidate light cache for model if model moved (make it dirty), same if we moved local light source
        self.transforms[handle.index as usize] = *transform;
    }

    fn update_light_color(&mut self, handle: RenderWorldHandle, color: Vec3) {
        if !self.is_valid(handle) {
            return;
        }
        match &mut self.objects[handle.index as usize] {
            WorldObject::DirectionalLight(light) => light.color = color,
            WorldObject::PointLight(light) => light.color = color,
            _ => {}
        }
    }

    fn update_light_power(&mut self, handle: RenderWorldHandle, power: f32) {
        if !self.is_valid(handle) {
            return;
        }
        match &mut self.objects[handle.index as usize] {
            WorldObject::DirectionalLight(light) => light.power = power,
            WorldObject::PointLight(light) => light.power = power,
            _ => {}
        }
    }

    fn update_light_radius(&mut self, handle: RenderWorldHandle, radius: f32) {
        if !self.is_valid(handle) {
            return;
        }
        match &mut self.objects[handle.index as usize] {
            // TODO: invaludate per-object light cache.
            WorldObject::PointLight(light) => light.radius = radius,
            _ => {}
        }
    }

    fn update_render_model(&mut self, handle: RenderWorldHandle, model: Arc<RenderModel>) {
        if !self.is_valid(handle) {
            return;
        }
        match &self.objects[handle.index as usize] {
            WorldObject::Model(_) | WorldObject::PendingModel(_) => {
                // Invaludate light cache
                self.objects[handle.index as usize] = WorldObject::Model(model)
            }
            _ => {}
        }
    }

    fn load_model(&mut self, handle: RenderWorldHandle, model: ModelHandle) {
        if !self.is_valid(handle) {
            return;
        }
        match &self.objects[handle.index as usize] {
            WorldObject::Model(_) | WorldObject::PendingModel(_) => {
                self.objects[handle.index as usize] = WorldObject::PendingModel(model)
            }
            _ => {}
        }
    }

    fn is_valid(&self, handle: RenderWorldHandle) -> bool {
        let index = handle.index as usize;
        index < self.objects.len() && self.generations[index as usize] == handle.generation
    }

    fn invalidate(&mut self, handle: RenderWorldHandle) {
        debug_assert!(self.is_valid(handle));
        let index = handle.index as usize;
        self.generations[index] = self.generations[index].wrapping_add(1);
        self.objects[index] = WorldObject::Empty;
        self.empty.push(handle.index);
    }

    fn cull(&self, frustrum: &[Plane]) -> CullData {
        puffin::profile_function!();
        let mut ops = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
        let mut depth = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
        let mut opaque = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
        let mut transparent = Vec::with_capacity(DEFAULT_RENDER_OP_CAPACITY);
        let mut directional_lights = Vec::new();
        for (index, object) in self.objects.iter().enumerate() {
            let transform = self.transforms[index];
            match object {
                WorldObject::DirectionalLight(light) => {
                    directional_lights.push((transform, *light));
                }
                WorldObject::Model(model) => {
                    if model.bounds.transform(transform).is_visible(frustrum) {
                        for (node_index, mesh_index) in model
                            .node_to_mesh
                            .iter()
                            .copied()
                            .map(|(x, y)| (x as usize, y as usize))
                        {
                            let transform = transform * model.world_transforms[node_index];
                            let mesh = &model.meshes[mesh_index];
                            if mesh.bounds.transform(transform).is_visible(frustrum) {
                                for surface in mesh.surfaces.iter() {
                                    let index = ops.len();
                                    ops.push(RenderOpData {
                                        model: transform.into(),
                                        vertex_positions: mesh.positions,
                                        vertex_attributes: mesh.attributes,
                                        index_buffer: mesh.index_buffer,
                                        first_index: surface.first_index,
                                        index_count: surface.index_count,
                                        vertex_offset: surface.vertex_offset,
                                        uv_scale: mesh.uv_scale,
                                    });
                                    if surface.material.depth.is_valid() {
                                        depth.push(RenderOp {
                                            pipeline: surface.material.depth,
                                            ds: surface.material.instance.ds,
                                            op_index: index,
                                        });
                                    }
                                    // TODO: choose main or transparent based on overriden color
                                    if surface.material.main.is_valid() {
                                        opaque.push(RenderOp {
                                            pipeline: surface.material.main,
                                            ds: surface.material.instance.ds,
                                            op_index: index,
                                        });
                                    }
                                    if surface.material.transparent.is_valid() {
                                        transparent.push((
                                            0.0,
                                            RenderOp {
                                                pipeline: surface.material.transparent,
                                                ds: surface.material.instance.ds,
                                                op_index: index,
                                            },
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        radsort::sort_by_key(&mut depth, |x| {
            ((x.pipeline.index() as u64) << 32) | x.ds.index() as u64
        });
        radsort::sort_by_key(&mut opaque, |x| {
            ((x.pipeline.index() as u64) << 32) | x.ds.index() as u64
        });
        radsort::sort_by_key(&mut transparent, |(depth, _)| (depth * 1000.0) as u64);
        radsort::sort_by_key(&mut directional_lights, |(_, light)| {
            (light.power * 1000.0) as u64
        });
        let transparent = transparent.drain(..).map(|(_, op)| op).collect::<Vec<_>>();
        CullData {
            ops,
            depth,
            opaque,
            transparent,
            directional_lights,
        }
    }
}
