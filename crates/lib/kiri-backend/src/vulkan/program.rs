// Copyright (C) 2023-2024 gigablaster

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
    collections::HashMap,
    ffi::{CStr, CString},
    slice,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use gpu_descriptor::DescriptorTotalCount;

use crate::{BindType, Error, ProgramHandle, ShaderStage};

use super::{RenderDevice, SamplerDesc};

const MAX_SAMPLERS: usize = 32;
pub const FRAME_BINDING_SLOT: usize = 0;
pub const MATERIAL_BINDING_SLOT: usize = 1;
pub const OBJECT_BINDING_SLOT: usize = 2;
pub const DYNAMIC_BINDING_SLOT: usize = 3;
pub(crate) const MAX_DESCRIPTOR_SETS: usize = 4;

impl From<BindType> for vk::DescriptorType {
    fn from(value: BindType) -> Self {
        match value {
            BindType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
            BindType::DynamicUniform => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            BindType::Storage => vk::DescriptorType::STORAGE_BUFFER,
            BindType::DynamicStorage => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            BindType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            BindType::CombinedSampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            BindType::Sampler => vk::DescriptorType::SAMPLER,
        }
    }
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        let mut result = vk::ShaderStageFlags::empty();
        if value.contains(ShaderStage::Vertex) {
            result |= vk::ShaderStageFlags::VERTEX;
        }
        if value.contains(ShaderStage::Fragment) {
            result |= vk::ShaderStageFlags::FRAGMENT;
        }
        if value.contains(ShaderStage::Compute) {
            result |= vk::ShaderStageFlags::COMPUTE;
        }
        result
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupSlotDesc<'a> {
    pub slot: usize,
    pub name: &'a str,
    pub ty: BindType,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupDesc<'a> {
    pub stage: ShaderStage,
    pub set: &'a [BindGroupSlotDesc<'a>],
}

#[derive(Debug)]
pub(crate) struct DescriptorSetLayout {
    pub raw: vk::DescriptorSetLayout,
    pub count: DescriptorTotalCount,
    pub types: HashMap<usize, vk::DescriptorType>,
    pub names: HashMap<String, usize>,
}

impl DescriptorSetLayout {
    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_set_layout(self.raw, None) }
    }
}

pub(crate) fn create_descriptor_set_layout(
    device: &ash::Device,
    immutable_samplers: &HashMap<SamplerDesc, vk::Sampler>,
    set: &'static BindGroupDesc<'static>,
) -> Result<Arc<DescriptorSetLayout>, Error> {
    let mut samplers = ArrayVec::<_, MAX_SAMPLERS>::new();
    let mut bindings = HashMap::with_capacity(set.set.len());
    for binding in set.set.iter() {
        match binding.ty {
            BindType::Uniform
            | BindType::Storage
            | BindType::DynamicUniform
            | BindType::DynamicStorage
            | BindType::SampledImage => {
                bindings.insert(binding.slot, create_binding(set.stage, binding));
            }
            BindType::CombinedSampledImage | BindType::Sampler => {
                let sampler = immutable_samplers
                    .get(&get_suitable_sampler_desc(binding.name))
                    .unwrap();
                samplers.push((sampler, binding.slot, 1, binding.ty, set.stage));
                if binding.ty == BindType::CombinedSampledImage {
                    bindings.insert(binding.slot, create_binding(set.stage, binding));
                }
            }
        };
    }
    let mut count = DescriptorTotalCount::default();
    for binding in bindings.values() {
        match binding.descriptor_type {
            vk::DescriptorType::UNIFORM_BUFFER => count.uniform_buffer += binding.descriptor_count,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                count.uniform_buffer_dynamic += binding.descriptor_count
            }
            vk::DescriptorType::STORAGE_BUFFER => count.storage_buffer += binding.descriptor_count,
            vk::DescriptorType::STORAGE_IMAGE => count.storage_image += binding.descriptor_count,
            vk::DescriptorType::SAMPLED_IMAGE => count.sampled_image += binding.descriptor_count,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                count.combined_image_sampler += binding.descriptor_count
            }
            vk::DescriptorType::SAMPLER => count.sampler += binding.descriptor_count,
            _ => panic!(),
        }
    }
    for (sampler, slot, count, ty, stage) in &samplers {
        let layout_biding = vk::DescriptorSetLayoutBinding::default()
            .binding(*slot as _)
            .descriptor_count(*count as _)
            .descriptor_type((*ty).into())
            .stage_flags((*stage).into())
            .immutable_samplers(slice::from_ref(sampler));
        bindings.insert(*slot, layout_biding);
    }

    let layout = bindings.values().copied().collect::<Vec<_>>();
    let types = bindings
        .iter()
        .map(|(index, binding)| (*index, binding.descriptor_type))
        .collect::<HashMap<_, _>>();
    let names = set
        .set
        .iter()
        .map(|x| (x.name.to_owned(), x.slot))
        .collect::<HashMap<_, _>>();
    let layout_create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout);
    let layout = unsafe { device.create_descriptor_set_layout(&layout_create_info, None) }?;

    Ok(Arc::new(DescriptorSetLayout {
        raw: layout,
        count,
        types,
        names,
    }))
}

fn create_binding<'a>(
    stage: ShaderStage,
    binding: &'a BindGroupSlotDesc<'a>,
) -> vk::DescriptorSetLayoutBinding<'a> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding.slot as _)
        .descriptor_type(binding.ty.into())
        .descriptor_count(1)
        .stage_flags(stage.into())
}

fn get_suitable_sampler_desc(name: &str) -> SamplerDesc {
    if name.ends_with("_nr") {
        SamplerDesc {
            texel_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode: vk::SamplerAddressMode::REPEAT,
            anisotropy_level: 0, // TODO:: control anisotropy level
        }
    } else if name.ends_with("_nb") {
        SamplerDesc {
            texel_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            anisotropy_level: 0, // TODO:: control anisotropy level
        }
    } else if name.ends_with("_nm") {
        SamplerDesc {
            texel_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode: vk::SamplerAddressMode::MIRRORED_REPEAT,
            anisotropy_level: 0, // TODO:: control anisotropy level
        }
    } else if name.ends_with("_lb") {
        SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            anisotropy_level: 8, // TODO:: control anisotropy level
        }
    } else if name.ends_with("_lm") {
        SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode: vk::SamplerAddressMode::MIRRORED_REPEAT,
            anisotropy_level: 8, // TODO:: control anisotropy level
        }
    } else {
        SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode: vk::SamplerAddressMode::REPEAT,
            anisotropy_level: 8, // TODO:: control anisotropy level
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ShaderDesc<'a> {
    pub stage: ShaderStage,
    pub entry: &'a str,
    pub code: &'a [u8],
}

impl<'a> ShaderDesc<'a> {
    pub fn new(stage: ShaderStage, code: &'a [u8]) -> Self {
        Self {
            stage,
            entry: "main",
            code,
        }
    }

    pub fn vertex(code: &'a [u8]) -> Self {
        Self {
            stage: ShaderStage::Vertex,
            entry: "main",
            code,
        }
    }

    pub fn fragment(code: &'a [u8]) -> Self {
        Self {
            stage: ShaderStage::Fragment,
            entry: "main",
            code,
        }
    }

    pub fn compute(code: &'a [u8]) -> Self {
        Self {
            stage: ShaderStage::Compute,
            entry: "main",
            code,
        }
    }

    pub fn entry(mut self, entry: &'a str) -> Self {
        self.entry = entry;
        self
    }
}

#[derive(Debug)]
pub(crate) struct Shader {
    pub raw: vk::ShaderModule,
    stage: vk::ShaderStageFlags,
    entry: CString,
}

impl Shader {
    fn new(device: &ash::Device, desc: &ShaderDesc) -> Result<Self, Error> {
        let shader_create_info =
            vk::ShaderModuleCreateInfo::default().code(desc.code.as_slice_of::<u32>().unwrap());

        let shader = unsafe { device.create_shader_module(&shader_create_info, None) }?;
        Ok(Self {
            raw: shader,
            stage: desc.stage.into(),
            entry: CString::new(desc.entry).unwrap(),
        })
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_shader_module(self.raw, None) }
    }

    pub fn stage(&self) -> vk::ShaderStageFlags {
        self.stage
    }

    pub fn entry(&self) -> &CStr {
        &self.entry
    }
}

/// Shader program similar to what we had in OpenGL.
///
/// Contains shader modules and layouts needed to create PSOs and descriptor sets.
#[derive(Debug)]
pub struct Program {
    pub(crate) shaders: Vec<Shader>,
    pub(crate) layouts: Vec<Arc<DescriptorSetLayout>>,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

impl Program {
    pub fn new(
        context: &RenderDevice,
        shaders: &[ShaderDesc],
        layout: &'static [BindGroupDesc<'static>],
    ) -> Result<Self, Error> {
        let mut stages = ShaderStage::empty();
        let shaders = shaders
            .iter()
            .map(|desc| {
                stages |= desc.stage;
                Shader::new(&context.device, desc).unwrap()
            })
            .collect::<Vec<_>>();
        let layouts = layout
            .iter()
            .map(|set| context.get_or_create_layout(set).unwrap())
            .collect::<Vec<_>>();
        let vk_layouts = layouts
            .iter()
            .map(|x| x.raw)
            .collect::<ArrayVec<_, MAX_DESCRIPTOR_SETS>>();
        let layout_desc = vk::PipelineLayoutCreateInfo::default().set_layouts(&vk_layouts);
        let pipeline_layout = unsafe { context.device.create_pipeline_layout(&layout_desc, None) }?;

        Ok(Self {
            layouts,
            pipeline_layout,
            shaders,
        })
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn free(self, device: &ash::Device) {
        self.shaders.iter().for_each(|shader| shader.free(device));
        unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None) };
        self.layouts.into_iter().for_each(|x| x.free(device));
    }
}

impl RenderDevice {
    pub fn create_program(
        &self,
        shaders: &[ShaderDesc],
        layout: &'static [BindGroupDesc<'static>],
    ) -> Result<ProgramHandle, Error> {
        let program = Program::new(self, shaders, layout)?;
        let mut programs = self.programs.write();
        Ok(programs.push(program))
    }

    pub fn destroy_program(&self, handle: ProgramHandle) {
        if let Some(program) = self.programs.write().remove(handle) {
            // Destory all pipeliens that use same pipeline layout
            let mut pipelines = self.pipelines.write();
            let mut to_destroy = pipelines
                .enumerate()
                .filter_map(|(handle, (_, layout))| {
                    if *layout == program.pipeline_layout {
                        Some(handle)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            to_destroy.drain(..).for_each(|handle| {
                if let Some((pipeline, _)) = pipelines.remove(handle) {
                    unsafe { self.device.destroy_pipeline(pipeline, None) };
                }
            });
            program.free(&self.device);
        }
    }
}
