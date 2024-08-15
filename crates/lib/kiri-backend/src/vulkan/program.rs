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
    collections::{BTreeMap, HashMap},
    ffi::{CStr, CString},
    slice,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use gpu_descriptor::DescriptorTotalCount;
use rspirv_reflect::{BindingCount, DescriptorInfo, Reflection};

use crate::{BindType, Error, ProgramHandle, ShaderStage};

use super::{RenderDevice, SamplerDesc};

const MAX_SAMPLERS: usize = 32;
pub const FRAME_BINDING_SLOT: usize = 0;
pub const OBJECT_BINDING_SLOT: usize = 1;
pub const MATERIAL_BINDING_SLOT: usize = 2;
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
    pub name: &'a str,
    pub slot: u32,
    pub ty: BindType,
    pub count: u32,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupDesc<'a> {
    pub stage: ShaderStage,
    pub set: &'a [BindGroupSlotDesc<'a>],
}

impl<'a> BindGroupDesc<'a> {
    pub fn to_pool_size(self, count: u32) -> Vec<vk::DescriptorPoolSize> {
        self.set
            .iter()
            .map(|x| {
                vk::DescriptorPoolSize::default()
                    .ty(x.ty.into())
                    .descriptor_count(x.count * count)
            })
            .collect::<Vec<_>>()
    }
}

type ReflectedDescriptorSet = HashMap<usize, (String, BindType, u32)>;
type RelfectedDescriptorSetLayout = HashMap<usize, ReflectedDescriptorSet>;

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
    set: &BindGroupDesc,
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
        .map(|(index, binding)| (*index as usize, binding.descriptor_type))
        .collect::<HashMap<_, _>>();
    let names = set
        .set
        .iter()
        .map(|x| (x.name.to_owned(), x.slot as usize))
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
    binding: &'a BindGroupSlotDesc,
) -> vk::DescriptorSetLayoutBinding<'a> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding.slot as _)
        .descriptor_type(binding.ty.into())
        .descriptor_count(binding.count)
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
    } else if name.ends_with("_lr") {
        SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode: vk::SamplerAddressMode::REPEAT,
            anisotropy_level: 8, // TODO:: control anisotropy level
        }
    } else {
        panic!("Unkown sampler type {}", name)
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
    layout: RelfectedDescriptorSetLayout,
}

impl Shader {
    fn new(device: &ash::Device, desc: &ShaderDesc) -> Result<Self, Error> {
        let layout = Self::reflect(desc.code)?;
        let shader_create_info =
            vk::ShaderModuleCreateInfo::default().code(desc.code.as_slice_of::<u32>().unwrap());

        let shader = unsafe { device.create_shader_module(&shader_create_info, None) }?;
        Ok(Self {
            raw: shader,
            stage: desc.stage.into(),
            entry: CString::new(desc.entry).unwrap(),
            layout,
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

    fn reflect(code: &[u8]) -> Result<RelfectedDescriptorSetLayout, Error> {
        let reflection = Reflection::new_from_spirv(code)?.get_descriptor_sets()?;
        let mut layout = RelfectedDescriptorSetLayout::default();
        for (index, set) in reflection.into_iter() {
            layout.insert(
                index as usize,
                Self::reflect_descriptor(set, index == DYNAMIC_BINDING_SLOT as u32)?,
            );
        }
        Ok(layout)
    }

    fn reflect_descriptor(
        value: BTreeMap<u32, DescriptorInfo>,
        dynamic: bool,
    ) -> Result<ReflectedDescriptorSet, Error> {
        let mut result = ReflectedDescriptorSet::new();
        for (index, info) in value.into_iter() {
            if info.binding_count != BindingCount::One {
                return Err(Error::ArrayBindingsArentSupported);
            }
            let ty = match info.ty {
                rspirv_reflect::DescriptorType::SAMPLER => BindType::Sampler,
                rspirv_reflect::DescriptorType::SAMPLED_IMAGE => BindType::SampledImage,

                rspirv_reflect::DescriptorType::STORAGE_BUFFER if dynamic => {
                    BindType::DynamicStorage
                }
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER if dynamic => {
                    BindType::DynamicUniform
                }
                rspirv_reflect::DescriptorType::STORAGE_BUFFER => BindType::Storage,
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER => BindType::Uniform,
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER_DYNAMIC => BindType::DynamicUniform,
                rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => BindType::DynamicStorage,
                rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    BindType::CombinedSampledImage
                }
                _ => panic!("Not supported {}", info.ty.0),
            };
            let count = match info.binding_count {
                rspirv_reflect::BindingCount::One => 1,
                rspirv_reflect::BindingCount::StaticSized(count) => count as u32,
                _ => unimplemented!("{:?}", info.binding_count),
            };

            result.insert(index as usize, (info.name, ty, count));
        }
        Ok(result)
    }
}

/// Shader program similar to what we had in OpenGL.
///
/// Contains shader modules and layouts needed to create PSOs and descriptor sets.
#[derive(Debug)]
pub(crate) struct Program {
    pub(crate) shaders: Vec<Shader>,
    pub(crate) layouts: Vec<Arc<DescriptorSetLayout>>,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

impl Program {
    pub fn new(context: &RenderDevice, shaders: &[ShaderDesc]) -> Result<Self, Error> {
        let mut stages = ShaderStage::empty();
        let shaders = shaders
            .iter()
            .map(|desc| {
                stages |= desc.stage;
                Shader::new(&context.device, desc).unwrap()
            })
            .collect::<Vec<_>>();
        let layouts = shaders.iter().map(|x| &x.layout).collect::<Vec<_>>();
        let layout = Self::merge_reflected_layouts(&layouts);
        let mut layout = layout.into_iter().collect::<Vec<_>>();
        layout.sort_by_key(|(index, _)| *index);
        let layouts = layout
            .iter()
            .map(|(_, set)| {
                let descs = set
                    .iter()
                    .map(|(index, desc)| BindGroupSlotDesc {
                        name: &desc.0,
                        slot: *index as u32,
                        ty: desc.1,
                        count: desc.2,
                    })
                    .collect::<Vec<_>>();
                let desc = BindGroupDesc {
                    stage: stages,
                    set: &descs,
                };
                create_descriptor_set_layout(&context.device, &context.samplers, &desc).unwrap()
            })
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

    fn merge_reflected_layouts(
        layouts: &[&RelfectedDescriptorSetLayout],
    ) -> RelfectedDescriptorSetLayout {
        let mut result = RelfectedDescriptorSetLayout::new();
        layouts
            .iter()
            .for_each(|x| Self::merge_reflected_layout_set(&mut result, x));
        for i in 0..MAX_DESCRIPTOR_SETS {
            result.entry(i).or_default();
        }
        result
    }

    fn merge_reflected_layout_set(
        target: &mut RelfectedDescriptorSetLayout,
        next: &RelfectedDescriptorSetLayout,
    ) {
        next.iter().for_each(|(index, set)| {
            target
                .entry(*index)
                .and_modify(|existing| {
                    set.iter().for_each(|(index, set)| {
                        existing.insert(*index, set.clone());
                    })
                })
                .or_insert(set.clone());
        });
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
    pub fn create_program(&self, shaders: &[ShaderDesc]) -> Result<ProgramHandle, Error> {
        let program = Program::new(self, shaders)?;
        let mut programs = self.programs.write();
        let index = programs.len();
        programs.push(program);
        Ok(ProgramHandle(index as u32))
    }
}

#[cfg(test)]
mod test {
    use crate::BindType;

    use super::Program;

    use super::{ReflectedDescriptorSet, RelfectedDescriptorSetLayout};

    #[test]
    fn merge_refected_layouts() {
        let mut set1 = ReflectedDescriptorSet::new();
        set1.insert(0, ("shared1".into(), BindType::SampledImage, 1));
        set1.insert(1, ("shared2".into(), BindType::Uniform, 1));
        let mut set2 = ReflectedDescriptorSet::new();
        set2.insert(0, ("set_a".into(), BindType::Storage, 1));
        let mut set3 = ReflectedDescriptorSet::new();
        set3.insert(1, ("set_b".into(), BindType::DynamicStorage, 1));

        let mut combined = ReflectedDescriptorSet::new();
        combined.insert(0, ("set_a".into(), BindType::Storage, 1));
        combined.insert(1, ("set_b".into(), BindType::DynamicStorage, 1));

        let mut a = RelfectedDescriptorSetLayout::new();
        a.insert(0, set1.clone());
        a.insert(2, set2);
        // a.insert(0, )
        let mut b = RelfectedDescriptorSetLayout::new();
        b.insert(0, set1.clone());
        b.insert(2, set3);
        let merged = Program::merge_reflected_layouts(&[&a, &b]);
        let rset1 = merged.get(&0).unwrap();
        let rset2 = merged.get(&2).unwrap();
        assert!(merged.get(&1).unwrap().is_empty());
        assert!(merged.get(&3).unwrap().is_empty());
        assert_eq!(rset1, &set1);
        assert_eq!(rset2, &combined);
    }
}
