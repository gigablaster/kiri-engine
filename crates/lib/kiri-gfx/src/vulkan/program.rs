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
};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use rspirv_reflect::{BindingCount, DescriptorInfo, Reflection};

use crate::{Error, ProgramHandle};

use super::{RenderContext, SamplerDesc};

const MAX_SAMPLERS: usize = 32;
pub(crate) const BINDLESS_BINDING_SLOT: usize = 0;
pub(crate) const DYNAMIC_BINDING_SLOT: usize = 3;
pub(crate) const MAX_DESCRIPTOR_SETS: usize = 4;

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBindingDesc<'a> {
    pub name: &'a str,
    pub slot: usize,
    pub ty: vk::DescriptorType,
    pub count: u32,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorSetLayoutDesc<'a> {
    pub stage: vk::ShaderStageFlags,
    pub set: &'a [DescriptorBindingDesc<'a>],
}

type ReflectedDescriptorSet = HashMap<usize, (String, vk::DescriptorType, u32)>;
type RelfectedDescriptorSetLayout = HashMap<usize, ReflectedDescriptorSet>;

pub(crate) fn create_descriptor_set_layout(
    device: &ash::Device,
    immutable_samplers: &HashMap<SamplerDesc, vk::Sampler>,
    set: &DescriptorSetLayoutDesc,
) -> Result<vk::DescriptorSetLayout, Error> {
    let mut samplers = ArrayVec::<_, MAX_SAMPLERS>::new();
    let mut bindings = HashMap::with_capacity(set.set.len());
    for binding in set.set.iter() {
        match binding.ty {
            vk::DescriptorType::UNIFORM_BUFFER
            | vk::DescriptorType::STORAGE_BUFFER
            | vk::DescriptorType::STORAGE_IMAGE
            | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            | vk::DescriptorType::SAMPLED_IMAGE => {
                bindings.insert(binding.slot, create_binding(set.stage, binding));
            }
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER | vk::DescriptorType::SAMPLER => {
                let sampler = immutable_samplers
                    .get(&get_suitable_sampler_desc())
                    .unwrap();
                samplers.push((sampler, binding.slot, 1, binding.ty, set.stage));
            }
            _ => panic!("Not yet implemented {:?}", binding.ty),
        };
    }
    for (sampler, slot, count, ty, stage) in &samplers {
        let layout_biding = vk::DescriptorSetLayoutBinding::default()
            .binding(*slot as _)
            .descriptor_count(*count as _)
            .descriptor_type(*ty)
            .stage_flags(*stage)
            .immutable_samplers(slice::from_ref(sampler));
        bindings.insert(*slot, layout_biding);
    }

    let layoyt = bindings.values().copied().collect::<Vec<_>>();
    let mut types = HashMap::with_capacity(set.set.len());
    bindings.into_iter().for_each(|(index, binding)| {
        types.insert(index, binding.descriptor_type);
    });
    let layout_create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layoyt);
    let layout = unsafe { device.create_descriptor_set_layout(&layout_create_info, None) }?;

    Ok(layout)
}

fn create_binding<'a>(
    stage: vk::ShaderStageFlags,
    binding: &'a DescriptorBindingDesc,
) -> vk::DescriptorSetLayoutBinding<'a> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding.slot as _)
        .descriptor_type(binding.ty)
        .descriptor_count(binding.count.into())
        .stage_flags(stage)
}

fn get_suitable_sampler_desc() -> SamplerDesc {
    SamplerDesc {
        texel_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode: vk::SamplerAddressMode::REPEAT,
        anisotropy_level: 16, // TODO:: control anisotropy level
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ShaderDesc<'a> {
    pub stage: vk::ShaderStageFlags,
    pub entry: &'a str,
    pub code: &'a [u8],
}

impl<'a> ShaderDesc<'a> {
    pub fn new(stage: vk::ShaderStageFlags, code: &'a [u8]) -> Self {
        Self {
            stage,
            entry: "main",
            code,
        }
    }

    pub fn vertex(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::VERTEX,
            entry: "main",
            code,
        }
    }

    pub fn fragment(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::FRAGMENT,
            entry: "main",
            code,
        }
    }

    pub fn compute(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::COMPUTE,
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
            stage: desc.stage,
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
                rspirv_reflect::DescriptorType::SAMPLER => vk::DescriptorType::SAMPLER,
                rspirv_reflect::DescriptorType::SAMPLED_IMAGE => vk::DescriptorType::SAMPLED_IMAGE,

                rspirv_reflect::DescriptorType::STORAGE_BUFFER if dynamic => {
                    vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                }
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER if dynamic => {
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                }
                rspirv_reflect::DescriptorType::STORAGE_BUFFER => {
                    vk::DescriptorType::STORAGE_BUFFER
                }
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER => {
                    vk::DescriptorType::UNIFORM_BUFFER
                }
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                }
                rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                }
                rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                }
                rspirv_reflect::DescriptorType::STORAGE_IMAGE => vk::DescriptorType::STORAGE_IMAGE,
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
    layouts: Vec<vk::DescriptorSetLayout>,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

impl Program {
    pub fn new(context: &RenderContext, shaders: &[ShaderDesc]) -> Result<Self, Error> {
        let mut stages = vk::ShaderStageFlags::empty();
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
                    .map(|(index, desc)| DescriptorBindingDesc {
                        name: &desc.0,
                        slot: *index,
                        ty: desc.1,
                        count: desc.2,
                    })
                    .collect::<Vec<_>>();
                let desc = DescriptorSetLayoutDesc {
                    stage: stages,
                    set: &descs,
                };
                create_descriptor_set_layout(&context.device, &context.samplers, &desc).unwrap()
            })
            .collect::<Vec<_>>();
        let layout_desc = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let pipeline_layout = unsafe { context.device.create_pipeline_layout(&layout_desc, None) }?;

        Ok(Self {
            layouts,
            pipeline_layout,
            shaders,
        })
    }

    pub(crate) fn shaders(&self) -> impl Iterator<Item = &Shader> {
        self.shaders.iter()
    }

    fn merge_reflected_layouts(
        layouts: &[&RelfectedDescriptorSetLayout],
    ) -> RelfectedDescriptorSetLayout {
        let mut result = RelfectedDescriptorSetLayout::new();
        layouts
            .iter()
            .for_each(|x| Self::merge_reflected_layout_set(&mut result, x));
        for i in 0..MAX_DESCRIPTOR_SETS {
            result.entry(i).or_insert(ReflectedDescriptorSet::default());
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
        self.layouts
            .iter()
            .for_each(|x| unsafe { device.destroy_descriptor_set_layout(*x, None) });
    }
}

impl<'game> RenderContext<'game> {
    pub fn create_program(&self, shaders: &[ShaderDesc]) -> Result<ProgramHandle, Error> {
        let program = Program::new(&self, shaders)?;
        let mut programs = self.programs.write();
        let index = programs.len();
        programs.push(program);
        Ok(ProgramHandle(index as u32))
    }
}

#[cfg(test)]
mod test {
    use ash::vk;

    use super::Program;

    use super::{ReflectedDescriptorSet, RelfectedDescriptorSetLayout};

    #[test]
    fn merge_refected_layouts() {
        let mut set1 = ReflectedDescriptorSet::new();
        set1.insert(0, ("shared1".into(), vk::DescriptorType::SAMPLED_IMAGE, 1));
        set1.insert(1, ("shared2".into(), vk::DescriptorType::UNIFORM_BUFFER, 1));
        let mut set2 = ReflectedDescriptorSet::new();
        set2.insert(0, ("set_a".into(), vk::DescriptorType::STORAGE_BUFFER, 1));
        let mut set3 = ReflectedDescriptorSet::new();
        set3.insert(
            1,
            ("set_b".into(), vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1),
        );

        let mut combined = ReflectedDescriptorSet::new();
        combined.insert(0, ("set_a".into(), vk::DescriptorType::STORAGE_BUFFER, 1));
        combined.insert(
            1,
            ("set_b".into(), vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1),
        );

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
