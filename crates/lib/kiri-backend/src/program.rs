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
    ffi::CString,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk::{self};
use byte_slice_cast::AsSliceOf;
use kiri_common::TempList;
use rspirv_reflect::{BindingCount, DescriptorInfo, Reflection};

use crate::{DescriptorCount, Error, SamplerDesc};

use super::RenderDevice;

pub const PASS_BINDING_SLOT: usize = 0;
pub const OBJECT_BINDING_SLOT: usize = 1;
pub const MATERIAL_BINDING_SLOT: usize = 2;
pub const DYNAMIC_BINDING_SLOT: usize = 3;
pub const MAX_DESCRIPTOR_SETS: usize = 4;

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

type ReflectedDescriptorSetDesc = HashMap<u32, (String, vk::DescriptorType, u32)>;
type ReflectedDescriptorSetLayoutDesc = HashMap<u32, ReflectedDescriptorSetDesc>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DescriptorSetDesc {
    pub name: String,
    pub ty: vk::DescriptorType,
    pub count: u32,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct DescriptorSetLayoutDesc {
    layout: Vec<(u32, DescriptorSetDesc)>,
}

impl DescriptorSetLayoutDesc {
    pub fn slot(mut self, slot: u32, name: &str, ty: vk::DescriptorType, count: u32) -> Self {
        self.layout.push((
            slot,
            DescriptorSetDesc {
                name: name.to_owned(),
                ty,
                count,
            },
        ));
        self
    }

    pub fn get_descriptor_count(&self) -> DescriptorCount {
        let mut count = DescriptorCount::default();
        for (_, data) in &self.layout {
            match data.ty {
                vk::DescriptorType::SAMPLED_IMAGE => count.sampled_images += data.count,
                vk::DescriptorType::UNIFORM_BUFFER => count.unifroms_buffers += data.count,
                vk::DescriptorType::STORAGE_BUFFER => count.storage_buffers += data.count,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                    count.dynamic_uniform_buffers += data.count
                }
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    count.dynamic_storage_buffers += data.count
                }
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    count.combined_image_samplers += data.count
                }
                vk::DescriptorType::STORAGE_IMAGE => count.storage_images += data.count,
                ty => panic!("Descriptor set type {:?} not supported", ty),
            }
        }
        count
    }

    pub fn get_slot_by_name(&self, name: &str) -> Option<u32> {
        self.layout.iter().find_map(
            |(slot, data)| {
                if name == data.name {
                    Some(*slot)
                } else {
                    None
                }
            },
        )
    }

    pub fn get_desc(&self, slot: u32) -> Option<&DescriptorSetDesc> {
        self.layout
            .iter()
            .find_map(|(x, data)| if slot == *x { Some(data) } else { None })
    }

    /// Remove names and sort by slot.
    ///
    /// This way same layouts but woth different slot names will be seen as same.
    pub(super) fn normalize(&self) -> DescriptorSetLayoutDesc {
        let mut normalized = self.clone();
        normalized.layout.iter_mut().for_each(|x| {
            x.1.name = Default::default();
        });
        normalized.layout.sort_by(|a, b| a.0.cmp(&b.0));
        normalized
    }
}

impl From<ReflectedDescriptorSetDesc> for DescriptorSetLayoutDesc {
    fn from(value: ReflectedDescriptorSetDesc) -> Self {
        let layout = value
            .into_iter()
            .map(|(slot, data)| {
                (
                    slot,
                    DescriptorSetDesc {
                        name: data.0,
                        ty: data.1,
                        count: data.2,
                    },
                )
            })
            .collect();
        DescriptorSetLayoutDesc { layout }
    }
}

const MAX_SHADERS: usize = 2;

#[derive(Debug)]
pub struct Program {
    device: Arc<RenderDevice>,
    pub stages: vk::ShaderStageFlags,
    pub shaders: ArrayVec<(vk::ShaderModule, vk::ShaderStageFlags, CString), MAX_SHADERS>,
    pub pipeline_layout: vk::PipelineLayout,
    pub layouts: ArrayVec<vk::DescriptorSetLayout, MAX_DESCRIPTOR_SETS>,
    pub desc: ArrayVec<DescriptorSetLayoutDesc, MAX_DESCRIPTOR_SETS>,
}

impl Program {
    pub fn new(device: &Arc<RenderDevice>, shaders: &[ShaderDesc]) -> Result<Self, Error> {
        let mut stages = vk::ShaderStageFlags::empty();
        let mut layouts = Vec::new();
        for shader in shaders {
            let descriptor_layout = Self::reflect(shader.code)?;
            layouts.push(descriptor_layout);
            stages |= shader.stage;
        }
        let mut desc = merge_reflected_layouts(layouts.iter())
            .into_iter()
            .map(|(slot, data)| (slot, data.into()))
            .collect::<Vec<_>>();
        desc.sort_by(|a, b| a.0.cmp(&b.0));
        let mut modules = ArrayVec::<_, MAX_SHADERS>::new();
        for shader in shaders {
            stages |= shader.stage;
            modules.push(Self::create_shader(&device.raw, shader, shader.entry)?);
        }
        let mut layouts = ArrayVec::<_, MAX_DESCRIPTOR_SETS>::new();
        for (_, info) in &desc {
            layouts.push(device.get_or_create_layout(stages, info)?);
        }
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&create_info, None) }?;
        Ok(Self {
            device: device.clone(),
            stages,
            shaders: modules,
            pipeline_layout,
            layouts,
            desc: desc.into_iter().map(|x| x.1).collect(),
        })
    }

    fn create_shader(
        device: &ash::Device,
        desc: &ShaderDesc,
        entry: &str,
    ) -> Result<(vk::ShaderModule, vk::ShaderStageFlags, CString), Error> {
        let shader_create_info =
            vk::ShaderModuleCreateInfo::default().code(desc.code.as_slice_of::<u32>().unwrap());

        Ok((
            unsafe { device.create_shader_module(&shader_create_info, None) }?,
            desc.stage,
            CString::new(entry).unwrap(),
        ))
    }

    fn reflect(code: &[u8]) -> Result<ReflectedDescriptorSetLayoutDesc, Error> {
        let reflection = Reflection::new_from_spirv(code)?;
        let descriptor_sets = reflection.get_descriptor_sets()?;
        let mut layout = ReflectedDescriptorSetLayoutDesc::default();
        for (index, set) in descriptor_sets.into_iter() {
            layout.insert(
                index,
                Self::reflect_descriptor(set, index == DYNAMIC_BINDING_SLOT as u32)?,
            );
        }
        Ok(layout)
    }

    fn reflect_descriptor(
        value: BTreeMap<u32, DescriptorInfo>,
        dynamic: bool,
    ) -> Result<ReflectedDescriptorSetDesc, Error> {
        let mut result = ReflectedDescriptorSetDesc::new();
        for (index, info) in value.into_iter() {
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
                BindingCount::One => 1,
                BindingCount::StaticSized(count) => count as u32,
                BindingCount::Unbounded => panic!("Unbounded descriptors aren't supported"),
            };
            result.insert(index, (info.name, ty, count));
        }
        Ok(result)
    }
}

pub fn create_descriptor_layout(
    device: &RenderDevice,
    stage: vk::ShaderStageFlags,
    layout: &DescriptorSetLayoutDesc,
) -> Result<vk::DescriptorSetLayout, Error> {
    let samplers = TempList::new();
    let bindings = layout
        .layout
        .iter()
        .map(|(index, data)| {
            let mut binding = vk::DescriptorSetLayoutBinding::default()
                .binding(*index)
                .descriptor_count(data.count)
                .descriptor_type(data.ty)
                .stage_flags(stage);
            if data.ty == vk::DescriptorType::SAMPLER
                || data.ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            {
                binding = binding.immutable_samplers(samplers.add(vec![
                    device.sampler(get_sampler_desc(&data.name)).unwrap();
                    data.count as _
                ]));
            }
            binding
        })
        .collect::<Vec<_>>();
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    Ok(unsafe {
        device
            .raw
            .create_descriptor_set_layout(&create_info, None)?
    })
}

fn get_sampler_desc(name: &str) -> SamplerDesc {
    if name.ends_with("_pr") {
        SamplerDesc {
            texel_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode: vk::SamplerAddressMode::REPEAT,
            anisotropy_level: 0,
        }
    } else if name.ends_with("_pb") {
        SamplerDesc {
            texel_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            anisotropy_level: 0,
        }
    } else if name.ends_with("_lb") {
        SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            anisotropy_level: 0,
        }
    } else {
        SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode: vk::SamplerAddressMode::REPEAT,
            anisotropy_level: 4,
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        self.shaders.drain(..).for_each(|(shader, _, _)| unsafe {
            self.device.raw.destroy_shader_module(shader, None)
        });
        unsafe {
            self.device
                .raw
                .destroy_pipeline_layout(self.pipeline_layout, None)
        };
    }
}

fn merge_reflected_layouts<'a>(
    layouts: impl Iterator<Item = &'a ReflectedDescriptorSetLayoutDesc>,
) -> ReflectedDescriptorSetLayoutDesc {
    let mut result = ReflectedDescriptorSetLayoutDesc::new();
    layouts.for_each(|x| merge_reflected_layout_set(&mut result, x));
    for i in 0..MAX_DESCRIPTOR_SETS {
        result.entry(i as u32).or_default();
    }
    result
}

fn merge_reflected_layout_set(
    target: &mut ReflectedDescriptorSetLayoutDesc,
    next: &ReflectedDescriptorSetLayoutDesc,
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

#[cfg(test)]
mod test {
    use ash::vk;

    use crate::program::merge_reflected_layouts;

    use super::{ReflectedDescriptorSetDesc, ReflectedDescriptorSetLayoutDesc};

    #[test]
    fn merge_refected_layouts() {
        let mut set1 = ReflectedDescriptorSetDesc::new();
        set1.insert(0, ("shared1".into(), vk::DescriptorType::SAMPLED_IMAGE, 1));
        set1.insert(1, ("shared2".into(), vk::DescriptorType::UNIFORM_BUFFER, 1));
        let mut set2 = ReflectedDescriptorSetDesc::new();
        set2.insert(0, ("set_a".into(), vk::DescriptorType::STORAGE_BUFFER, 1));
        let mut set3 = ReflectedDescriptorSetDesc::new();
        set3.insert(
            1,
            ("set_b".into(), vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1),
        );

        let mut combined = ReflectedDescriptorSetDesc::new();
        combined.insert(0, ("set_a".into(), vk::DescriptorType::STORAGE_BUFFER, 1));
        combined.insert(
            1,
            ("set_b".into(), vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1),
        );

        let mut a = ReflectedDescriptorSetLayoutDesc::new();
        a.insert(0, set1.clone());
        a.insert(2, set2);
        // a.insert(0, )
        let mut b = ReflectedDescriptorSetLayoutDesc::new();
        b.insert(0, set1.clone());
        b.insert(2, set3);
        let merged = merge_reflected_layouts([a, b].iter());
        let rset1 = merged.get(&0).unwrap();
        let rset2 = merged.get(&2).unwrap();
        assert!(merged.get(&1).unwrap().is_empty());
        assert!(merged.get(&3).unwrap().is_empty());
        assert_eq!(rset1, &set1);
        assert_eq!(rset2, &combined);
    }
}
