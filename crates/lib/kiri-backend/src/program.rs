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
use ash::vk::{self, PushConstantRange};
use byte_slice_cast::AsSliceOf;
use kiri_common::{DefaultPoolLimits, PoolLimits, TempList};
use rspirv_reflect::{BindingCount, DescriptorInfo, Reflection};

use crate::{Error, SamplerDesc};

use super::RenderDevice;

pub const BINDLESS_BINDING_SLOT: usize = 0;
pub const SAMPLERS_BINDING_SLOT: usize = 1;
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

pub type DescriptorSetDesc = HashMap<usize, (String, vk::DescriptorType, usize)>;
type RelfectedDescriptorSetLayout = HashMap<usize, DescriptorSetDesc>;

const MAX_SHADERS: usize = 2;

#[derive(Debug)]
pub struct Program {
    device: Arc<RenderDevice>,
    pub stages: vk::ShaderStageFlags,
    pub push_range: PushConstantRange,
    pub shaders: ArrayVec<(vk::ShaderModule, vk::ShaderStageFlags, CString), MAX_SHADERS>,
    pub pipeline_layout: vk::PipelineLayout,
    pub layouts: ArrayVec<vk::DescriptorSetLayout, MAX_DESCRIPTOR_SETS>,
}

impl Program {
    pub fn new(device: &Arc<RenderDevice>, shaders: &[ShaderDesc]) -> Result<Self, Error> {
        let mut stages = vk::ShaderStageFlags::empty();
        let mut layouts = Vec::new();
        let mut push_ranges = Vec::new();
        for shader in shaders {
            let (descriptor_layout, push_range, _) = Self::reflect(shader.code)?;
            layouts.push(descriptor_layout);
            push_ranges.push(push_range.stage_flags(shader.stage));
            stages |= shader.stage;
        }
        let mut layout = merge_reflected_layouts(layouts.iter())
            .into_iter()
            .collect::<Vec<_>>();
        layout.sort_by(|a, b| a.0.cmp(&b.0));
        let mut modules = ArrayVec::<_, MAX_SHADERS>::new();
        for shader in shaders {
            stages |= shader.stage;
            modules.push(Self::create_shader(&device.raw, shader, shader.entry)?);
        }
        let mut layouts = ArrayVec::<_, MAX_DESCRIPTOR_SETS>::new();
        for (slot, info) in layout {
            layouts.push(create_descriptor_layout(device, stages, &info, slot == 0)?);
        }
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&create_info, None) }?;
        Ok(Self {
            device: device.clone(),
            stages,
            push_range: vk::PushConstantRange::default().size(
                push_ranges
                    .iter()
                    .map(|x| x.offset + x.size)
                    .max()
                    .unwrap_or_default(),
            ),
            shaders: modules,
            pipeline_layout,
            layouts,
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

    fn reflect(
        code: &[u8],
    ) -> Result<
        (
            RelfectedDescriptorSetLayout,
            vk::PushConstantRange,
            (u32, u32, u32),
        ),
        Error,
    > {
        let reflection = Reflection::new_from_spirv(code)?;
        let descriptor_sets = reflection.get_descriptor_sets()?;
        let push_range = reflection
            .get_push_constant_range()?
            .map(|x| {
                vk::PushConstantRange::default()
                    .offset(x.offset)
                    .size(x.size)
            })
            .unwrap_or_default();
        let group_size = reflection.get_compute_group_size().unwrap_or_default();
        let mut layout = RelfectedDescriptorSetLayout::default();
        for (index, set) in descriptor_sets.into_iter() {
            layout.insert(
                index as usize,
                Self::reflect_descriptor(set, index == DYNAMIC_BINDING_SLOT as u32)?,
            );
        }
        Ok((layout, push_range, group_size))
    }

    fn reflect_descriptor(
        value: BTreeMap<u32, DescriptorInfo>,
        dynamic: bool,
    ) -> Result<DescriptorSetDesc, Error> {
        let mut result = DescriptorSetDesc::new();
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
                BindingCount::StaticSized(count) => count,
                BindingCount::Unbounded => DefaultPoolLimits::max_index() as usize,
            };
            result.insert(index as usize, (info.name, ty, count));
        }
        Ok(result)
    }
}

pub fn create_descriptor_layout(
    device: &RenderDevice,
    stage: vk::ShaderStageFlags,
    layout: &DescriptorSetDesc,
    bindless: bool,
) -> Result<vk::DescriptorSetLayout, Error> {
    let flags = if bindless {
        vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
    } else {
        vk::DescriptorSetLayoutCreateFlags::empty()
    };
    let samplers = TempList::new();
    let bindings = layout
        .iter()
        .map(|(index, (name, ty, count))| {
            let mut binding = vk::DescriptorSetLayoutBinding::default()
                .binding(*index as _)
                .descriptor_count(*count as _)
                .descriptor_type(*ty)
                .stage_flags(stage);
            if *ty == vk::DescriptorType::SAMPLER
                || *ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            {
                binding = binding.immutable_samplers(samplers.add(vec![
                    device.sampler(get_sampler_desc(name)).unwrap();
                    *count
                ]));
            }
            binding
        })
        .collect::<Vec<_>>();
    let mut create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(flags)
        .bindings(&bindings);
    let flags = vec![
        vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
        bindings.len()
    ];
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);

    if bindless {
        create_info = create_info.push_next(&mut binding_flags);
    }
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
    layouts: impl Iterator<Item = &'a RelfectedDescriptorSetLayout>,
) -> RelfectedDescriptorSetLayout {
    let mut result = RelfectedDescriptorSetLayout::new();
    layouts.for_each(|x| merge_reflected_layout_set(&mut result, x));
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

#[cfg(test)]
mod test {
    use ash::vk;

    use crate::program::merge_reflected_layouts;

    use super::{DescriptorSetDesc, RelfectedDescriptorSetLayout};

    #[test]
    fn merge_refected_layouts() {
        let mut set1 = DescriptorSetDesc::new();
        set1.insert(0, ("shared1".into(), vk::DescriptorType::SAMPLED_IMAGE, 1));
        set1.insert(1, ("shared2".into(), vk::DescriptorType::UNIFORM_BUFFER, 1));
        let mut set2 = DescriptorSetDesc::new();
        set2.insert(0, ("set_a".into(), vk::DescriptorType::STORAGE_BUFFER, 1));
        let mut set3 = DescriptorSetDesc::new();
        set3.insert(
            1,
            ("set_b".into(), vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1),
        );

        let mut combined = DescriptorSetDesc::new();
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
        let merged = merge_reflected_layouts([a, b].iter());
        let rset1 = merged.get(&0).unwrap();
        let rset2 = merged.get(&2).unwrap();
        assert!(merged.get(&1).unwrap().is_empty());
        assert!(merged.get(&3).unwrap().is_empty());
        assert_eq!(rset1, &set1);
        assert_eq!(rset2, &combined);
    }
}
