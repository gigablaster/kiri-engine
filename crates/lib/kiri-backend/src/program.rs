// Copyright (C) 2023-2025 gigablaster

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

use std::{collections::HashMap, ffi::CString, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use byte_slice_cast::AsSliceOf;
use kiri_common::TempList;

use crate::{DescriptorSetCount, Error, SamplerDesc, MAX_RESOURCES};

use super::RenderDevice;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DescriptorSetDesc {
    pub name: String,
    pub ty: vk::DescriptorType,
    pub count: u32,
}

impl DescriptorSetDesc {
    fn new(value: rspirv_reflect::DescriptorInfo) -> Self {
        let count = match value.binding_count {
            rspirv_reflect::BindingCount::One => 1,
            rspirv_reflect::BindingCount::StaticSized(count) => count as u32,
            rspirv_reflect::BindingCount::Unbounded => MAX_RESOURCES,
        };
        match value.ty {
            rspirv_reflect::DescriptorType::SAMPLED_IMAGE => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                count,
            },
            rspirv_reflect::DescriptorType::STORAGE_IMAGE => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::STORAGE_IMAGE,
                count,
            },
            rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                count,
            },
            rspirv_reflect::DescriptorType::STORAGE_BUFFER => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count,
            },
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER_DYNAMIC => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                count,
            },
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count,
            },
            rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count,
            },
            rspirv_reflect::DescriptorType::SAMPLER => DescriptorSetDesc {
                name: value.name,
                ty: vk::DescriptorType::SAMPLER,
                count,
            },
            other => panic!("Descriptor set type {:?} isn't supported", other),
        }
    }
}

type ReflectedDescriptorSet = HashMap<u32, DescriptorSetDesc>;

#[derive(Debug, Clone, Default)]
struct ReflectedDescriptorLayout {
    layout: HashMap<u32, ReflectedDescriptorSet>,
    push_constant_range: Option<(u32, u32)>,
    compute_groups_size: Option<(u32, u32, u32)>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct DescriptorSetLayoutDesc {
    pub layout: Vec<(u32, DescriptorSetDesc)>,
    pub push_constant_size: Option<(u32, u32)>,
    pub compute_groups_size: Option<(u32, u32, u32)>,
    pub update_after_bind: bool,
}

impl DescriptorSetLayoutDesc {
    pub fn get_descriptor_count(&self) -> DescriptorSetCount {
        let mut count = DescriptorSetCount::default();
        for (_, data) in self.layout.iter() {
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

    pub fn has_slot(&self, index: u32) -> bool {
        self.layout.iter().any(|(x, _)| *x == index)
    }

    pub fn get_slot(&self, name: &str) -> Option<u32> {
        self.layout
            .iter()
            .find_map(|(slot, desc)| (desc.name == name).then_some(*slot))
    }

    pub fn get_desc(&self, slot: u32) -> Option<&DescriptorSetDesc> {
        self.layout
            .iter()
            .find_map(|(x, data)| if slot == *x { Some(data) } else { None })
    }

    pub fn get_layout(&self) -> &[(u32, DescriptorSetDesc)] {
        &self.layout
    }
}

const MAX_SHADERS: usize = 2;

pub trait Program {
    fn pipeline_layout(&self) -> vk::PipelineLayout;
    fn descritpor_set_layouts(&self) -> &[vk::DescriptorSetLayout];
    fn shader_stages(&self) -> vk::ShaderStageFlags;
}

#[derive(Debug)]
pub struct RasterProgram {
    pub(super) device: Arc<RenderDevice>,
    pub stages: vk::ShaderStageFlags,
    pub shaders: ArrayVec<(vk::ShaderModule, vk::ShaderStageFlags, CString), MAX_SHADERS>,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_layouts: ArrayVec<vk::DescriptorSetLayout, MAX_DESCRIPTOR_SETS>,
    pub layout: Vec<DescriptorSetLayoutDesc>,
}

impl Program for RasterProgram {
    fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    fn descritpor_set_layouts(&self) -> &[vk::DescriptorSetLayout] {
        &self.descriptor_layouts
    }

    fn shader_stages(&self) -> vk::ShaderStageFlags {
        self.stages
    }
}

impl RasterProgram {
    pub fn new(device: &Arc<RenderDevice>, shaders: &[ShaderDesc]) -> Result<Self, Error> {
        let mut stages = vk::ShaderStageFlags::empty();
        for shader in shaders {
            stages |= shader.stage;
        }
        let layout = reflect(shaders)?;

        let mut modules = ArrayVec::<_, MAX_SHADERS>::new();
        for shader in shaders {
            stages |= shader.stage;
            modules.push(create_shader(&device.raw, shader, shader.entry)?);
        }
        let mut layouts = ArrayVec::<_, MAX_DESCRIPTOR_SETS>::new();
        for info in layout.iter() {
            layouts.push(device.get_or_create_layout(stages, info)?);
        }
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&create_info, None) }?;
        Ok(Self {
            device: device.clone(),
            stages,
            shaders: modules,
            pipeline_layout,
            descriptor_layouts: layouts,
            layout,
        })
    }
}

fn reflect(shaders: &[ShaderDesc]) -> Result<Vec<DescriptorSetLayoutDesc>, Error> {
    let mut layouts = Vec::new();
    for shader in shaders {
        layouts.push(reflect_shader(shader)?);
    }
    let layout = merge_reflected_layouts(layouts);
    let layout = layout
        .layout
        .iter()
        .map(|(_, descriptor_set)| DescriptorSetLayoutDesc {
            layout: descriptor_set
                .iter()
                .map(|(index, descriptor)| (*index, descriptor.clone()))
                .collect(),
            update_after_bind: false,
            push_constant_size: layout
                .push_constant_range
                .map(|(start, end)| (start, end - start)),
            compute_groups_size: layout.compute_groups_size,
        })
        .collect::<Vec<_>>();
    Ok(layout)
}

fn reflect_shader(shader: &ShaderDesc) -> Result<ReflectedDescriptorLayout, Error> {
    let reflection = rspirv_reflect::Reflection::new_from_spirv(&shader.code)?;
    let descriptor_sets = reflection.get_descriptor_sets()?;
    let mut layout = HashMap::new();
    for (set_index, set) in descriptor_sets {
        let mut descriptor_set = HashMap::new();
        for (index, bind) in set {
            descriptor_set.insert(index, DescriptorSetDesc::new(bind));
        }
        layout.insert(set_index, descriptor_set);
    }
    Ok(ReflectedDescriptorLayout {
        layout,
        push_constant_range: reflection
            .get_push_constant_range()?
            .map(|x| (x.offset, x.size)),
        compute_groups_size: reflection.get_compute_group_size(),
    })
}

fn merge_reflected_layouts(layouts: Vec<ReflectedDescriptorLayout>) -> ReflectedDescriptorLayout {
    let mut result = ReflectedDescriptorLayout::default();
    for layout in layouts {
        for (stage_index, descriptor_set) in layout.layout {
            result
                .layout
                .entry(stage_index)
                .and_modify(|entry| {
                    for (set_index, set) in descriptor_set.iter() {
                        entry.insert(*set_index, set.clone());
                    }
                })
                .or_insert(descriptor_set.clone());
        }
        if let Some((offset, size)) = layout.push_constant_range {
            let (current_start, current_end) = result.push_constant_range.unwrap_or_default();
            result.push_constant_range =
                Some((current_start.min(offset), current_end.max(offset + size)))
        }
        result.compute_groups_size = layout.compute_groups_size;
    }

    if let Some(max) = result.layout.iter().map(|(set_index, _)| *set_index).max() {
        for i in 0..max {
            result.layout.entry(i).or_default();
        }
    }
    result
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
    let layout = unsafe {
        device
            .raw
            .create_descriptor_set_layout(&create_info, None)?
    };
    device.set_object_name(layout, format!("{:?}", layout));
    Ok(layout)
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

impl Drop for RasterProgram {
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
