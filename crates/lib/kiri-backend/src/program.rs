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

use std::{ffi::CString, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use byte_slice_cast::AsSliceOf;
use kiri_common::TempList;

use crate::{DescriptorSetCount, Error, SamplerDesc};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorSetDesc<'a> {
    pub name: &'a str,
    pub ty: vk::DescriptorType,
    pub count: u32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorSetLayoutDesc<'a> {
    pub layout: &'a [(u32, DescriptorSetDesc<'a>)],
    pub update_after_bind: bool,
}

pub const EMPTY_DESCRIPTOR_LAYOUT: DescriptorSetLayoutDesc = DescriptorSetLayoutDesc {
    layout: &[],
    update_after_bind: false,
};

impl<'a> DescriptorSetLayoutDesc<'a> {
    pub fn get_descriptor_count(&self) -> DescriptorSetCount {
        let mut count = DescriptorSetCount::default();
        for (_, data) in self.layout {
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
        self.layout
    }
}

const MAX_SHADERS: usize = 2;

#[derive(Debug)]
pub struct Program {
    device: Arc<RenderDevice>,
    pub stages: vk::ShaderStageFlags,
    pub shaders: ArrayVec<(vk::ShaderModule, vk::ShaderStageFlags, CString), MAX_SHADERS>,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_layouts: ArrayVec<vk::DescriptorSetLayout, MAX_DESCRIPTOR_SETS>,
    pub layout: &'static [DescriptorSetLayoutDesc<'static>],
}

impl Program {
    pub fn new(
        device: &Arc<RenderDevice>,
        layout: &'static [DescriptorSetLayoutDesc<'static>],
        shaders: &[ShaderDesc],
    ) -> Result<Self, Error> {
        let mut stages = vk::ShaderStageFlags::empty();
        for shader in shaders {
            stages |= shader.stage;
        }
        let mut modules = ArrayVec::<_, MAX_SHADERS>::new();
        for shader in shaders {
            stages |= shader.stage;
            modules.push(Self::create_shader(&device.raw, shader, shader.entry)?);
        }
        let mut layouts = ArrayVec::<_, MAX_DESCRIPTOR_SETS>::new();
        for info in layout {
            layouts.push(device.get_or_create_layout(stages, *info)?);
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
}

pub(super) fn create_descriptor_layout(
    device: &RenderDevice,
    stage: vk::ShaderStageFlags,
    layout: DescriptorSetLayoutDesc,
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
                    device.sampler(get_sampler_desc(data.name)).unwrap();
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
