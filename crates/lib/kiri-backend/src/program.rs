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

use std::{ffi::CString, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use byte_slice_cast::AsSliceOf;
use gpu_descriptor::DescriptorTotalCount;

use crate::Error;

use super::RenderDevice;

pub const DYNAMIC_DESCRIPTOR_SLOT_INDEX: usize = 3;
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
    pub count: usize,
}

impl<'a> DescriptorSetDesc<'a> {
    pub fn single(name: &'a str, ty: vk::DescriptorType) -> Self {
        Self { name, ty, count: 1 }
    }

    pub fn count(name: &'a str, ty: vk::DescriptorType, count: usize) -> Self {
        Self { name, ty, count }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorSetLayoutDesc<'a> {
    pub layout: &'a [(usize, DescriptorSetDesc<'a>)],
    pub push_constant_size: Option<(u32, u32)>,
    pub compute_groups_size: Option<(u32, u32, u32)>,
    pub update_after_bind: bool,
}

impl<'a> DescriptorSetLayoutDesc<'a> {
    pub fn get_descriptor_count(&self) -> DescriptorTotalCount {
        let mut count = DescriptorTotalCount::default();
        for (_, data) in self.layout.iter() {
            match data.ty {
                vk::DescriptorType::SAMPLED_IMAGE => count.sampled_image += data.count as u32,
                vk::DescriptorType::UNIFORM_BUFFER => count.uniform_buffer += data.count as u32,
                vk::DescriptorType::STORAGE_BUFFER => count.storage_buffer += data.count as u32,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                    count.uniform_buffer_dynamic += data.count as u32
                }
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    count.uniform_buffer_dynamic += data.count as u32
                }
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    count.combined_image_sampler += data.count as u32
                }
                vk::DescriptorType::STORAGE_IMAGE => count.storage_image += data.count as u32,
                ty => panic!("Descriptor set type {:?} not supported", ty),
            }
        }
        count
    }

    pub fn has_slot(&self, index: usize) -> bool {
        self.layout.iter().any(|(x, _)| *x == index)
    }

    pub fn get_slot(&self, name: &str) -> Option<usize> {
        self.layout
            .iter()
            .find_map(|(slot, desc)| (desc.name == name).then_some(*slot))
    }

    pub fn get_desc(&self, slot: usize) -> Option<&DescriptorSetDesc> {
        self.layout
            .iter()
            .find_map(|(x, data)| if slot == *x { Some(data) } else { None })
    }

    pub fn get_layout(&self) -> &[(usize, DescriptorSetDesc<'a>)] {
        self.layout
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
    pub layout: &'static [DescriptorSetLayoutDesc<'static>],
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
            modules.push(create_shader(&device.raw, shader, shader.entry)?);
        }
        let mut layouts = ArrayVec::<_, MAX_DESCRIPTOR_SETS>::new();
        for info in layout.iter() {
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
