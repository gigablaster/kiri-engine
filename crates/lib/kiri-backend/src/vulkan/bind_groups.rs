// Copyright (C) 2024 gigablaster

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

use core::slice;
use std::{
    collections::HashSet,
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    slice::from_raw_parts,
    sync::Arc,
};

use ash::vk::{self};
use gpu_alloc_ash::AshMemoryDevice;
use gpu_descriptor::DescriptorSetLayoutCreateFlags;
use gpu_descriptor_ash::AshDescriptorDevice;
use kiri_common::{BlockAllocator, TempList};

use crate::{vulkan::ImageViewDesc, ImageAspect};

use super::{
    create_descriptor_set_layout, BindGroupDesc, BindGroupHandle, BindGroupPool, BufferHandle,
    BufferPool, BufferSlice, DescriptorSetLayout, Error, GpuAllocator, GpuDescriptor, GpuMemory,
    ImageHandle, PhysicalDevice, ProgramHandle, RenderDevice,
};

pub(crate) const UNIFORM_BUFFER_SIZE: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct ShaderUniform(u32);

impl ShaderUniform {
    pub fn offset(&self) -> usize {
        self.0 as usize
    }
}

const MAX_UNIFORM_SIZE: u32 = 16384;
const PAGE_SIZE: u32 = 65536;

#[derive(Debug)]
struct UniformPage {
    offset: u32,
    size: u32,
    min: u32,
    max: u32,
    allocator: BlockAllocator,
}

impl UniformPage {
    fn new(offset: u32, block_size: u32, count: u32) -> Self {
        Self {
            offset,
            size: block_size * count,
            min: block_size >> 1, // Assume that block_size is pow2
            max: block_size,
            allocator: BlockAllocator::new(block_size as _, count as _),
        }
    }

    fn belong(&self, offset: u32) -> bool {
        offset >= self.offset && offset < self.offset + self.size
    }

    fn fit(&self, size: u32) -> bool {
        size > self.min && size <= self.max
    }

    fn allocate(&mut self) -> Option<usize> {
        self.allocator
            .allocate()
            .map(|offset| offset + self.offset as usize)
    }

    fn deallocate(&mut self, offset: u32) {
        let offset = offset - self.offset;
        self.allocator.dealloc(offset as _);
    }
}

#[derive(Debug)]
pub struct Uniforms {
    buffer: vk::Buffer,
    memory: Option<GpuMemory>,
    mapping: NonNull<u8>,
    pages: Vec<UniformPage>,
    page_allocator: BlockAllocator,
    min: u32,
}

unsafe impl Send for Uniforms {}
unsafe impl Sync for Uniforms {}

impl Uniforms {
    pub fn new(
        device: &ash::Device,
        size: u32,
        allocator: &mut GpuAllocator,
        pdevice: &PhysicalDevice,
    ) -> Result<Self, Error> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size as _)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);

        let buffer = unsafe { device.create_buffer(&create_info, None) }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut memory = RenderDevice::allocate_impl(
            device,
            allocator,
            requirements,
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS | gpu_alloc::UsageFlags::HOST_ACCESS,
            true,
        )?;
        unsafe { device.bind_buffer_memory(buffer, *memory.memory(), memory.offset()) }?;

        let mapping = unsafe { memory.map(AshMemoryDevice::wrap(device), 0, size as _) }?;
        Ok(Self {
            buffer,
            mapping,
            memory: Some(memory),
            pages: Vec::new(),
            page_allocator: BlockAllocator::new(PAGE_SIZE as _, (size / PAGE_SIZE) as _),
            min: pdevice
                .properties
                .limits
                .min_uniform_buffer_offset_alignment as u32,
        })
    }

    pub fn allocate(&mut self, data: &[u8]) -> Result<ShaderUniform, Error> {
        let size = data.len();
        let offset = self.allocate_impl(size as _)?;
        unsafe { copy_nonoverlapping(data.as_ptr(), self.mapping.as_ptr().add(offset), size) }
        Ok(ShaderUniform(offset as u32))
    }

    pub fn deallocate(&mut self, uniform: ShaderUniform) {
        let offset = uniform.0;
        let page = self
            .pages
            .iter_mut()
            .find(|page| page.belong(offset))
            .unwrap();
        page.deallocate(offset);
    }

    fn allocate_impl(&mut self, size: u32) -> Result<usize, Error> {
        debug_assert!(size <= MAX_UNIFORM_SIZE);
        // Find existing page
        let offset = self.pages.iter_mut().find_map(|page| {
            if page.fit(size) {
                page.allocate()
            } else {
                None
            }
        });
        if let Some(offset) = offset {
            Ok(offset)
        } else {
            // Allocate new page
            let page = self
                .page_allocator
                .allocate()
                .ok_or(Error::OutOfUniformBuffer)? as u32;
            let block_size = size.next_power_of_two().max(self.min);
            let mut page = UniformPage::new(page, block_size, PAGE_SIZE / block_size);
            let offset = page.allocate().ok_or(Error::OutOfUniformBuffer)?;
            self.pages.push(page);
            Ok(offset)
        }
    }

    pub fn free(&mut self, device: &ash::Device, allocator: &mut GpuAllocator) {
        if let Some(memory) = self.memory.take() {
            unsafe {
                allocator.dealloc(AshMemoryDevice::wrap(device), memory);
                device.destroy_buffer(self.buffer, None);
            }
        }
    }
}

#[derive(Debug)]
struct Binding<T: Copy> {
    slot: u32,
    ty: vk::DescriptorType,
    data: Option<T>,
}

impl<T: Copy> Binding<T> {
    pub fn new(slot: u32, ty: vk::DescriptorType) -> Self {
        Self {
            slot,
            ty,
            data: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BufferRef {
    Handle(BufferHandle),
    Real(vk::Buffer, ShaderUniform),
}

impl BufferRef {
    fn resolve(&self, buffers: &BufferPool) -> Result<vk::Buffer, Error> {
        match self {
            BufferRef::Handle(handle) => buffers
                .get(*handle)
                .copied()
                .ok_or(Error::InvalidBufferHandle(*handle)),
            BufferRef::Real(buffer, _) => Ok(*buffer),
        }
    }
}

type ImageBinding = Binding<(ImageHandle, ImageAspect)>;
type UniformBufferBinding = Binding<(BufferRef, u64, u64)>;
type StorageBufferBinding = Binding<(BufferHandle, u64, u64)>;
type DynamicBufferBinding = Binding<(vk::Buffer, u64)>;

#[derive(Debug)]
pub(crate) struct BindGroupData {
    pub set: Option<GpuDescriptor>,
    layout: Arc<DescriptorSetLayout>,
    images: Vec<ImageBinding>,
    uniforms: Vec<UniformBufferBinding>,
    dynamic_uniforms: Vec<DynamicBufferBinding>,
    storages: Vec<StorageBufferBinding>,
    dynamic_storages: Vec<DynamicBufferBinding>,
}

impl RenderDevice {
    pub fn create_bind_group(&self, layout: BindGroupDesc) -> Result<BindGroupHandle, Error> {
        let layout = self.get_or_create_layout(&layout)?;
        self.create_bind_group_from_layout(&layout)
    }

    pub fn create_bind_group_from_program(
        &self,
        handle: ProgramHandle,
        slot: usize,
    ) -> Result<BindGroupHandle, Error> {
        let programs = self.programs.read();
        let program = programs
            .get(handle)
            .ok_or(Error::InvalidProgramHandle(handle))?;
        let layout = program
            .layouts
            .get(slot)
            .ok_or(Error::NoDescriptorSetInProgram(slot, handle))?;
        self.create_bind_group_from_layout(layout)
    }

    fn create_bind_group_from_layout(
        &self,
        layout: &Arc<DescriptorSetLayout>,
    ) -> Result<BindGroupHandle, Error> {
        let images = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::SAMPLED_IMAGE
                    || *ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                {
                    Some(ImageBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let uniforms = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER {
                    Some(UniformBufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let storages = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::STORAGE_BUFFER {
                    Some(StorageBufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let dynamic_uniforms = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                    Some(DynamicBufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let dynamic_storages = layout
            .types
            .iter()
            .filter_map(|(slot, ty)| {
                if *ty == vk::DescriptorType::STORAGE_BUFFER_DYNAMIC {
                    Some(DynamicBufferBinding::new(*slot as _, *ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let data = BindGroupData {
            set: None,
            layout: layout.clone(),
            images,
            uniforms,
            dynamic_uniforms,
            storages,
            dynamic_storages,
        };
        // Value that isn't default so slot will be valid. No need to allocate descriptor for real there, will be destroyed anyway.
        let handle = self
            .bind_groups
            .lock()
            .push(<vk::DescriptorSet as vk::Handle>::from_raw(u64::MAX), data);
        self.dirty_bind_groups.lock().insert(handle);
        Ok(handle)
    }

    pub fn destroy_bind_group(&self, handle: BindGroupHandle) {
        if let Some((_, mut bind_group)) = self.bind_groups.lock().remove(handle) {
            if let Some(descriptor) = bind_group.set.take() {
                self.with_drop_list(|drop_list| {
                    drop_list.drop_descriptor(descriptor);
                });
            }
        }
    }

    pub(crate) fn get_or_create_layout(
        &self,
        layout: &BindGroupDesc,
    ) -> Result<Arc<DescriptorSetLayout>, Error> {
        let mut layouts = self.layouts.lock();
        if let Some(layout) = layouts.get(layout).cloned() {
            Ok(layout)
        } else {
            let result = create_descriptor_set_layout(&self.device, &self.samplers, layout)?;
            layouts.insert(layout.clone(), result.clone());
            Ok(result)
        }
    }

    pub(crate) fn update_descriptors(&self) -> Result<(), Error> {
        puffin::profile_function!();
        let mut bind_groups = self.bind_groups.lock();
        let mut dirty = self.dirty_bind_groups.lock();
        let images = self.images.read();
        let buffers = self.buffers.read();
        let dirty = dirty.drain().collect::<Vec<_>>();
        let mut to_destroy = Vec::new();
        // Remove old descriptors
        dirty.iter().for_each(|handle| {
            if let Some(data) = bind_groups.get_cold_mut(*handle) {
                if let Some(set) = data.set.take() {
                    to_destroy.push(set);
                }
            }
        });
        self.with_drop_list(|drop_list| {
            to_destroy.drain(..).for_each(|x| {
                drop_list.drop_descriptor(x);
            });
        });
        // Allocate new descriptors
        self.with_descriptor_allocator(|allocator| {
            dirty.iter().try_for_each(|handle| -> Result<(), Error> {
                let set = if let Some(data) = bind_groups.get_cold_mut(*handle) {
                    let set = unsafe {
                        allocator.allocate(
                            AshDescriptorDevice::wrap(&self.device),
                            &data.layout.raw,
                            DescriptorSetLayoutCreateFlags::empty(),
                            &data.layout.count,
                            1,
                        )
                    }?
                    .remove(0);
                    let raw = *set.raw();

                    data.set = Some(set);
                    Some(raw)
                } else {
                    None
                };
                if let Some(set) = set {
                    bind_groups.replace(*handle, set);
                }
                Ok(())
            })?;
            Ok(())
        })?;
        // Bind everything
        let mut writes = Vec::with_capacity(8192);
        let image_writes = TempList::new();
        let buffer_writes = TempList::new();
        for handle in &dirty {
            let data = bind_groups
                .get_cold(*handle)
                .ok_or(Error::InvalidBindGroupHandle(*handle))?;
            let descriptor = if let Some(set) = &data.set {
                *set.raw()
            } else {
                unreachable!();
            };
            data.images.iter().try_for_each(|x| -> Result<(), Error> {
                let (handle, aspect) = x
                    .data
                    .ok_or(Error::EmptyBindGroupSlot(*handle, x.slot as usize))?;
                let image = images
                    .get_cold(handle)
                    .ok_or(Error::InvalidImageHandle(handle))?;
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .image_info(slice::from_ref(
                            image_writes.add(
                                vk::DescriptorImageInfo::default()
                                    .image_view(
                                        image.view(&self.device, ImageViewDesc::new(aspect))?,
                                    )
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                            ),
                        ))
                        .descriptor_count(1)
                        .descriptor_type(x.ty)
                        .dst_binding(x.slot)
                        .dst_set(descriptor),
                );
                Ok(())
            })?;
            data.uniforms
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (buffer_ref, offset, range) = x
                        .data
                        .ok_or(Error::EmptyBindGroupSlot(*handle, x.slot as usize))?;
                    let buffer = buffer_ref.resolve(&buffers)?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(offset)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
            data.storages
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (handle, offset, range) = x
                        .data
                        .ok_or(Error::EmptyBindGroupSlot(*handle, x.slot as usize))?;
                    let buffer = buffers
                        .get(handle)
                        .copied()
                        .ok_or(Error::InvalidBufferHandle(handle))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(offset)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
            data.dynamic_uniforms
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (buffer, range) = x
                        .data
                        .ok_or(Error::EmptyBindGroupSlot(*handle, x.slot as usize))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(0)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
            data.dynamic_storages
                .iter()
                .try_for_each(|x| -> Result<(), Error> {
                    let (buffer, range) = x
                        .data
                        .ok_or(Error::EmptyBindGroupSlot(*handle, x.slot as usize))?;
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .buffer_info(slice::from_ref(
                                buffer_writes.add(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer)
                                        .offset(0)
                                        .range(range),
                                ),
                            ))
                            .descriptor_count(1)
                            .descriptor_type(x.ty)
                            .dst_binding(x.slot)
                            .dst_set(descriptor),
                    );
                    Ok(())
                })?;
        }
        // And now update all descriptors
        unsafe {
            puffin::profile_scope!("Update_descriptor_sets");
            self.device.update_descriptor_sets(&writes, &[])
        };
        Ok(())
    }

    pub(crate) fn image_updated(&self, image: ImageHandle) {
        self.bind_groups
            .lock()
            .enumerate()
            .for_each(|(handle, _, data)| {
                if data.images.iter().any(|x| {
                    if let Some((image_handle, _)) = x.data {
                        image_handle == image
                    } else {
                        false
                    }
                }) {
                    self.dirty_bind_groups.lock().insert(handle);
                }
            });
    }

    pub fn update_bind_groups<CB: FnOnce(&mut BindGroupUpdateContext) -> Result<(), Error>>(
        &self,
        cb: CB,
    ) -> Result<(), Error> {
        let mut context = BindGroupUpdateContext {
            bind_groups: &mut self.bind_groups.lock(),
            uniforms: &mut self.uniforms.lock(),
            dirty: &mut self.dirty_bind_groups.lock(),
            retired_uniforms: Default::default(),
            temp: self.temp_buffer,
        };
        cb(&mut context)?;
        self.with_drop_list(|drop_list| {
            context
                .retired_uniforms
                .drain(..)
                .for_each(|x| drop_list.drop_uniform(x));
        });
        Ok(())
    }
}

pub struct BindGroupUpdateContext<'a> {
    bind_groups: &'a mut BindGroupPool,
    uniforms: &'a mut Uniforms,
    dirty: &'a mut HashSet<BindGroupHandle>,
    retired_uniforms: Vec<ShaderUniform>,
    temp: vk::Buffer,
}

impl<'a> BindGroupUpdateContext<'a> {
    pub fn bind_image(
        &mut self,
        handle: BindGroupHandle,
        name: &str,
        image: ImageHandle,
        aspect: ImageAspect,
    ) -> Result<(), Error> {
        let data = self
            .bind_groups
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBindGroupHandle(handle))?;
        let slot = Self::get_slot_by_name(data, name)?;
        let index = Self::get_bind_index(&data.images, slot as _)?;
        data.images[index].data = Some((image, aspect));
        self.dirty.insert(handle);
        Ok(())
    }

    pub fn bind_uniform(
        &mut self,
        handle: BindGroupHandle,
        name: &str,
        buffer: BufferSlice,
        size: usize,
    ) -> Result<(), Error> {
        let data = self
            .bind_groups
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBindGroupHandle(handle))?;
        let slot = Self::get_slot_by_name(data, name)?;
        let index = Self::get_bind_index(&data.uniforms, slot as _)?;
        if let Some((BufferRef::Real(_, uniform), ..)) = data.uniforms[index].data.take() {
            self.retired_uniforms.push(uniform);
        }
        data.uniforms[index].data =
            Some((BufferRef::Handle(buffer.0), buffer.1 as u64, size as u64));
        self.dirty.insert(handle);
        Ok(())
    }

    pub fn push_uniform<T>(
        &mut self,
        handle: BindGroupHandle,
        name: &str,
        uniform_data: &[T],
    ) -> Result<(), Error> {
        let size = mem::size_of_val(uniform_data);
        let data = self
            .bind_groups
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBindGroupHandle(handle))?;
        let slot = Self::get_slot_by_name(data, name)?;
        let index = Self::get_bind_index(&data.uniforms, slot as _)?;
        let uniform = self
            .uniforms
            .allocate(unsafe { from_raw_parts([uniform_data].as_ptr() as *const u8, size) })?;
        if let Some((BufferRef::Real(_, uniform), ..)) = data.uniforms[index].data.take() {
            self.retired_uniforms.push(uniform);
        }
        data.uniforms[index].data = Some((
            BufferRef::Real(self.uniforms.buffer, uniform),
            uniform.offset() as _,
            size as _,
        ));
        self.dirty.insert(handle);
        Ok(())
    }

    pub fn bind_dynamic_uniform(
        &mut self,
        handle: BindGroupHandle,
        name: &str,
        size: usize,
    ) -> Result<(), Error> {
        let data = self
            .bind_groups
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBindGroupHandle(handle))?;
        let slot = Self::get_slot_by_name(data, name)?;
        let index = Self::get_bind_index(&data.dynamic_uniforms, slot as _)?;
        data.dynamic_uniforms[index].data = Some((self.temp, size as u64));
        self.dirty.insert(handle);
        Ok(())
    }

    pub fn bind_storage(
        &mut self,
        handle: BindGroupHandle,
        name: &str,
        buffer: BufferSlice,
        size: usize,
    ) -> Result<(), Error> {
        let data = self
            .bind_groups
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBindGroupHandle(handle))?;
        let slot = Self::get_slot_by_name(data, name)?;
        let index = Self::get_bind_index(&data.storages, slot as _)?;
        data.storages[index].data = Some((buffer.0, buffer.1 as u64, size as u64));
        self.dirty.insert(handle);
        Ok(())
    }

    pub fn bind_dynamic_storage(
        &mut self,
        handle: BindGroupHandle,
        name: &str,
        size: usize,
    ) -> Result<(), Error> {
        let data = self
            .bind_groups
            .get_cold_mut(handle)
            .ok_or(Error::InvalidBindGroupHandle(handle))?;
        let slot = Self::get_slot_by_name(data, name)?;
        let index = Self::get_bind_index(&data.dynamic_storages, slot as _)?;
        data.dynamic_storages[index].data = Some((self.temp, size as u64));
        self.dirty.insert(handle);
        Ok(())
    }

    fn get_slot_by_name(data: &BindGroupData, name: &str) -> Result<usize, Error> {
        data.layout
            .names
            .get(name)
            .copied()
            .ok_or(Error::BindSlotWithNameDoesntExist(name.into()))
    }

    fn get_bind_index<T: Copy>(bindings: &[Binding<T>], slot: u32) -> Result<usize, Error> {
        bindings
            .iter()
            .enumerate()
            .find_map(|(index, x)| if x.slot == slot { Some(index) } else { None })
            .ok_or(Error::BindSlotWithIndexDoesntExist(slot as usize))
    }
}
