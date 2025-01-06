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

use kiri_backend::{ash::vk, Image};
use parking_lot::{Mutex, RwLock};

use crate::{
    BufferHandle, BufferSlice, DescriptorHandle, DescriptorPool, DescriptorSetBuilder,
    DynamicGpuMemory, DynamicWriter, Error, PassDispatcher, Renderer,
};

pub struct RenderContext<'a> {
    renderer: &'a Renderer,
    dynamic: &'a DynamicGpuMemory,
    passes: Mutex<Vec<Box<dyn PassDispatcher>>>,
    descriptors: &'a RwLock<DescriptorPool>,
    trash_descriptors: Mutex<Vec<DescriptorHandle>>,
    pub backbuffer: &'a Image,
}

unsafe impl<'a> Sync for RenderContext<'a> {}
unsafe impl<'a> Send for RenderContext<'a> {}

impl<'a> RenderContext<'a> {
    pub(crate) fn new(
        renderer: &'a Renderer,
        dynamic: &'a DynamicGpuMemory,
        descriptors: &'a RwLock<DescriptorPool>,
        backbuffer: &'a Image,
    ) -> Self {
        Self {
            renderer,
            dynamic,
            passes: Default::default(),
            descriptors,
            trash_descriptors: Default::default(),
            backbuffer,
        }
    }

    pub fn push_dynamic_data<T: Copy>(&self, data: &[T]) -> Result<BufferSlice, Error> {
        self.dynamic
            .push(&self.renderer.device.physical_device, data)
    }

    pub fn write_dynamic_data<T: Copy>(&self, count: usize) -> Result<DynamicWriter<T>, Error> {
        self.dynamic
            .write(&self.renderer.device.physical_device, count)
    }

    pub fn get_temprary_buffer(&self) -> BufferHandle {
        self.dynamic.get_buffer_handle()
    }

    pub fn get_descriptor_set(
        &self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let handle = self.descriptors.write().push(
            vk::DescriptorSet::null(),
            builder.build(&self.renderer.device)?,
        );
        self.trash_descriptors.lock().push(handle);
        Ok(handle)
    }

    pub fn submit(&self, pass: Box<dyn PassDispatcher>) {
        self.passes.lock().push(pass);
    }

    pub(super) fn consume(self) -> (Vec<DescriptorHandle>, Vec<Box<dyn PassDispatcher>>) {
        (
            self.trash_descriptors.into_inner(),
            self.passes.into_inner(),
        )
    }
}
