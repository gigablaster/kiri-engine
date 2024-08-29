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

use arrayvec::ArrayVec;
use ash::vk::{self, Rect2D};
use kiri_backend::{Image, MAX_COLOR_ATTACHMENTS};
use parking_lot::{Mutex, RwLock};

use crate::{
    BufferHandle, BufferSlice, DescriptorHandle, DescriptorPool, DescriptorSetBuilder, DrawStream,
    DynamicGpuMemory, DynamicWriter, Error, PassDispatcher, RasterizerPassDispatcher, RenderTarget,
    Renderer,
};

pub struct RasterizerPassBuilder<'a> {
    context: &'a RenderContext<'a>,
    color_targets: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth_target: Option<RenderTarget>,
    streams: Vec<DrawStream>,
    descriptor_sets: Vec<DescriptorHandle>,
    area: Option<Rect2D>,
    name: &'a str,
}

impl<'a> RasterizerPassBuilder<'a> {
    pub fn draw(&mut self, stream: DrawStream) {
        self.streams.push(stream);
    }

    pub fn get_descriptor_set(
        &mut self,
        builder: DescriptorSetBuilder,
    ) -> Result<DescriptorHandle, Error> {
        let handle = self.context.descriptors.write().push(
            vk::DescriptorSet::null(),
            builder.build(&self.context.renderer.device)?,
        );
        self.descriptor_sets.push(handle);
        Ok(handle)
    }

    pub fn build(mut self) -> Box<dyn PassDispatcher> {
        self.context
            .trash_descriptors
            .lock()
            .append(&mut self.descriptor_sets);
        Box::new(RasterizerPassDispatcher::new(
            self.name,
            &self.color_targets,
            self.depth_target,
            self.streams,
            self.area,
        ))
    }
}

pub struct RenderContext<'a> {
    renderer: &'a Renderer,
    dynamic: &'a DynamicGpuMemory,
    passes: Mutex<Vec<Box<dyn PassDispatcher>>>,
    descriptors: &'a RwLock<DescriptorPool>,
    trash_descriptors: Mutex<Vec<DescriptorHandle>>,
    pub backbuffer: &'a Image,
}

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

    pub fn create_rasterizer_pass(
        &'a self,
        name: &'a str,
        color: &[RenderTarget],
        depth: Option<RenderTarget>,
        area: Option<Rect2D>,
    ) -> RasterizerPassBuilder {
        RasterizerPassBuilder {
            context: self,
            color_targets: color
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>(),
            depth_target: depth,
            streams: Default::default(),
            descriptor_sets: Default::default(),
            name,
            area,
        }
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
