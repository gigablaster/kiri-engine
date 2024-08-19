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
use parking_lot::Mutex;

use crate::{vulkan::DYNAMIC_BINDING_SLOT, PipelineHandle};

use super::{
    DescriptorSetBuilder, BufferPool, BufferSlice, DescriptorSetHandle, DescriptorSetManager, Error,
    Frame, ImageDesc, ImageHandle, PipelinePool, RenderDevice, RenderPassHandle, RenderPassAttachment,
    MAX_ATTACHMENTS, MAX_DESCRIPTOR_SETS,
};

const MAX_VERTEX_STREAMS: usize = 2;
const MAX_DYNAMIC_OFFSETS: usize = 2;

#[derive(Debug)]
struct DrawState {
    pipeline: PipelineHandle,
    first_index: u32,
    index_count: u32,
    instance_count: u32,
    first_instance: u32,
    vertex_offset: u32,
    streams: [BufferSlice; MAX_VERTEX_STREAMS],
    indices: BufferSlice,
    descriptor_sets: [DescriptorSetHandle; MAX_DESCRIPTOR_SETS],
    dynamic_offsets: [u32; MAX_DYNAMIC_OFFSETS],
}

#[derive(Debug, Default)]
pub struct DrawStreamRecorder {
    current: DrawState,
    mask: u16,
    stream: Vec<u16>,
}

#[derive(Debug)]
pub struct DrawStream {
    stream: Vec<u16>,
}

struct DrawStreamReader<'a> {
    stream: &'a [u16],
    cursor: usize,
}

const PIPELINE_MASK: u16 = 1 << 0;
const VERTEX_STREAM_MASK: u16 = 1 << 1;
const INDEX_STREAM_MASK: u16 = VERTEX_STREAM_MASK << MAX_VERTEX_STREAMS;
const DESCRIPTOR_SET_MASK: u16 = INDEX_STREAM_MASK << 1;
const DYANMIC_OFFSET_MASK: u16 = DESCRIPTOR_SET_MASK << MAX_DESCRIPTOR_SETS;
const FIRST_INDEX_MASK: u16 = DYANMIC_OFFSET_MASK << 1;
const INDEX_COUNT_MASK: u16 = FIRST_INDEX_MASK << 1;
const FIRST_INSTANCE_MASK: u16 = INDEX_COUNT_MASK << 1;
const INSTANCE_COUNT_MASK: u16 = FIRST_INSTANCE_MASK << 1;
const VERTEX_OFFSET_MASK: u16 = INSTANCE_COUNT_MASK << 1;
const ALL_DESCRIPTOR_SETS_MASK: u16 = ((1 << MAX_DESCRIPTOR_SETS) - 1) << 4;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DrawStreamError {
    #[error("Unexpected end of stream")]
    EndOfStream,
    #[error("Invalid pipeline handle {0:?}")]
    InvalidPipelineHandle(PipelineHandle),
}

impl<'a> DrawStreamReader<'a> {
    fn new(stream: &'a [u16]) -> Self {
        Self { stream, cursor: 0 }
    }

    fn read(&mut self) -> Result<u16, DrawStreamError> {
        if self.cursor >= self.stream.len() {
            Err(DrawStreamError::EndOfStream)
        } else {
            let data = self.stream[self.cursor];
            self.cursor += 1;
            Ok(data)
        }
    }

    fn read_u32(&mut self) -> Result<u32, DrawStreamError> {
        let a = self.read()? as u32;
        let b = self.read()? as u32;
        Ok((a << 16) | b)
    }

    fn read_buffer_slice(&mut self) -> Result<BufferSlice, DrawStreamError> {
        let handle = self.read_u32()?.into();
        let offset = self.read_u32()?;
        Ok(BufferSlice(handle, offset))
    }

    fn read_descriptor_set(&mut self) -> Result<DescriptorSetHandle, DrawStreamError> {
        Ok(self.read_u32()?.into())
    }
}

impl DrawStreamRecorder {
    pub fn finish(self) -> DrawStream {
        DrawStream {
            stream: self.stream,
        }
    }

    fn write_u32(&mut self, value: u32) {
        let (first, second) = (((value & 0xffff0000) >> 16) as u16, (value & 0xffff) as u16);
        self.stream.push(first);
        self.stream.push(second);
    }

    fn write_buffer_slice(&mut self, value: BufferSlice) {
        self.write_u32(value.0.into());
        self.write_u32(value.1);
    }

    /// Resets descriptor sets and dynamic offsets
    pub fn pipeline(&mut self, pipeline: PipelineHandle) {
        if self.current.pipeline != pipeline {
            self.mask |= PIPELINE_MASK;
            self.current.pipeline = pipeline;
            for i in 0..MAX_DESCRIPTOR_SETS {
                self.descriptor_set(i, None);
            }
            for i in 0..MAX_DYNAMIC_OFFSETS {
                self.dynamic_offset(i, None);
            }
        }
    }

    pub fn vertex_stream(&mut self, stream: usize, buffer: Option<BufferSlice>) {
        debug_assert!(stream < MAX_VERTEX_STREAMS);
        let buffer = buffer.unwrap_or_default();
        if self.current.streams[stream] != buffer {
            self.mask |= VERTEX_STREAM_MASK << stream;
            self.current.streams[stream] = buffer;
        }
    }

    pub fn indices(&mut self, buffer: BufferSlice) {
        if self.current.indices != buffer {
            self.mask |= INDEX_STREAM_MASK;
            self.current.indices = buffer
        }
    }

    pub fn descriptor_set(&mut self, slot: usize, group: Option<DescriptorSetHandle>) {
        debug_assert!(slot < MAX_DESCRIPTOR_SETS);
        let group = group.unwrap_or_default();
        if self.current.descriptor_sets[slot] != group {
            self.mask |= DESCRIPTOR_SET_MASK << slot;
            self.current.descriptor_sets[slot] = group;
        }
    }

    pub fn dynamic_offset(&mut self, slot: usize, offset: Option<u32>) {
        debug_assert!(slot < MAX_DYNAMIC_OFFSETS);
        let offset = offset.unwrap_or(u32::MAX);
        if self.current.dynamic_offsets[slot] != offset {
            self.mask |= DYANMIC_OFFSET_MASK << slot;
            self.current.dynamic_offsets[slot] = offset;
        }
    }

    pub fn vertex_offset(&mut self, offset: u32) {
        if self.current.vertex_offset != offset {
            self.mask |= VERTEX_OFFSET_MASK;
            self.current.vertex_offset = offset;
        }
    }

    pub fn draw(
        &mut self,
        first_index: u32,
        index_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        if self.current.first_index != first_index {
            self.mask |= FIRST_INDEX_MASK;
            self.current.first_index = first_index;
        }
        if self.current.index_count != index_count {
            self.mask |= INDEX_COUNT_MASK;
            self.current.index_count = index_count;
        }
        if self.current.first_instance != first_instance {
            self.mask |= FIRST_INSTANCE_MASK;
            self.current.first_instance = first_instance;
        }
        if self.current.instance_count != instance_count {
            self.mask |= INSTANCE_COUNT_MASK;
            self.current.instance_count = instance_count;
        }
        self.stream.push(self.mask);
        if self.mask & PIPELINE_MASK == PIPELINE_MASK {
            self.write_u32(self.current.pipeline.into());
        }
        for i in 0..MAX_VERTEX_STREAMS {
            if self.mask & (VERTEX_STREAM_MASK << i) == (VERTEX_STREAM_MASK << i) {
                self.write_buffer_slice(self.current.streams[i]);
            }
        }
        for i in 0..MAX_DESCRIPTOR_SETS {
            if self.mask & (DESCRIPTOR_SET_MASK << i) == (DESCRIPTOR_SET_MASK << i) {
                self.write_u32(self.current.descriptor_sets[i].into());
            }
        }
        for i in 0..MAX_DYNAMIC_OFFSETS {
            if self.mask & (DYANMIC_OFFSET_MASK << i) == (DYANMIC_OFFSET_MASK << i) {
                self.write_u32(self.current.dynamic_offsets[i]);
            }
        }
        if self.mask & INDEX_STREAM_MASK == INDEX_STREAM_MASK {
            self.write_buffer_slice(self.current.indices);
        }
        if self.mask & FIRST_INDEX_MASK == FIRST_INDEX_MASK {
            self.write_u32(self.current.first_index);
        }
        if self.mask & INDEX_COUNT_MASK == INDEX_COUNT_MASK {
            self.write_u32(self.current.index_count);
        }
        if self.mask & FIRST_INSTANCE_MASK == FIRST_INSTANCE_MASK {
            self.write_u32(self.current.first_instance);
        }
        if self.mask & INSTANCE_COUNT_MASK == INSTANCE_COUNT_MASK {
            self.write_u32(self.current.instance_count);
        }
        if self.mask & VERTEX_OFFSET_MASK == VERTEX_OFFSET_MASK {
            self.write_u32(self.current.vertex_offset);
        }
    }
}

impl Default for DrawState {
    fn default() -> Self {
        Self {
            pipeline: Default::default(),
            first_index: u32::MAX,
            index_count: u32::MAX,
            instance_count: u32::MAX,
            first_instance: u32::MAX,
            vertex_offset: 0,
            streams: Default::default(),
            indices: Default::default(),
            descriptor_sets: Default::default(),
            dynamic_offsets: [0, 0],
        }
    }
}

pub(super) struct DrawStreamExecuteContext<'a> {
    pub device: &'a ash::Device,
    pub frame: &'a Frame,
    pub pipelines: &'a PipelinePool,
    pub descriptor_manager: &'a DescriptorSetManager,
    pub buffers: &'a BufferPool,
    pub empty: DescriptorSetHandle,
    pub fbo: vk::Framebuffer,
    pub pass: vk::RenderPass,
    pub subpass: u32,
    pub render_area: Rect2D,
}

impl DrawStream {
    pub(super) fn execute(
        &self,
        context: DrawStreamExecuteContext,
    ) -> Result<vk::CommandBuffer, Error> {
        puffin::profile_function!();
        let cb = context.frame.secondary_buffer(context.device)?;
        let inheritence = vk::CommandBufferInheritanceInfo::default()
            .framebuffer(context.fbo)
            .render_pass(context.pass)
            .subpass(context.subpass);

        let begin_info = vk::CommandBufferBeginInfo::default()
            .inheritance_info(&inheritence)
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE);

        unsafe {
            context.device.begin_command_buffer(cb, &begin_info)?;
            context.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport::default()
                    .width(context.render_area.extent.width as _)
                    .height(context.render_area.extent.height as _)
                    .max_depth(0.0)
                    .max_depth(1.0)],
            );
            context
                .device
                .cmd_set_scissor(cb, 0, &[context.render_area]);
        }
        let mut reader = DrawStreamReader::new(&self.stream);
        let mut first_index = 0;
        let mut index_count = 0;
        let mut first_instance = 0;
        let mut instance_count = 0;
        let mut pipeline_layout = vk::PipelineLayout::null();
        let mut dynamic_offsets = [u32::MAX; MAX_DYNAMIC_OFFSETS];
        let mut descriptor_sets = [DescriptorSetHandle::invalid(); MAX_DESCRIPTOR_SETS];
        let mut vertex_offset = 0;
        let mut dynamic_offset_changed = false;
        let mut rebind_all = false;

        while let Ok(mask) = reader.read() {
            if mask & PIPELINE_MASK == PIPELINE_MASK {
                let handle = reader.read_u32()?.into();
                let (pipeline, layout) = *context
                    .pipelines
                    .get(handle)
                    .ok_or(DrawStreamError::InvalidPipelineHandle(handle))?;
                pipeline_layout = layout;
                unsafe {
                    context
                        .device
                        .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                }
                rebind_all = true;
            }
            for i in 0..MAX_VERTEX_STREAMS {
                if mask & (VERTEX_STREAM_MASK << i) == (VERTEX_STREAM_MASK << i) {
                    let buffer = reader.read_buffer_slice()?;
                    let (buffer, offset) = if buffer.0.is_valid() {
                        (
                            context
                                .buffers
                                .get(buffer.0)
                                .copied()
                                .ok_or(Error::InvalidBufferHandle(buffer.0))?,
                            buffer.1,
                        )
                    } else {
                        (vk::Buffer::null(), 0)
                    };
                    unsafe {
                        context.device.cmd_bind_vertex_buffers(
                            cb,
                            i as _,
                            &[buffer],
                            &[offset as u64],
                        )
                    }
                }
            }
            if mask & INDEX_STREAM_MASK == INDEX_STREAM_MASK {
                let buffer = reader.read_buffer_slice()?;
                let (buffer, offset) = (
                    context
                        .buffers
                        .get(buffer.0)
                        .copied()
                        .ok_or(Error::InvalidBufferHandle(buffer.0))?,
                    buffer.1,
                );
                unsafe {
                    context.device.cmd_bind_index_buffer(
                        cb,
                        buffer,
                        offset as _,
                        vk::IndexType::UINT16,
                    )
                }
            }
            rebind_all |= (mask & ALL_DESCRIPTOR_SETS_MASK) == ALL_DESCRIPTOR_SETS_MASK;
            for (i, target) in descriptor_sets
                .iter_mut()
                .enumerate()
                .take(MAX_DESCRIPTOR_SETS - 1)
            {
                if mask & (DESCRIPTOR_SET_MASK << i) == (DESCRIPTOR_SET_MASK << i) {
                    let descriptor_set = reader.read_descriptor_set()?;
                    *target = descriptor_set;
                    if !rebind_all {
                        let ds = context.descriptor_manager.resolve(descriptor_set)?;
                        unsafe {
                            context.device.cmd_bind_descriptor_sets(
                                cb,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline_layout,
                                i as _,
                                &[ds],
                                &[],
                            )
                        };
                    }
                }
            }
            if mask & (DESCRIPTOR_SET_MASK << DYNAMIC_BINDING_SLOT)
                == DESCRIPTOR_SET_MASK << DYNAMIC_BINDING_SLOT
            {
                descriptor_sets[DYNAMIC_BINDING_SLOT] = reader.read_descriptor_set()?;
                dynamic_offsets = [u32::MAX; MAX_DYNAMIC_OFFSETS];
                dynamic_offset_changed = true;
            }
            for (i, target) in dynamic_offsets
                .iter_mut()
                .enumerate()
                .take(MAX_DYNAMIC_OFFSETS)
            {
                if mask & (DYANMIC_OFFSET_MASK << i) == DYANMIC_OFFSET_MASK << i {
                    let offset = reader.read_u32()?;
                    *target = offset;
                    dynamic_offset_changed = true;
                }
            }
            if mask & FIRST_INDEX_MASK == FIRST_INDEX_MASK {
                first_index = reader.read_u32()?;
            }
            if mask & INDEX_COUNT_MASK == INDEX_COUNT_MASK {
                index_count = reader.read_u32()?;
            }
            if mask & FIRST_INSTANCE_MASK == FIRST_INSTANCE_MASK {
                first_instance = reader.read_u32()?;
            }
            if mask & INSTANCE_COUNT_MASK == INSTANCE_COUNT_MASK {
                instance_count = reader.read_u32()?;
            }
            if mask & VERTEX_OFFSET_MASK == VERTEX_OFFSET_MASK {
                vertex_offset = reader.read_u32()?;
            }
            if rebind_all {
                let mut descriptors =
                    [context.descriptor_manager.resolve(context.empty)?; MAX_DESCRIPTOR_SETS];
                for (index, descriptor_set) in descriptor_sets.iter().enumerate() {
                    if descriptor_set.is_valid() {
                        descriptors[index] = context.descriptor_manager.resolve(*descriptor_set)?;
                    }
                }
                let offsets = dynamic_offsets
                    .iter()
                    .filter_map(|x| (*x != u32::MAX).then_some(*x))
                    .collect::<ArrayVec<_, MAX_DYNAMIC_OFFSETS>>();
                unsafe {
                    context.device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &descriptors,
                        &offsets,
                    )
                }
                rebind_all = false;
                dynamic_offset_changed = false;
            }
            if dynamic_offset_changed {
                let descriptor = context
                    .descriptor_manager
                    .resolve(descriptor_sets[DYNAMIC_BINDING_SLOT])?;
                let offsets = dynamic_offsets
                    .iter()
                    .filter_map(|x| (*x != u32::MAX).then_some(*x))
                    .collect::<ArrayVec<_, MAX_DYNAMIC_OFFSETS>>();
                unsafe {
                    context.device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        DYNAMIC_BINDING_SLOT as _,
                        &[descriptor],
                        &offsets,
                    )
                }
                dynamic_offset_changed = false;
            }
            unsafe {
                context.device.cmd_draw_indexed(
                    cb,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset as _,
                    first_instance,
                );
            }
        }
        unsafe { context.device.end_command_buffer(cb) }?;
        Ok(cb)
    }
}

#[derive(Debug)]
pub struct RenderPassRecorder<'a> {
    context: &'a RenderContext<'a>,
    pub targets: &'a [RenderPassAttachment],
    pass: RenderPassHandle,
    subpass: u32,
    streams: Mutex<Vec<DrawStream>>,
}

#[derive(Debug)]
pub(super) struct RecordedRenderPass {
    pub targets: ArrayVec<RenderPassAttachment, MAX_ATTACHMENTS>,
    pub pass: RenderPassHandle,
    pub subpass: u32,
    pub streams: Vec<DrawStream>,
}

#[derive(Debug)]
pub struct RenderContext<'a> {
    pub(super) device: &'a RenderDevice,
    pub(super) _frame: &'a Frame,
    pub(super) passes: Mutex<Vec<RecordedRenderPass>>,
    pub backbuffer: ImageHandle,
    pub backbuffer_desc: ImageDesc,
    pub(super) descriptor_manager: &'a mut DescriptorSetManager,
    pub(super) temp_descriptor_sets: Vec<DescriptorSetHandle>,
}

impl<'a> RenderContext<'a> {
    pub(super) fn finish(self) -> (Vec<RecordedRenderPass>, Vec<DescriptorSetHandle>) {
        (self.passes.into_inner(), self.temp_descriptor_sets)
    }

    pub fn record(
        &'a self,
        pass: RenderPassHandle,
        subpass: u32,
        targets: &'a [RenderPassAttachment],
    ) -> RenderPassRecorder<'a> {
        RenderPassRecorder {
            context: self,
            targets,
            pass,
            subpass,
            streams: Default::default(),
        }
    }

    /// Creates temporary descriptor set, it will be invalidated at the end of the frame
    pub fn create_frame_descriptor_set<'b>(
        &mut self,
        builder: DescriptorSetBuilder<'static, 'b>,
    ) -> Result<DescriptorSetHandle, Error> {
        let handle = self.descriptor_manager.create_descriptor_set(
            &self.device.device,
            &self.device.samplers,
            builder,
        )?;
        self.temp_descriptor_sets.push(handle);
        Ok(handle)
    }
}

impl<'a, 'b> RenderPassRecorder<'a> {
    pub fn record(&self, stream: DrawStreamRecorder) {
        self.streams.lock().push(stream.finish());
    }

    pub fn finish(self) {
        self.context.passes.lock().push(RecordedRenderPass {
            pass: self.pass,
            targets: self
                .targets
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_ATTACHMENTS>>(),
            subpass: self.subpass,
            streams: self.streams.into_inner(),
        });
    }
}

impl RecordedRenderPass {
    pub(super) fn consume(
        self,
    ) -> (
        RenderPassHandle,
        u32,
        Vec<DrawStream>,
        ArrayVec<RenderPassAttachment, MAX_ATTACHMENTS>,
    ) {
        (self.pass, self.subpass, self.streams, self.targets)
    }
}
