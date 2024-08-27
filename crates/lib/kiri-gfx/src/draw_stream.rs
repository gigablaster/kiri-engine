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

use std::io::{self, Cursor, Read};

use crate::{BufferPointer, DescriptorHandle, Error, PipelineHandle, RenderResourceResolver};
use arrayvec::ArrayVec;
use ash::vk::{self, Rect2D};
use byteorder::{NativeEndian, ReadBytesExt, WriteBytesExt};
use kiri_backend::{DYNAMIC_BINDING_SLOT, MAX_DESCRIPTOR_SETS};

const MAX_VERTEX_STREAMS: usize = 2;
const MAX_DYNAMIC_OFFSETS: usize = 2;

#[derive(Debug)]
struct DrawState {
    pipeline: PipelineHandle,
    first_index: u32,
    index_count: u32,
    instance_count: u32,
    first_instance: u32,
    vertex_offset: i32,
    streams: [BufferPointer; MAX_VERTEX_STREAMS],
    indices: BufferPointer,
    bind_groups: [DescriptorHandle; MAX_DESCRIPTOR_SETS],
    dynamic_offsets: [u32; MAX_DYNAMIC_OFFSETS],
}

#[derive(Debug)]
pub struct DrawStreamBuilder {
    subpass: u32,
    current: DrawState,
    mask: u16,
    stream: Cursor<Vec<u8>>,
    commands: usize,
}

#[derive(Debug)]
pub struct DrawStream {
    stream: Vec<u8>,
    commands: usize,
    pub subpass: u32,
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

impl DrawStreamBuilder {
    pub fn new(subpass: u32) -> Self {
        Self {
            subpass,
            current: Default::default(),
            mask: Default::default(),
            stream: Default::default(),
            commands: Default::default(),
        }
    }

    pub fn build(self) -> DrawStream {
        DrawStream {
            stream: self.stream.into_inner(),
            commands: self.commands,
            subpass: self.subpass,
        }
    }

    fn write_buffer_pointer(&mut self, value: BufferPointer) {
        self.stream
            .write_u64::<NativeEndian>(value.handle.into())
            .unwrap();
        self.stream.write_u64::<NativeEndian>(value.offset).unwrap()
    }

    /// Resets bind groups and dynamic offsets
    pub fn set_pipeline(&mut self, pipeline: PipelineHandle) {
        if self.current.pipeline != pipeline {
            self.mask |= PIPELINE_MASK;
            self.current.pipeline = pipeline;
            for i in 0..MAX_DESCRIPTOR_SETS {
                self.set_descriptor(i, None);
            }
            for i in 0..MAX_DYNAMIC_OFFSETS {
                self.set_dynamic_offset(i, None);
            }
        }
    }

    pub fn set_vertex_buffer(&mut self, stream: usize, buffer: Option<BufferPointer>) {
        debug_assert!(stream < MAX_VERTEX_STREAMS);
        let buffer = buffer.unwrap_or_default();
        if self.current.streams[stream] != buffer {
            self.mask |= VERTEX_STREAM_MASK << stream;
            self.current.streams[stream] = buffer;
        }
    }

    pub fn set_index_buffer(&mut self, buffer: BufferPointer) {
        if self.current.indices != buffer {
            self.mask |= INDEX_STREAM_MASK;
            self.current.indices = buffer
        }
    }

    pub fn set_descriptor(&mut self, slot: usize, group: Option<DescriptorHandle>) {
        debug_assert!(slot < MAX_DESCRIPTOR_SETS);
        let group = group.unwrap_or_default();
        if self.current.bind_groups[slot] != group {
            self.mask |= DESCRIPTOR_SET_MASK << slot;
            self.current.bind_groups[slot] = group;
        }
    }

    pub fn set_dynamic_offset(&mut self, slot: usize, offset: Option<u32>) {
        debug_assert!(slot < MAX_DYNAMIC_OFFSETS);
        let offset = offset.unwrap_or(u32::MAX);
        if self.current.dynamic_offsets[slot] != offset {
            self.mask |= DYANMIC_OFFSET_MASK << slot;
            self.current.dynamic_offsets[slot] = offset;
        }
    }

    pub fn set_vertex_offset(&mut self, offset: i32) {
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
        self.stream.write_u16::<NativeEndian>(self.mask).unwrap();
        if self.mask & PIPELINE_MASK == PIPELINE_MASK {
            self.stream
                .write_u64::<NativeEndian>(self.current.pipeline.into())
                .unwrap();
        }
        for i in 0..MAX_VERTEX_STREAMS {
            if self.mask & (VERTEX_STREAM_MASK << i) == (VERTEX_STREAM_MASK << i) {
                self.write_buffer_pointer(self.current.streams[i]);
            }
        }
        if self.mask & INDEX_STREAM_MASK == INDEX_STREAM_MASK {
            self.write_buffer_pointer(self.current.indices);
        }
        for i in 0..MAX_DESCRIPTOR_SETS {
            if self.mask & (DESCRIPTOR_SET_MASK << i) == (DESCRIPTOR_SET_MASK << i) {
                self.stream
                    .write_u64::<NativeEndian>(self.current.bind_groups[i].into())
                    .unwrap();
            }
        }
        for i in 0..MAX_DYNAMIC_OFFSETS {
            if self.mask & (DYANMIC_OFFSET_MASK << i) == (DYANMIC_OFFSET_MASK << i) {
                self.stream
                    .write_u32::<NativeEndian>(self.current.dynamic_offsets[i])
                    .unwrap();
            }
        }
        if self.mask & FIRST_INDEX_MASK == FIRST_INDEX_MASK {
            self.stream
                .write_u32::<NativeEndian>(self.current.first_index)
                .unwrap();
        }
        if self.mask & INDEX_COUNT_MASK == INDEX_COUNT_MASK {
            self.stream
                .write_u32::<NativeEndian>(self.current.index_count)
                .unwrap();
        }
        if self.mask & FIRST_INSTANCE_MASK == FIRST_INSTANCE_MASK {
            self.stream
                .write_u32::<NativeEndian>(self.current.first_instance)
                .unwrap();
        }
        if self.mask & INSTANCE_COUNT_MASK == INSTANCE_COUNT_MASK {
            self.stream
                .write_u32::<NativeEndian>(self.current.instance_count)
                .unwrap();
        }
        if self.mask & VERTEX_OFFSET_MASK == VERTEX_OFFSET_MASK {
            self.stream
                .write_i32::<NativeEndian>(self.current.vertex_offset)
                .unwrap();
        }
        self.commands += 1;
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
            bind_groups: Default::default(),
            dynamic_offsets: [0, 0],
        }
    }
}

impl DrawStream {
    fn read_buffer_pointer<R: Read>(mut r: R) -> io::Result<BufferPointer> {
        Ok(BufferPointer {
            handle: r.read_u64::<NativeEndian>()?.into(),
            offset: r.read_u64::<NativeEndian>()?,
        })
    }

    pub(super) fn execute(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        render_area: Rect2D,
        resolver: &RenderResourceResolver,
    ) -> Result<(), Error> {
        unsafe {
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::default()
                    .width(render_area.extent.width as _)
                    .height(render_area.extent.height as _)
                    .max_depth(0.0)
                    .max_depth(1.0)],
            );
            device.cmd_set_scissor(command_buffer, 0, &[render_area]);
        }
        let mut reader = Cursor::new(&self.stream);
        let mut first_index = 0;
        let mut index_count = 0;
        let mut first_instance = 0;
        let mut instance_count = 0;
        let mut pipeline_layout = vk::PipelineLayout::null();
        let mut dynamic_offsets = [u32::MAX; MAX_DYNAMIC_OFFSETS];
        let mut descriptor_sets = [DescriptorHandle::invalid(); MAX_DESCRIPTOR_SETS];
        let mut vertex_offset = 0;
        let mut dynamic_offset_changed = false;
        let mut rebind_all = false;

        for _ in 0..self.commands {
            let mask = reader.read_u16::<NativeEndian>().unwrap();
            if mask & PIPELINE_MASK == PIPELINE_MASK {
                let handle = reader.read_u64::<NativeEndian>().unwrap().into();
                let (pipeline, layout) = resolver.resolve_pipeline(handle)?;
                pipeline_layout = layout;
                unsafe {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                }
                rebind_all = true;
            }
            for i in 0..MAX_VERTEX_STREAMS {
                if mask & (VERTEX_STREAM_MASK << i) == (VERTEX_STREAM_MASK << i) {
                    let buffer = Self::read_buffer_pointer(&mut reader).unwrap();
                    let (buffer, offset) = if buffer.handle.is_valid() {
                        (resolver.resolve_buffer(buffer.handle)?, buffer.offset)
                    } else {
                        (vk::Buffer::null(), 0)
                    };
                    unsafe {
                        device.cmd_bind_vertex_buffers(command_buffer, i as _, &[buffer], &[offset])
                    }
                }
            }
            if mask & INDEX_STREAM_MASK == INDEX_STREAM_MASK {
                let buffer = Self::read_buffer_pointer(&mut reader).unwrap();
                let (buffer, offset) = (resolver.resolve_buffer(buffer.handle)?, buffer.offset);
                unsafe {
                    device.cmd_bind_index_buffer(
                        command_buffer,
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
                    let descriptor = reader.read_u64::<NativeEndian>().unwrap().into();
                    *target = descriptor;
                    if !rebind_all {
                        let ds = resolver.resolve_descriptor_set(descriptor)?;
                        unsafe {
                            device.cmd_bind_descriptor_sets(
                                command_buffer,
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
                descriptor_sets[DYNAMIC_BINDING_SLOT] =
                    reader.read_u64::<NativeEndian>().unwrap().into();
                dynamic_offsets = [u32::MAX; MAX_DYNAMIC_OFFSETS];
                dynamic_offset_changed = true;
            }
            for (i, target) in dynamic_offsets
                .iter_mut()
                .enumerate()
                .take(MAX_DYNAMIC_OFFSETS)
            {
                if mask & (DYANMIC_OFFSET_MASK << i) == DYANMIC_OFFSET_MASK << i {
                    let offset = reader.read_u32::<NativeEndian>().unwrap();
                    *target = offset;
                    dynamic_offset_changed = true;
                }
            }
            if mask & FIRST_INDEX_MASK == FIRST_INDEX_MASK {
                first_index = reader.read_u32::<NativeEndian>().unwrap();
            }
            if mask & INDEX_COUNT_MASK == INDEX_COUNT_MASK {
                index_count = reader.read_u32::<NativeEndian>().unwrap();
            }
            if mask & FIRST_INSTANCE_MASK == FIRST_INSTANCE_MASK {
                first_instance = reader.read_u32::<NativeEndian>().unwrap();
            }
            if mask & INSTANCE_COUNT_MASK == INSTANCE_COUNT_MASK {
                instance_count = reader.read_u32::<NativeEndian>().unwrap();
            }
            if mask & VERTEX_OFFSET_MASK == VERTEX_OFFSET_MASK {
                vertex_offset = reader.read_i32::<NativeEndian>().unwrap();
            }
            if rebind_all {
                let mut descriptors = [resolver.empty_descriptor_set; MAX_DESCRIPTOR_SETS];
                for (index, descriptor_set) in descriptor_sets.iter().enumerate() {
                    if descriptor_set.is_valid() {
                        descriptors[index] = resolver.resolve_descriptor_set(*descriptor_set)?;
                    }
                }
                let offsets = dynamic_offsets
                    .iter()
                    .filter_map(|x| (*x != u32::MAX).then_some(*x))
                    .collect::<ArrayVec<_, MAX_DYNAMIC_OFFSETS>>();
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
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
                let descriptor =
                    resolver.resolve_descriptor_set(descriptor_sets[DYNAMIC_BINDING_SLOT])?;
                let offsets = dynamic_offsets
                    .iter()
                    .filter_map(|x| (*x != u32::MAX).then_some(*x))
                    .collect::<ArrayVec<_, MAX_DYNAMIC_OFFSETS>>();
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
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
                device.cmd_draw_indexed(
                    command_buffer,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset as _,
                    first_instance,
                );
            }
        }
        Ok(())
    }
}
