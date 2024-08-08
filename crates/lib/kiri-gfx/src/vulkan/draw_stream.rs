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

use std::{io::Read, mem, slice, u32};

use arrayvec::ArrayVec;
use ash::vk::{self, Pipeline, Rect2D};
use parking_lot::Mutex;

use crate::{PipelineHandle, PipelinePool, RenderPass};

use super::{Frame, ImageHandle};

const PUSH_SIZE: usize = 128;

#[derive(Debug)]
struct DrawState {
    pipeline: PipelineHandle,
    first_index: u32,
    index_count: u32,
    instance_count: u32,
    first_instance: u32,
    push: ArrayVec<u8, PUSH_SIZE>,
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
const FIRST_INDEX_MASK: u16 = 1 << 1;
const INDEX_COUNT_MASK: u16 = 1 << 2;
const FIRST_INSTANCE_MASK: u16 = 1 << 3;
const INSTANCE_COUNT_MASK: u16 = 1 << 4;
const PUSH_MASK: u16 = 1 << 5;
const PUSH_DATA_SIZE_SHIFT: u16 = 6;
const PUSH_DATA_SIZE_MASK: u16 = 127 << PUSH_DATA_SIZE_SHIFT;

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

    fn read_array(&mut self, count: u16) -> Result<ArrayVec<u8, PUSH_SIZE>, DrawStreamError> {
        debug_assert!(count % 2 == 0);
        let to_read = count / 2;
        let mut result = ArrayVec::new();
        for _ in 0..to_read {
            let word = self.read()?;
            let a = (word & 0xff00) >> 8;
            let b = word & 0xff;
            result.push(a as u8);
            result.push(b as u8);
        }
        Ok(result)
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

    fn write_bytes(&mut self, value: &[u8]) {
        debug_assert!(value.len() % 2 == 0);
        let count = value.len() / 2;
        for i in 0..count {
            let a = value[i * 2] as u16;
            let b = value[i * 2 + 1] as u16;
            self.stream.push(a | b << 8);
        }
    }

    fn is_empty(&self) -> bool {
        self.stream.is_empty()
    }

    pub fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        if self.current.pipeline != pipeline {
            self.mask |= PIPELINE_MASK;
            self.current.pipeline = pipeline;
        }
    }

    pub fn push_data<T: Copy + Sized>(&mut self, data: T) {
        let bytes = [data].as_ptr() as *const u8;
        let data = unsafe { slice::from_raw_parts(bytes, mem::size_of::<T>()) };
        debug_assert!(data.len() <= PUSH_SIZE);
        unsafe { self.current.push.set_len(data.len()) };
        self.current.push.copy_from_slice(data);
        self.mask |= PUSH_MASK;
        self.mask |= (data.len() as u16) << PUSH_DATA_SIZE_SHIFT;
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
            self.write_u32(self.current.pipeline.0);
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
        if self.mask & PUSH_MASK == PUSH_MASK {
            self.write_bytes(&self.current.push.clone());
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
            push: Default::default(),
        }
    }
}

pub(crate) struct DrawStreamExecuteContext<'a> {
    pub device: &'a ash::Device,
    pub cb: vk::CommandBuffer,
    pub pipelines: &'a PipelinePool,
    pub descriptors: &'a [vk::DescriptorSet],
}

impl DrawStream {
    pub(crate) fn execute(&self, context: DrawStreamExecuteContext) -> Result<(), DrawStreamError> {
        puffin::profile_function!();
        let mut reader = DrawStreamReader::new(&self.stream);
        let mut first_index = u32::MAX;
        let mut index_count = u32::MAX;
        let mut first_instance = u32::MAX;
        let mut instance_count = u32::MAX;
        let mut pipeline_layout = vk::PipelineLayout::null();
        while let Ok(mask) = reader.read() {
            if mask & PIPELINE_MASK == PIPELINE_MASK {
                let index = reader.read_u32()?;
                let (pipeline, layout) = *context.pipelines.get(index as usize).ok_or(
                    DrawStreamError::InvalidPipelineHandle(PipelineHandle(index)),
                )?;
                pipeline_layout = layout;
                unsafe {
                    context.device.cmd_bind_pipeline(
                        context.cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                    context.device.cmd_bind_descriptor_sets(
                        context.cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
                        0,
                        context.descriptors,
                        &[],
                    );
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
            if mask & PUSH_MASK == PUSH_MASK {
                let count = (mask & PUSH_DATA_SIZE_MASK) >> PUSH_DATA_SIZE_SHIFT;
                let data = reader.read_array(count)?;
                unsafe {
                    context.device.cmd_push_constants(
                        context.cb,
                        pipeline_layout,
                        vk::ShaderStageFlags::ALL_GRAPHICS,
                        0,
                        &data,
                    )
                }
            }
            unsafe {
                context.device.cmd_draw_indexed(
                    context.cb,
                    index_count,
                    instance_count,
                    first_index,
                    0,
                    first_instance,
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct RenderPassRecorder<'a> {
    context: &'a FrameRecorder<'a>,
    pass: RenderPass,
    streams: Mutex<Vec<DrawStream>>,
}

#[derive(Debug)]
pub(crate) struct RecorderRenderPass {
    pub pass: RenderPass,
    pub streams: Mutex<Vec<DrawStream>>,
}

#[derive(Debug)]
pub struct FrameRecorder<'a> {
    pub(crate) frame: &'a Frame,
    pub(crate) passes: Mutex<Vec<RecorderRenderPass>>,
    pub backbuffer: ImageHandle,
}

impl<'a> FrameRecorder<'a> {
    pub(crate) fn finish(self) -> Vec<RecorderRenderPass> {
        self.passes.into_inner()
    }

    pub fn record(&'a self, pass: RenderPass) -> RenderPassRecorder<'a> {
        RenderPassRecorder {
            context: &self,
            pass: pass,
            streams: Default::default(),
        }
    }
}

impl<'a> RenderPassRecorder<'a> {
    pub fn push(&self, stream: DrawStreamRecorder) {
        self.streams.lock().push(stream.finish());
    }

    pub fn finish(self) {
        self.context.passes.lock().push(RecorderRenderPass {
            pass: self.pass,
            streams: self.streams,
        });
    }
}

impl RecorderRenderPass {
    pub(crate) fn consume(self) -> (RenderPass, Vec<DrawStream>) {
        (self.pass, self.streams.into_inner())
    }
}
