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

use kiri_backend::ash::vk;
use thiserror::Error;

use crate::{BufferHandle, DescriptorHandle, ImageHandle, PipelineHandle, ProgramHandle};

#[derive(Debug, Error)]
pub enum Error {
    #[error("Backend error {0}")]
    BackendError(#[from] kiri_backend::Error),
    #[error("Slot with name {0} not found")]
    InvalidDescriptorSlotName(String),
    #[error("Slot with index {0} not found")]
    InvalidDescriptorSlotIndex(u32),
    #[error("Image is too big")]
    ImageTooBig,
    #[error("Out of dynamic memory")]
    OutOfDynamicMemory,
    #[error("Invalid image handle {0}")]
    InvalidImageHandle(ImageHandle),
    #[error("Invalid buffer handle {0}")]
    InvalidBufferHandle(BufferHandle),
    #[error("Descriptor binding slot with name {0} not found")]
    BindingSlotNotFound(String),
    #[error("Invalid pipeline handle {0}")]
    InvalidPipelineHandle(PipelineHandle),
    #[error("Invalid descriptor handle {0}")]
    InvalidDescriptorHandle(DescriptorHandle),
    #[error("Invalid program handle {0}")]
    InvalidProgramHandle(ProgramHandle),
}

impl From<vk::Result> for Error {
    fn from(value: vk::Result) -> Self {
        Self::BackendError(value.into())
    }
}
