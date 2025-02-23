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

use ash::vk;
use thiserror::Error;

use crate::vulkan::{
    BufferHandle, DescriptorHandle, ImageHandle, RasterPipelineHandle, SamplerDesc,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("Out of device memory")]
    OutOfDeviceMemory,
    #[error("Out of host memory")]
    OutOfHostMemory,
    #[error("Too many objects")]
    TooManyObjects,
    #[error("Not supported")]
    NotSupported,
    #[error("Vulkan not found or failed to load")]
    VulkanFailedToLoad,
    #[error("Can't find suitable device")]
    NoSuitableDevice,
    #[error("Extension {0} not found")]
    ExtensionNotFound(String),
    #[error("Can't find suitable queue")]
    NoSuitableQueue,
    #[error("Failed to map memory")]
    MemoryMapFailed,
    #[error("No suitable memory type found")]
    NoSuitableMemoryType,
    #[error("Sampler not found: {0:?}")]
    SamplerNotFound(SamplerDesc),
    #[error("Wrong memory type")]
    WrongMemoryType,
    #[error("Memory mapping failed")]
    MapFailed,
    #[error("Memory is already mapped")]
    AlreadyMapped,
    #[error("Image is too big")]
    ImageTooBig,
    #[error("Descriptor pool fragmentation")]
    Fragmentation,
    #[error("Out of pool memory")]
    OutOfPoolMemory,
    #[error("Invalid image handle {0}")]
    InvalidImageHandle(ImageHandle),
    #[error("Invalid buffer handle {0}")]
    InvalidBufferHandle(BufferHandle),
    #[error("Descriptor binding slot with name {0} not found")]
    BindingSlotNotFound(String),
    #[error("Buffer {0} isn't mapped")]
    BufferIsntMapped(BufferHandle),
    #[error("Invalid descriptor handle {0}")]
    InvalidDescriptorHandle(DescriptorHandle),
    #[error("Texture slot {0} not found")]
    TextureSlotNotFound(String),
    #[error("Invalid raster pipeline handle {0:?}")]
    InvalidRasterPipelineHandle(RasterPipelineHandle),
    #[error("Dynamic GPU memory allocation failed")]
    DynamicGpuMemoryAllocationFailed,
}

impl From<vk::Result> for Error {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_FORMAT_NOT_SUPPORTED
            | vk::Result::ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR => Self::NotSupported,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfHostMemory,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Self::OutOfDeviceMemory,
            vk::Result::ERROR_TOO_MANY_OBJECTS => Self::TooManyObjects,
            _ => panic!("Unexpected error {:?}", value),
        }
    }
}

impl From<gpu_descriptor::AllocationError> for Error {
    fn from(value: gpu_descriptor::AllocationError) -> Self {
        match value {
            gpu_descriptor::AllocationError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            gpu_descriptor::AllocationError::OutOfHostMemory => Self::OutOfHostMemory,
            gpu_descriptor::AllocationError::Fragmentation => Self::Fragmentation,
        }
    }
}

impl From<gpu_descriptor::CreatePoolError> for Error {
    fn from(value: gpu_descriptor::CreatePoolError) -> Self {
        match value {
            gpu_descriptor::CreatePoolError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            gpu_descriptor::CreatePoolError::OutOfHostMemory => Self::OutOfHostMemory,
            gpu_descriptor::CreatePoolError::Fragmentation => Self::Fragmentation,
        }
    }
}

impl From<gpu_descriptor::DeviceAllocationError> for Error {
    fn from(value: gpu_descriptor::DeviceAllocationError) -> Self {
        match value {
            gpu_descriptor::DeviceAllocationError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            gpu_descriptor::DeviceAllocationError::OutOfHostMemory => Self::OutOfHostMemory,
            gpu_descriptor::DeviceAllocationError::OutOfPoolMemory => Self::OutOfPoolMemory,
            gpu_descriptor::DeviceAllocationError::FragmentedPool => Self::Fragmentation,
        }
    }
}

impl From<gpu_alloc::AllocationError> for Error {
    fn from(value: gpu_alloc::AllocationError) -> Self {
        match value {
            gpu_alloc::AllocationError::OutOfDeviceMemory => Error::OutOfDeviceMemory,
            gpu_alloc::AllocationError::OutOfHostMemory => Error::OutOfHostMemory,
            gpu_alloc::AllocationError::NoCompatibleMemoryTypes => Error::NoSuitableMemoryType,
            gpu_alloc::AllocationError::TooManyObjects => Error::TooManyObjects,
        }
    }
}

impl From<gpu_alloc::MapError> for Error {
    fn from(value: gpu_alloc::MapError) -> Self {
        match value {
            gpu_alloc::MapError::OutOfDeviceMemory => Error::OutOfDeviceMemory,
            gpu_alloc::MapError::OutOfHostMemory => Error::OutOfHostMemory,
            gpu_alloc::MapError::NonHostVisible => Error::WrongMemoryType,
            gpu_alloc::MapError::MapFailed => Error::MapFailed,
            gpu_alloc::MapError::AlreadyMapped => Error::AlreadyMapped,
        }
    }
}

impl From<ash::LoadingError> for Error {
    fn from(_: ash::LoadingError) -> Self {
        Self::VulkanFailedToLoad
    }
}

impl From<(Vec<vk::Pipeline>, vk::Result)> for Error {
    fn from(value: (Vec<vk::Pipeline>, vk::Result)) -> Self {
        value.1.into()
    }
}
