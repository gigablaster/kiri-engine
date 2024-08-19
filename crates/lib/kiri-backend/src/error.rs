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

use ash::vk;
use thiserror::Error;

use crate::SamplerDesc;

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
    #[error("Memory isn't allocated")]
    MemoryNotAllocated,
    #[error("Sampler not found: {0:?}")]
    SamplerNotFound(SamplerDesc),
    #[error("Shader reflection error: {0}")]
    ShaderReflectionError(rspirv_reflect::ReflectError),
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

impl From<gpu_alloc::AllocationError> for Error {
    fn from(value: gpu_alloc::AllocationError) -> Self {
        match value {
            gpu_alloc::AllocationError::NoCompatibleMemoryTypes => Self::NotSupported,
            gpu_alloc::AllocationError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            gpu_alloc::AllocationError::OutOfHostMemory => Self::OutOfHostMemory,
            gpu_alloc::AllocationError::TooManyObjects => Self::TooManyObjects,
        }
    }
}

impl From<gpu_alloc::MapError> for Error {
    fn from(value: gpu_alloc::MapError) -> Self {
        match value {
            gpu_alloc::MapError::NonHostVisible => Self::NotSupported,
            gpu_alloc::MapError::AlreadyMapped | gpu_alloc::MapError::MapFailed => {
                Self::MemoryMapFailed
            }
            gpu_alloc::MapError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            gpu_alloc::MapError::OutOfHostMemory => Self::OutOfHostMemory,
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

impl From<rspirv_reflect::ReflectError> for Error {
    fn from(value: rspirv_reflect::ReflectError) -> Self {
        Self::ShaderReflectionError(value)
    }
}
