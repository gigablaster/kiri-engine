mod buffer;
mod descriptors;
mod drop_list;
mod frame;
mod graphics_device;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod staging;
mod swapchain;

pub use ash;
use ash::vk;
pub use buffer::*;
pub use descriptors::*;
use drop_list::*;
pub use frame::*;
pub use graphics_device::*;
pub use image::*;
pub use instance::*;
pub use physical_device::*;
pub use pipeline::*;
use staging::*;
pub use swapchain::*;

pub(crate) type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
pub(crate) type GpuMemoryBlock = gpu_alloc::MemoryBlock<vk::DeviceMemory>;
pub(crate) type GpuDescriptorAllocator =
    gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>;
pub(crate) type GpuDescriptor = gpu_descriptor::DescriptorSet<vk::DescriptorSet>;
