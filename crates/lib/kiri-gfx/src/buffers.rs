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

use std::sync::Arc;

use ash::vk;
use kiri_backend::{Buffer, BufferCreateDesc, RenderDevice};
use kiri_common::{Handle, HotColdPool};

use crate::Error;

pub type BufferHandle = Handle<vk::DeviceAddress>;
type BufferPool = HotColdPool<vk::DeviceAddress, Arc<Buffer>>;

#[derive(Debug)]
pub struct BufferManager {
    device: Arc<RenderDevice>,
    pool: BufferPool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlice {
    pub buffer: BufferHandle,
    pub offset: u32,
}

impl BufferSlice {
    pub fn new(buffer: BufferHandle, offset: usize) -> Self {
        Self {
            buffer,
            offset: offset as u32,
        }
    }
}

/// Simple buffer manager
///
/// So we can carry around simple 32-bit handles instead of atmocis.
impl BufferManager {
    pub fn new(device: &Arc<RenderDevice>) -> Self {
        Self {
            device: device.clone(),
            pool: Default::default(),
        }
    }

    pub fn create(&mut self, size: usize) -> Result<BufferHandle, Error> {
        let buffer = Buffer::new(&self.device, BufferCreateDesc::gpu(size).storage_buffer())?;
        Ok(self.pool.push(buffer.device_address(), buffer.into()))
    }

    pub fn register(&mut self, buffer: Arc<Buffer>) -> BufferHandle {
        assert!(buffer
            .desc
            .usage
            .contains(vk::BufferUsageFlags::STORAGE_BUFFER));
        self.pool.push(buffer.device_address(), buffer)
    }

    pub fn remove(&mut self, handle: BufferHandle) {
        self.pool.remove(handle);
    }

    pub fn resolve_address(&self, handle: BufferHandle) -> Result<vk::DeviceAddress, Error> {
        self.pool
            .get(handle)
            .copied()
            .ok_or(Error::InvaludBufferHandle(handle))
    }

    pub fn resolve(&self, handle: BufferHandle) -> Result<Arc<Buffer>, Error> {
        self.pool
            .get_cold(handle)
            .cloned()
            .ok_or(Error::InvaludBufferHandle(handle))
    }
}
