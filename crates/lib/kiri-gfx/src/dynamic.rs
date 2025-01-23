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

use std::{
    marker::PhantomData,
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    sync::Arc,
};

use kiri_backend::{Buffer, BufferCreateDesc, PhysicalDevice, RenderDevice};
use kiri_common::BumpAllocator;

use crate::{BufferHandle, BufferSlice, Error, Renderer};

#[derive(Debug)]
pub struct DynamicGpuMemory {
    buffer_handle: BufferHandle,
    mapping: NonNull<u8>,
    allocator: BumpAllocator,
}

unsafe impl Send for DynamicGpuMemory {}
unsafe impl Sync for DynamicGpuMemory {}

impl DynamicGpuMemory {
    pub fn new(renderer: &Renderer, size: usize) -> Result<Self, Error> {
        let buffer = Buffer::new(
            &renderer.device,
            BufferCreateDesc::shared(size)
                .name("Dynamic data")
                .storage_buffer()
                .device_address()
                .indirect_draw()
                .uniform_buffer(),
        )?;
        let mapping = buffer.mapping.unwrap();
        Ok(Self {
            buffer_handle: renderer.register_buffer(buffer),
            mapping,
            allocator: BumpAllocator::new(size as _),
        })
    }

    pub fn push<T: Copy>(
        &self,
        pdevice: &PhysicalDevice,
        data: &[T],
    ) -> Result<BufferSlice, Error> {
        let size = mem::size_of_val(data);
        if let Some(offset) = self.allocator.allocate(
            size as _,
            pdevice
                .properties
                .limits
                .min_uniform_buffer_offset_alignment as _,
        ) {
            unsafe {
                copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    self.mapping.as_ptr().add(offset as _),
                    size,
                )
            }
            Ok(BufferSlice::new(self.buffer_handle, offset, size))
        } else {
            Err(Error::OutOfDynamicMemory)
        }
    }

    pub fn write<'a, T: Copy>(
        &self,
        pdevice: &PhysicalDevice,
        count: usize,
    ) -> Result<DynamicWriter<'a, T>, Error> {
        let size = mem::size_of::<T>() * count;
        if let Some(offset) = self.allocator.allocate(
            size as _,
            pdevice
                .properties
                .limits
                .min_uniform_buffer_offset_alignment as _,
        ) {
            Ok(DynamicWriter::<T>::new(
                self.mapping.as_ptr(),
                offset,
                count,
            ))
        } else {
            Err(Error::OutOfDynamicMemory)
        }
    }

    pub fn get_buffer_handle(&self) -> BufferHandle {
        self.buffer_handle
    }

    pub fn recycle(&self) {
        self.allocator.reset();
    }
}

#[derive(Debug)]
pub struct DynamicGpuMemoryPool {
    device: Arc<RenderDevice>,
    pool: Vec<Arc<DynamicGpuMemory>>,
    used: Vec<Arc<DynamicGpuMemory>>,
    recycle: Vec<Arc<DynamicGpuMemory>>,
}

const DYNAMIC_PAGE_SIZE: usize = 16 * 1024 * 1024;

impl DynamicGpuMemoryPool {
    pub fn new(device: Arc<RenderDevice>) -> Self {
        Self {
            device,
            pool: Default::default(),
            used: Default::default(),
            recycle: Default::default(),
        }
    }
    pub fn take(&mut self, renderer: &Renderer) -> Result<Arc<DynamicGpuMemory>, Error> {
        if let Some(page) = self.pool.pop() {
            self.used.push(page.clone());
            Ok(page)
        } else {
            let page = Arc::new(DynamicGpuMemory::new(renderer, DYNAMIC_PAGE_SIZE)?);
            self.used.push(page.clone());
            Ok(page)
        }
    }

    pub fn recycle(&mut self) {
        self.pool.append(&mut self.recycle);
        self.recycle.append(&mut self.used);
        self.pool.iter().for_each(|x| x.recycle());
    }
}

#[derive(Debug)]
pub struct DynamicWriter<'a, T: Copy> {
    memory: *mut u8,
    cursor: usize,
    size: usize,
    pub offset: usize,
    _phantom: PhantomData<&'a T>,
}

impl<T: Copy> DynamicWriter<'_, T> {
    fn new(memory: *mut u8, offset: usize, count: usize) -> Self {
        Self {
            memory: unsafe { memory.add(offset as _) },
            cursor: 0,
            size: count * mem::size_of::<T>(),
            offset,
            _phantom: PhantomData,
        }
    }

    pub fn write(&mut self, data: T) -> Result<(), Error> {
        let data_size = mem::size_of::<T>();
        if data_size > (self.size - self.cursor) {
            return Err(Error::OutOfDynamicMemory);
        }
        let src = [data].as_ptr() as *const u8;
        unsafe {
            let dst = self.memory.add(self.cursor);
            copy_nonoverlapping(src, dst, data_size);
        }
        self.cursor += data_size;
        Ok(())
    }
}
