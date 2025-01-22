// Copyright (C) 2024-2025 gigablaster

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

use std::{marker::PhantomData, mem, sync::Arc};

use crate::{BufferHandle, BufferSlice, Error, Renderer};
use kiri_backend::{Buffer, BufferCreateDesc};
use kiri_common::{Align, BlockAllocator};
use parking_lot::Mutex;

const MATERIALS_PER_PAGE: u64 = 256;

#[derive(Debug)]
struct ConstUniformBufferPage {
    handle: BufferHandle,
    allocator: BlockAllocator,
}
/// Static uniform allocator
///
/// Allocate uniforms of single type.
#[derive(Debug)]
pub struct ConstUniformBuffer<T: Copy> {
    renderer: Arc<Renderer>,
    pages: Mutex<Vec<ConstUniformBufferPage>>,
    _phantom: PhantomData<T>,
}

impl<T: Copy> Drop for ConstUniformBuffer<T> {
    fn drop(&mut self) {
        self.pages
            .lock()
            .drain(..)
            .for_each(|x| self.renderer.destroy_buffer(x.handle));
    }
}

impl<T: Copy> ConstUniformBuffer<T> {
    pub fn new(renderer: &Arc<Renderer>) -> Self {
        Self {
            renderer: renderer.clone(),
            pages: Default::default(),
            _phantom: PhantomData,
        }
    }

    pub fn allocate(&self, data: T) -> Result<BufferSlice, Error> {
        let mut pages = self.pages.lock();
        let item_size = mem::size_of::<T>() as u64;
        let allocated = pages
            .iter_mut()
            .find_map(|x| {
                x.allocator
                    .allocate()
                    .map(|offset| BufferSlice::new(x.handle, offset, item_size))
            })
            .unwrap_or_else(|| {
                let chunk_size = item_size.align(
                    self.renderer
                        .device
                        .physical_device
                        .properties
                        .limits
                        .min_uniform_buffer_offset_alignment,
                );
                // Fixme:: unwrap
                let buffer = Arc::new(
                    Buffer::new(
                        &self.renderer.device,
                        BufferCreateDesc::gpu(chunk_size * MATERIALS_PER_PAGE)
                            .transfer_destination()
                            .uniform_buffer(),
                    )
                    .unwrap(),
                );

                let mut allocator = BlockAllocator::new(chunk_size, MATERIALS_PER_PAGE);
                let offset = allocator.allocate().unwrap();
                let handle = self.renderer.register_buffer(buffer);
                pages.push(ConstUniformBufferPage { handle, allocator });
                BufferSlice::new(handle, offset, item_size)
            });
        drop(pages);
        self.renderer
            .upload_buffer_data(allocated.into(), &[data])?;
        Ok(allocated)
    }

    pub fn free(&self, data: BufferSlice) {
        let mut pages = self.pages.lock();
        let page = pages
            .iter_mut()
            .find(|page| page.handle == data.handle)
            .expect("Uniform must be freed from it's own allocator");
        page.allocator.dealloc(data.offset);
    }
}
