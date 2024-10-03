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

use kiri_backend::BufferCreateDesc;
use kiri_common::BlockAllocator;
use kiri_gfx::{BufferHandle, BufferSlice, Renderer};
use parking_lot::Mutex;

use crate::Error;

const MATERIALS_PER_PAGE: u64 = 256;

#[derive(Debug)]
struct ConstUniformBufferPage {
    pub buffer: BufferHandle,
    allocator: BlockAllocator,
}
/// Static uniform allocator
///
/// Allocate uniforms of single type.
#[derive(Debug)]
pub struct ConstUniformBuffer {
    renderer: Arc<Renderer>,
    item_size: u64,
    pages: Mutex<Vec<ConstUniformBufferPage>>,
}

impl Drop for ConstUniformBuffer {
    fn drop(&mut self) {
        self.pages
            .lock()
            .drain(..)
            .for_each(|x| self.renderer.destroy_buffer(x.buffer));
    }
}

impl ConstUniformBuffer {
    pub fn new(renderer: &Arc<Renderer>, item_size: u64) -> Self {
        Self {
            renderer: renderer.clone(),
            item_size,
            pages: Default::default(),
        }
    }

    pub fn allocate(&self, data: &[u8]) -> Result<BufferSlice, kiri_gfx::Error> {
        let mut pages = self.pages.lock();
        let allocated = pages
            .iter_mut()
            .find_map(|x| {
                x.allocator
                    .allocate()
                    .map(|offset| BufferSlice::new(x.buffer, offset, self.item_size))
            })
            .unwrap_or_else(|| {
                let chunk_size = self
                    .renderer
                    .device
                    .physical_device
                    .properties
                    .limits
                    .min_uniform_buffer_offset_alignment
                    .max(self.item_size);
                // Fixme:: unwrap
                let buffer = self
                    .renderer
                    .create_buffer(
                        BufferCreateDesc::gpu(chunk_size * MATERIALS_PER_PAGE)
                            .transfer_destination()
                            .uniform_buffer(),
                    )
                    .unwrap();
                let mut allocator = BlockAllocator::new(chunk_size, MATERIALS_PER_PAGE);
                let offset = allocator.allocate().unwrap();
                pages.push(ConstUniformBufferPage { buffer, allocator });
                BufferSlice::new(buffer, offset, self.item_size)
            });
        drop(pages);
        self.renderer.upload_buffer(allocated.into(), data)?;
        Ok(allocated)
    }

    pub fn free(&self, data: BufferSlice) {
        let mut pages = self.pages.lock();
        let page = pages
            .iter_mut()
            .find(|page| page.buffer == data.handle)
            .expect("Uniform must be freed from it's own alloactor");
        page.allocator.dealloc(data.offset);
    }
}
