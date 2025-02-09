use std::{mem, sync::Arc};

use kiri_backend::vulkan::{BufferCreateDesc, BufferHandle, BufferSlice, GraphicsDevice};
use kiri_common::BlockAllocator;
use parking_lot::Mutex;

use crate::Error;

#[derive(Debug)]
struct UniformPage {
    handle: BufferHandle,
    size_range: (usize, usize),
    allocator: BlockAllocator,
}

const UNIFORM_PAGE_SIZE: usize = 64536;
const MAX_UNIFOMR_SIZE: usize = 16384;

pub struct ShaderUniforms {
    device: Arc<GraphicsDevice>,
    uniforms: Mutex<Vec<UniformPage>>,
}

impl ShaderUniforms {
    pub fn allocate<T: Copy>(&self, data: T) -> Result<BufferSlice, Error> {
        debug_assert!(mem::size_of::<T>() < MAX_UNIFOMR_SIZE);
        let mut pages = self.uniforms.lock();
        let item_size = mem::size_of::<T>();
        let upper_bound = item_size.next_power_of_two();
        let lower_bound = item_size.next_power_of_two() / 2 - 1;
        let allocated = pages
            .iter_mut()
            .find_map(|x| {
                if x.size_range.0 > lower_bound && x.size_range.1 <= upper_bound {
                    x.allocator
                        .allocate()
                        .map(|offset| BufferSlice::new(x.handle, offset, item_size as _))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                // Fixme:: unwrap
                let handle = self
                    .device
                    .create_buffer(
                        BufferCreateDesc::gpu(UNIFORM_PAGE_SIZE as _)
                            .transfer_destination()
                            .uniform_buffer(),
                    )
                    .unwrap();

                let mut allocator = BlockAllocator::new(
                    UNIFORM_PAGE_SIZE as _,
                    (UNIFORM_PAGE_SIZE / upper_bound) as _,
                );
                let offset = allocator.allocate().unwrap();
                pages.push(UniformPage {
                    handle,
                    allocator,
                    size_range: (lower_bound, upper_bound),
                });
                BufferSlice::new(handle, offset, item_size as _)
            });
        drop(pages);
        self.device.upload_buffer(allocated.into(), &[data])?;
        Ok(allocated)
    }

    pub fn free(&self, uniform: BufferSlice) {
        let mut pages = self.uniforms.lock();
        pages
            .iter_mut()
            .find(|page| page.handle == uniform.handle)
            .iter_mut()
            .for_each(|page| page.allocator.dealloc(uniform.offset as usize));
    }
}
