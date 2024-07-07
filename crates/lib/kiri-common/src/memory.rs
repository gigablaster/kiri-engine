// Copyright (C) 2023 gigablaster

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
    cmp::min,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::Align;

#[derive(Debug, Clone, Copy)]
struct BlockData(usize, usize);

#[derive(Debug, Clone, Copy)]
enum Block {
    Free(BlockData),
    Used(BlockData),
}

/// Sub-buffer allocator for all in-game geometry.
/// It's simple free-list allocator
#[derive(Debug)]
pub struct DynamicAllocator {
    granularity: usize,
    blocks: Vec<Block>,
}

fn align(value: usize, align: usize) -> usize {
    if value == 0 || value % align == 0 {
        value
    } else {
        (value & !(align - 1)) + align
    }
}

impl DynamicAllocator {
    pub fn new(size: usize, granularity: usize) -> Self {
        let mut blocks = Vec::with_capacity(256);
        blocks.push(Block::Free(BlockData(0, size)));
        Self {
            blocks,
            granularity,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        if let Some(index) = self.find_first_free_block(size) {
            self.split_and_insert_block(index, size)
        } else {
            None
        }
    }

    pub fn allocate_back(&mut self, size: usize) -> Option<usize> {
        if let Some(index) = self.find_last_free_block(size) {
            self.split_and_insert_block_end(index, size)
        } else {
            None
        }
    }

    pub fn deallocate(&mut self, offset: usize) {
        if let Some(index) = self.find_used_block(offset) {
            if let Block::Used(block) = self.blocks[index] {
                self.blocks[index] = Block::Free(block);
                self.merge_free_blocks(index);
                return;
            }
        }

        panic!("Attempt to free already freed block or block from different allocator");
    }

    fn find_used_block(&self, offset: usize) -> Option<usize> {
        self.blocks.iter().enumerate().find_map(|(index, block)| {
            if let Block::Used(block) = block {
                if block.0 == offset {
                    return Some(index);
                }
            }
            None
        })
    }

    fn find_last_free_block(&self, size: usize) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, block)| {
                if let Block::Free(block) = block {
                    if block.1 >= size {
                        Some(index)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
    }

    fn find_first_free_block(&self, size: usize) -> Option<usize> {
        self.blocks.iter().enumerate().find_map(|(index, block)| {
            if let Block::Free(block) = block {
                if block.1 >= size {
                    Some(index)
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    fn merge_free_blocks(&mut self, index: usize) {
        let mut index = index;
        loop {
            if index > 0 {
                if let Some(block) = self.blocks.get(index - 1) {
                    match block {
                        Block::Free(_) => {
                            index -= 1;
                            continue;
                        }
                        Block::Used(_) => {}
                    }
                }
            }
            break;
        }
        loop {
            if let Block::Free(block) = self.blocks[index] {
                if let Some(next) = self.blocks.get(index + 1) {
                    match next {
                        Block::Free(next) => {
                            self.blocks[index] = Block::Free(BlockData(block.0, block.1 + next.1));
                            self.blocks.remove(index + 1);
                            continue;
                        }
                        Block::Used(_) => {}
                    }
                }
            }
            break;
        }
    }

    fn split_and_insert_block(&mut self, index: usize, size: usize) -> Option<usize> {
        let size = align(size, self.granularity);
        if let Some(block) = self.blocks.get(index) {
            let block = *block;
            if let Block::Free(block) = block {
                assert!(size <= block.1);
                let new_size = block.1 - size;
                self.blocks[index] = Block::Used(BlockData(block.0, size));
                if new_size > 0 {
                    self.blocks
                        .insert(index + 1, Block::Free(BlockData(block.0 + size, new_size)));
                }
                return Some(block.0);
            }
        }

        None
    }

    fn split_and_insert_block_end(&mut self, index: usize, size: usize) -> Option<usize> {
        let size = align(size, self.granularity);
        if let Some(block) = self.blocks.get(index) {
            let block = *block;
            if let Block::Free(block) = block {
                assert!(size <= block.1);
                let split = block.1 - size;
                self.blocks[index] = Block::Used(BlockData(block.0 + split, size));
                if split > 0 {
                    self.blocks
                        .insert(index, Block::Free(BlockData(block.0, split)));
                }
                return Some(block.0 + split);
            }
        }

        None
    }
}

pub struct RingAllocator {
    size: usize,
    aligment: usize,
    head: AtomicUsize,
}

impl RingAllocator {
    pub fn new(size: usize, aligment: usize) -> Self {
        Self {
            head: AtomicUsize::new(0),
            size,
            aligment,
        }
    }

    pub fn allocate(&self, size: usize) -> usize {
        assert!(size <= self.size);
        let aligned_size = align(size, self.aligment);
        loop {
            let old_head = self.head.load(Ordering::Acquire);
            let end = old_head + aligned_size;
            let (head, new_head) = if end > self.size {
                (0, aligned_size)
            } else {
                (old_head, end)
            };
            if self
                .head
                .compare_exchange(old_head, new_head, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                return head;
            }
        }
    }
}

#[derive(Debug)]
pub struct BumpAllocator {
    top: AtomicUsize,
    size: usize,
}

impl BumpAllocator {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            top: AtomicUsize::new(0),
        }
    }

    pub fn allocate(&self, size: usize, aligment: usize) -> Option<usize> {
        self.top
            .fetch_update(Ordering::Release, Ordering::SeqCst, |x| {
                let new_top = (x + size).align(aligment);
                if new_top <= self.size {
                    Some(new_top)
                } else {
                    None
                }
            })
            .ok()
    }

    pub fn reset(&self) {
        self.top.store(0, Ordering::SeqCst);
    }

    pub fn validate(&self, size: usize, aligment: usize) -> usize {
        let base = self.top.load(Ordering::SeqCst).align(aligment);
        min(size, self.size - base)
    }
}
#[derive(Debug)]
pub struct BlockAllocator {
    chunk_size: usize,
    chunk_count: usize,
    empty: Vec<usize>,
}

impl BlockAllocator {
    pub fn new(chunk_size: usize, chunk_count: usize) -> Self {
        let empty = (0..chunk_count).rev().collect::<Vec<_>>();

        Self {
            chunk_size,
            chunk_count,
            empty,
        }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        if let Some(slot) = self.empty.pop() {
            Some(slot * self.chunk_size)
        } else {
            None
        }
    }

    pub fn dealloc(&mut self, offset: usize) {
        let index = offset / self.chunk_size;
        assert!(index < self.chunk_count && offset % self.chunk_size == 0);
        assert!(!self.empty.contains(&index));
        self.empty.push(index);
    }
}

#[cfg(test)]
mod test {
    use crate::memory::RingAllocator;

    use super::BumpAllocator;

    #[test]
    fn ring_allocator() {
        let buffer = RingAllocator::new(1024, 64);
        assert_eq!(0, buffer.allocate(500));
        assert_eq!(512, buffer.allocate(500));
        assert_eq!(0, buffer.allocate(128));
    }

    #[test]
    fn bump_allocator() {
        let allocator = BumpAllocator::new(1024);
        assert_eq!(Some(0), allocator.allocate(500, 64));
        assert_eq!(Some(512), allocator.allocate(100, 64));
        assert_eq!(Some(640), allocator.allocate(100, 64));
        assert_eq!(None, allocator.allocate(500, 64));
        allocator.reset();
        assert_eq!(Some(0), allocator.allocate(500, 64));
        assert_eq!(Some(512), allocator.allocate(100, 64));
        assert_eq!(Some(640), allocator.allocate(100, 64));
        assert_eq!(None, allocator.allocate(500, 64));
    }

    #[test]
    fn bump_allocator_align() {
        let allocator = BumpAllocator::new(1024);
        assert_eq!(Some(0), allocator.allocate(10, 128));
        assert_eq!(Some(128), allocator.allocate(10, 128));
    }
}
