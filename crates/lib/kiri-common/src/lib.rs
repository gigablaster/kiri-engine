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

mod chunky_list;
mod executors;
mod memory;
mod pool;
mod time;

use std::path::PathBuf;

pub use chunky_list::*;
use directories::ProjectDirs;
pub use executors::*;
pub use memory::*;
pub use pool::*;
pub use smol::block_on;
pub use smol::future::yield_now;
pub use smol::Task;
use speedy::{Readable, Writable};
pub use time::*;

pub trait Align<T> {
    fn align(self, align: T) -> Self;
}

impl Align<u32> for u32 {
    fn align(self, align: u32) -> Self {
        if self == 0 || self % align == 0 {
            self
        } else {
            (self & !(align - 1)) + align
        }
    }
}

impl Align<u64> for u64 {
    fn align(self, align: u64) -> Self {
        if self == 0 || self % align == 0 {
            self
        } else {
            (self & !(align - 1)) + align
        }
    }
}

impl Align<usize> for usize {
    fn align(self, align: usize) -> Self {
        if self == 0 || self % align == 0 {
            self
        } else {
            (self & !(align - 1)) + align
        }
    }
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn any_as_u8_slice<T: Sized + Copy>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}

pub struct GameAppConfig<'a> {
    pub developer: &'a str,
    pub name: &'a str,
}

impl GameAppConfig<'_> {
    pub fn cache(&self) -> Option<PathBuf> {
        ProjectDirs::from("com", self.developer, self.name).map(|dirs| dirs.cache_dir().into())
    }
}

#[cfg(test)]
mod test {
    use crate::Align;

    #[test]
    fn align() {
        assert_eq!(0, 0u32.align(64));
        assert_eq!(64, 50u32.align(64));
        assert_eq!(64, 64u32.align(64));
        assert_eq!(128, 100u32.align(64));
        assert_eq!(128, 128u32.align(64));
    }
}
