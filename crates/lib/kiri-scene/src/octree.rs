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

use glam::Affine3A;
use kiri_common::Handle;
use kiri_resources::ModelHandle;

pub type LeafHandle = Handle<ModelHandle>;

#[derive(Debug, Clone, Copy)]
struct Leaf {
    transform: Affine3A,
    handle: LeafHandle,
}

struct Node {
    parent: u32,
    leafs: Vec<Leaf>,
    children: Option<[u32; 8]>,
}
