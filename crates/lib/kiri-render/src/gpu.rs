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

use kiri_math::{Mat4, Vec3};

use crate::{DirectionalLight, HemisphericalAmbient};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct RenderPassGpuData {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
    pub eye_position: Vec3,
    pub lights: [DirectionalLight; 3],
    pub ambient: HemisphericalAmbient,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuInstanceData {
    pub model: Mat4,
    pub uv_scale: f32,
}
