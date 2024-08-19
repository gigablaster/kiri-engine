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

use ash::vk;

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuMeshMaterial {
    pub base_color: u32,
    pub normals: u32,
    pub metallic_roughness: u32,
    pub occlusion: u32,
    pub emissive: u32,
    pub emissive_power: f32,
    pub alpha_cutoff: f32
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuMeshDraw {
    pub world_transform: glam::Mat4,
    pub vertices: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
    pub material_index: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuDirectionalLight {
    pub direction: glam::Vec3A,
    pub color: glam::Vec3A
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct HemisphericalAmbientLight {
    pub top: glam::Vec3A,
    pub middle: glam::Vec3A,
    pub bottom: glam::Vec3A
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuPassData{
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub view_projection: glam::Mat4,
    pub directional_lights: [GpuDirectionalLight; 3],
}
