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

use kiri_assets::{MeshAssetMaterial, StaticMeshVertex};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuMeshMaterial {
    pub base_color: glam::Vec4,
    pub emissive_color: glam::Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub alpha_cutoff: f32,
    pub emissive_power: f32,
}

impl GpuMeshMaterial {
    pub fn new(value: &MeshAssetMaterial) -> Self {
        let [_, roughness, metallic, _] = value.metallic_roughness.get_color();
        let alpha_cutoff = value.blend.get_alpha_cut();
        let emissive_power = value.get_emissive_power();
        Self {
            base_color: value.base_color.get_color().into(),
            emissive_color: value.emissive.get_color().into(),
            metallic,
            roughness,
            alpha_cutoff,
            emissive_power,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuStaticVertex {
    pub position: glam::U16Vec3,
    _pad: u16,
    pub normal: glam::U16Vec2,
    pub tangent: glam::U16Vec2,
    pub uv1: glam::U16Vec2,
    pub uv2: glam::U16Vec2,
}

impl From<StaticMeshVertex> for GpuStaticVertex {
    fn from(value: StaticMeshVertex) -> Self {
        Self {
            position: glam::U16Vec3::from_array(value.position),
            _pad: 0,
            normal: glam::U16Vec2::from_array(value.normal),
            tangent: glam::U16Vec2::from_array(value.tangent),
            uv1: glam::U16Vec2::from_array(value.uv1),
            uv2: glam::U16Vec2::from_array(value.uv2),
        }
    }
}
