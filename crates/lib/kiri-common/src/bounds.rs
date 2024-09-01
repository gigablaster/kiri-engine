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

#[derive(Debug, Default, Clone, Copy)]
pub struct Bounds {
    pub center: glam::Vec3,
    pub radius: f32,
}

impl Bounds {
    pub fn from_array_and_radius(center: [f32; 3], radius: f32) -> Self {
        Self {
            center: glam::Vec3::from_array(center),
            radius,
        }
    }

    pub fn transform(self, transform: glam::Affine3A) -> Self {
        let (scale, _, _) = transform.to_scale_rotation_translation();
        let scale = scale.max_element();
        Self {
            center: transform.transform_point3(self.center),
            radius: self.radius * scale,
        }
    }
}
