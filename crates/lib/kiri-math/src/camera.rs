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

use glam::{Mat4, Vec3};

use crate::Plane;

pub trait Camera {
    fn view(&self) -> Mat4;
    fn projection(&self) -> Mat4;
    fn frustum(&self) -> [Plane; 6];
}

#[derive(Debug, Clone, Copy)]
pub struct PerspectiveCamera {
    pub origin: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl PerspectiveCamera {
    pub fn new(
        origin: Vec3,
        forward: Vec3,
        up: Vec3,
        fov: f32,
        aspect: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            origin,
            forward: forward.normalize(),
            up,
            fov,
            aspect,
            znear,
            zfar,
        }
    }
}

impl Camera for PerspectiveCamera {
    fn view(&self) -> Mat4 {
        Mat4::look_to_lh(self.origin, self.forward, self.up)
    }

    fn projection(&self) -> Mat4 {
        Mat4::perspective_lh(self.fov, self.aspect, self.znear, self.zfar)
    }

    fn frustum(&self) -> [Plane; 6] {
        let half_vside = self.zfar * (self.fov * 0.5).tan();
        let half_hside = half_vside * self.aspect;
        let far = self.zfar * self.forward;
        let right = self.forward.cross(self.up).normalize();
        [
            Plane::new(self.origin + self.znear * self.forward, self.forward),
            Plane::new(self.origin + far, -self.forward),
            Plane::new(self.origin, (far - right * half_hside).cross(self.up)),
            Plane::new(self.origin, self.up.cross(far + right * half_hside)),
            Plane::new(self.origin, right.cross(far - self.up * half_vside)),
            Plane::new(self.origin, (far + self.up * half_vside).cross(right)),
        ]
    }
}
