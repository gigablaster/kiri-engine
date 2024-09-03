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

use glam::{Vec3, Vec3A};

use crate::{Ray, EPSILON};

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub origin: Vec3A,
    pub normal: Vec3A,
}

impl Plane {
    pub fn new(origin: Vec3, normal: Vec3) -> Self {
        Self {
            origin: origin.into(),
            normal: normal.normalize().into(),
        }
    }

    pub fn intersects_ray(self, ray: Ray) -> Option<f32> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() > EPSILON {
            let t = (self.origin - ray.origin).dot(self.normal) / denom;
            if t >= 0.0 {
                return Some(t);
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use glam::{vec3, Vec3};

    use crate::Ray;

    use super::Plane;

    #[test]
    fn ray_plane_intersets_right_angle() {
        let plane = Plane::new(Vec3::ZERO, Vec3::Z);
        assert_eq!(
            Some(2.0),
            plane.intersects_ray(Ray::new(vec3(0.0, 0.0, 2.0), Vec3::NEG_Z))
        );
        assert_eq!(
            Some(2.0),
            plane.intersects_ray(Ray::new(vec3(0.0, 0.0, -2.0), Vec3::Z))
        );
    }

    #[test]
    fn ray_plane_not_intersects_opposite_direction() {
        let plane = Plane::new(Vec3::ZERO, Vec3::Z);
        assert_eq!(
            None,
            plane.intersects_ray(Ray::new(vec3(0.0, 0.0, 2.0), Vec3::Z))
        );
        assert_eq!(
            None,
            plane.intersects_ray(Ray::new(vec3(0.0, 0.0, -2.0), Vec3::NEG_Z))
        );
    }

    #[test]
    fn ray_plane_not_intersects_parallel() {
        let plane = Plane::new(Vec3::ZERO, Vec3::Z);
        assert_eq!(
            None,
            plane.intersects_ray(Ray::new(vec3(0.0, 0.0, 2.0), Vec3::Y))
        );
        assert_eq!(
            None,
            plane.intersects_ray(Ray::new(vec3(0.0, 0.0, 2.0), Vec3::X))
        );
    }
}
