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

mod bbox;
mod bsphere;
mod camera;
mod plane;
mod ray;

pub use bbox::*;
pub use bsphere::*;
pub use camera::*;
use glam::{Affine3A, Vec3, Vec3A};
pub use plane::*;
pub use ray::*;

const EPSILON: f32 = 0.0000001;

/// Shared traits for every bounding volume
pub trait Bounds: Copy {
    fn contains_point3(self, point: Vec3) -> bool {
        self.contains_point3a(point.into())
    }

    fn contains_point3a(self, point: Vec3A) -> bool;

    fn intersects_bbox(self, bbox: BoundingBox) -> bool;

    fn interesects_sphere(self, sphere: BoundingSphere) -> bool;

    fn intersects_ray(self, ray: Ray) -> Option<f32>;

    fn transform(self, transform: Affine3A) -> Self;

    fn is_on_or_forward_plane(self, plane: Plane) -> bool;

    fn is_visible(self, frustrum: &[Plane]) -> bool {
        frustrum.iter().all(|x| self.is_on_or_forward_plane(*x))
    }
}

pub fn ray_triangle_intersection(ray: Ray, p0: Vec3A, p1: Vec3A, p2: Vec3A) -> Option<f32> {
    let n = (p1 - p0).cross(p2 - p0);
    let plane = Plane {
        origin: p0,
        normal: n,
    };
    if let Some(t) = plane.intersects_ray(ray) {
        let x = ray.origin + ray.direction * t;
        if (p1 - p0).cross(x - p0).dot(n) >= 0.0
            && (p2 - p1).cross(x - p1).dot(n) >= 0.0
            && (p0 - p2).cross(x - p2).dot(n) >= 0.0
        {
            return Some(t);
        }
    }
    None
}

#[cfg(test)]
mod test {
    use glam::{vec3, vec3a, Vec3};

    use crate::{ray_triangle_intersection, Ray, EPSILON};

    #[test]
    fn ray_intersects_triangle() {
        let p0 = vec3a(-2.0, 2.0, 1.0);
        let p1 = vec3a(2.0, 0.0, -1.0);
        let p2 = vec3a(-2.0, -2.0, 1.0);
        let ray = Ray::new(vec3(0.0, 0.0, 3.0), Vec3::NEG_Z);
        assert!((ray_triangle_intersection(ray, p0, p1, p2).unwrap()).abs() - 3.0 < EPSILON);
    }

    #[test]
    fn ray_not_intersects_triangle_oppsite_direction() {
        let p0 = vec3a(-2.0, 2.0, 1.0);
        let p1 = vec3a(2.0, 0.0, -1.0);
        let p2 = vec3a(-2.0, -2.0, 1.0);
        let ray = Ray::new(vec3(0.0, 0.0, -3.0), Vec3::NEG_Z);
        assert_eq!(None, ray_triangle_intersection(ray, p0, p1, p2));
    }

    #[test]
    fn ray_not_intersects_triangle_outside() {
        let p0 = vec3a(-2.0, 2.0, 1.0);
        let p1 = vec3a(2.0, 0.0, -1.0);
        let p2 = vec3a(-2.0, -2.0, 1.0);
        let ray = Ray::new(vec3(5.0, 5.0, 3.0), Vec3::NEG_Z);
        assert_eq!(None, ray_triangle_intersection(ray, p0, p1, p2));
    }
}
