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

use glam::{Affine3A, Vec3, Vec3A};

use crate::{Bounds, Ray};

#[derive(Debug, Default, Clone, Copy)]
pub struct BoundingSphere {
    pub center: Vec3A,
    pub radius: f32,
}

impl BoundingSphere {
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center: center.into(),
            radius: radius,
        }
    }

    pub fn from_array_and_radius(center: [f32; 3], radius: f32) -> Self {
        Self {
            center: Vec3A::from_array(center),
            radius,
        }
    }
}

impl Bounds for BoundingSphere {
    fn contains_point3a(self, point: Vec3A) -> bool {
        self.center.distance_squared(point) <= self.radius * self.radius
    }

    fn intersects_bbox(self, bbox: crate::BoundingBox) -> bool {
        bbox.interesects_sphere(self)
    }

    fn interesects_sphere(self, sphere: BoundingSphere) -> bool {
        self.center.distance_squared(sphere.center)
            <= self.radius * self.radius + sphere.radius * sphere.radius
    }

    fn intersects_ray(self, ray: Ray) -> Option<f32> {
        // https://www.cs.colostate.edu/~cs410/yr2016fa/more_progress/cs410_F16_Lecture14.pdf
        let to_center = self.center - ray.origin;
        if to_center.dot(ray.direction) <= 0.0 {
            return None;
        }
        let r2 = self.radius * self.radius;
        let c2 = self.center.distance_squared(ray.origin);
        let v2 = (to_center.project_onto(ray.direction * to_center.length())).length_squared();
        let d2 = r2 - (c2 - v2);
        if d2 < 0.0 {
            return None;
        }
        Some(v2.sqrt() - d2.sqrt())
    }

    fn transform(self, transform: Affine3A) -> Self {
        let (scale, _, _) = transform.to_scale_rotation_translation();
        let scale = scale.max_element();
        Self {
            center: transform.transform_point3a(self.center),
            radius: self.radius * scale,
        }
    }
}

#[cfg(test)]
mod test {
    use glam::{vec3, Vec3};

    use crate::{Bounds, Ray};

    use super::BoundingSphere;

    #[test]
    fn ray_intersects_shpere() {
        let sphere = BoundingSphere::new(vec3(2.0, 2.0, 2.0), 1.0);
        assert_eq!(
            Some(1.0),
            sphere.intersects_ray(Ray::new(vec3(0.0, 2.0, 2.0), Vec3::X))
        );
        assert_eq!(
            Some(1.0),
            sphere.intersects_ray(Ray::new(vec3(2.0, 0.0, 2.0), Vec3::Y))
        );
        assert_eq!(
            Some(1.0),
            sphere.intersects_ray(Ray::new(vec3(2.0, 2.0, 0.0), Vec3::Z))
        );
        assert_eq!(
            Some(1.0),
            sphere.intersects_ray(Ray::new(vec3(4.0, 2.0, 2.0), Vec3::NEG_X))
        );
        assert_eq!(
            Some(1.0),
            sphere.intersects_ray(Ray::new(vec3(2.0, 4.0, 2.0), Vec3::NEG_Y))
        );
        assert_eq!(
            Some(1.0),
            sphere.intersects_ray(Ray::new(vec3(2.0, 2.0, 4.0), Vec3::NEG_Z))
        );
    }

    #[test]
    fn ray_not_intersects_sphere_in_opposite_direction() {
        let sphere = BoundingSphere::new(vec3(2.0, 2.0, 2.0), 1.0);
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(0.0, 2.0, 2.0), Vec3::NEG_X))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(2.0, 0.0, 2.0), Vec3::NEG_Y))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(2.0, 2.0, 0.0), Vec3::NEG_Z))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(4.0, 2.0, 2.0), Vec3::X))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(2.0, 4.0, 2.0), Vec3::Y))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(2.0, 2.0, 4.0), Vec3::Z))
        );
    }

    #[test]
    fn ray_not_intersects_shpere_when_outside() {
        let sphere = BoundingSphere::new(vec3(2.0, 2.0, 2.0), 1.0);
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(0.0, 5.0, 2.0), Vec3::X))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(5.0, 0.0, 2.0), Vec3::Y))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(2.0, 5.0, 0.0), Vec3::Z))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(4.0, 2.0, 5.0), Vec3::NEG_X))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(5.0, 4.0, 2.0), Vec3::NEG_Y))
        );
        assert_eq!(
            None,
            sphere.intersects_ray(Ray::new(vec3(2.0, 5.0, 4.0), Vec3::NEG_Z))
        );
    }
}
