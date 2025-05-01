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

use glam::{vec3a, Affine3A, Vec3, Vec3A};

use crate::{BoundingSphere, Bounds, Plane, Ray};

#[derive(Debug, Default, Clone, Copy)]
pub struct BoundingBox {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl BoundingBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        let mi = min.min(max);
        let ma = min.max(max);
        Self {
            min: mi.into(),
            max: ma.into(),
        }
    }

    pub fn from_extent(center: Vec3, extent: Vec3) -> Self {
        let ex = extent.abs();
        let mi = center - ex * 0.5;
        let ma = center + ex * 0.5;
        Self {
            min: mi.into(),
            max: ma.into(),
        }
    }

    pub fn from_points3(points: &[Vec3]) -> Self {
        debug_assert!(
            !points.is_empty(),
            "Can't caluculate bounds when there's no points"
        );
        let mut min = Vec3::MAX;
        let mut max = Vec3::MIN;
        for point in points {
            min = min.min(*point);
            max = max.max(*point);
        }
        Self::new(min, max)
    }

    pub fn from_points3a(points: &[Vec3A]) -> Self {
        debug_assert!(
            !points.is_empty(),
            "Can't caluculate bounds when there's no points"
        );
        let mut min = Vec3A::MAX;
        let mut max = Vec3A::MIN;
        for point in points {
            min = min.min(*point);
            max = max.max(*point);
        }
        Self { min, max }
    }

    pub fn from_arrays(min: [f32; 3], max: [f32; 3]) -> Self {
        Self::new(min.into(), max.into())
    }

    pub fn from_extent_array(center: [f32; 3], extent: [f32; 3]) -> Self {
        Self::from_extent(center.into(), extent.into())
    }

    pub fn from_points_array(points: &[[f32; 3]]) -> Self {
        debug_assert!(
            !points.is_empty(),
            "Can't caluculate bounds when there's no points"
        );
        let mut min = Vec3::MAX;
        let mut max = Vec3::MIN;
        for point in points {
            let point: Vec3 = (*point).into();
            min = min.min(point);
            max = max.max(point);
        }
        Self::new(min, max)
    }

    pub fn center(self) -> Vec3 {
        self.min.midpoint(self.max).into()
    }

    pub fn extents(self) -> Vec3 {
        (self.max - self.min).into()
    }

    pub fn corners(self) -> [Vec3A; 8] {
        [
            self.min,
            vec3a(self.min.x, self.max.y, self.min.z),
            vec3a(self.max.x, self.max.y, self.min.z),
            vec3a(self.max.x, self.min.y, self.min.z),
            vec3a(self.min.x, self.min.y, self.max.z),
            vec3a(self.min.x, self.max.y, self.max.z),
            self.max,
            vec3a(self.max.x, self.min.y, self.max.z),
        ]
    }

    pub fn inside(self, other: BoundingBox) -> bool {
        let min = self.min.cmple(other.min);
        let max = self.max.cmpge(other.max);
        (min & max).all()
    }
}

impl Bounds for BoundingBox {
    fn contains_point3a(self, point: Vec3A) -> bool {
        let min = self.min.cmple(point);
        let max = self.max.cmpge(point);
        (min & max).all()
    }

    fn intersects_bbox(self, bbox: BoundingBox) -> bool {
        let min = self.min.cmple(bbox.max);
        let max = self.max.cmpge(bbox.min);
        (min & max).all()
    }

    fn interesects_sphere(self, sphere: BoundingSphere) -> bool {
        // https://codereview.stackexchange.com/questions/145809/high-performance-branchless-intersection-testing-sphere-aabb-aabb-aabb
        let ex = (self.min - sphere.center).max(Vec3A::ZERO)
            + (sphere.center - self.max).max(Vec3A::ZERO);
        let less_than_radius = ex.cmple(vec3a(sphere.radius, sphere.radius, sphere.radius));
        less_than_radius.all() && ex.length_squared() < sphere.radius * sphere.radius
    }

    fn intersects_ray(self, ray: Ray) -> Option<f32> {
        // https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms#18459
        let inv_dir = ray.direction.recip();
        let tmin = (self.min - ray.origin) * inv_dir;
        let tmax = (self.max - ray.origin) * inv_dir;
        let min = tmin.min(tmax);
        let max = tmin.max(tmax);
        let min = min.max_element();
        let max = max.min_element();
        if max < 0.0 || min > max {
            return None;
        }
        Some(min)
    }

    fn transform(self, transform: Affine3A) -> Self {
        let points = self.corners().map(|x| transform.transform_point3a(x));
        Self::from_points3a(&points)
    }

    fn is_on_or_forward_plane(self, plane: Plane) -> bool {
        let extents = self.max - self.min;
        let center = self.max.midpoint(self.min);
        let r = (plane.normal.abs() * extents).length_squared();
        -r <= plane.signed_distance_point3a(center)
    }
}

#[cfg(test)]
mod test {
    use std::f32::consts::FRAC_PI_2;

    use glam::{vec3, vec3a, Affine3A, Vec3};

    use crate::{BoundingSphere, Bounds, Camera, PerspectiveCamera, Plane, Ray};

    use super::BoundingBox;

    #[test]
    fn bbox_create_correct_min_max() {
        let bbox = BoundingBox::new(Vec3::ONE, Vec3::NEG_ONE);
        assert_eq!(Vec3::NEG_ONE, bbox.min.into());
        assert_eq!(Vec3::ONE, bbox.max.into());
        assert_eq!(Vec3::ZERO, bbox.center());
        assert_eq!(vec3(2.0, 2.0, 2.0), bbox.extents());
        let bbox = BoundingBox::from_extent(Vec3::ONE, vec3(2.0, 2.0, 2.0));
        assert_eq!(Vec3::ZERO, bbox.min.into());
        assert_eq!(vec3(2.0, 2.0, 2.0), bbox.max.into());
        assert_eq!(Vec3::ONE, bbox.center());
        assert_eq!(vec3(2.0, 2.0, 2.0), bbox.extents());
    }

    #[test]
    fn bbox_from_point_array() {
        let points = [
            [-2.0, -3.0, -4.0],
            [4.0, 3.0, 2.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
            [1.0, 2.0, 2.0],
        ];
        let bbox = BoundingBox::from_points_array(&points);
        assert_eq!(vec3a(-2.0, -3.0, -4.0), bbox.min);
        assert_eq!(vec3a(4.0, 3.0, 2.0), bbox.max);
    }

    #[test]
    fn bbox_transform() {
        let bbox = BoundingBox::from_extent(Vec3::ZERO, vec3(2.0, 4.0, 8.0));
        let scaled = bbox.transform(Affine3A::from_scale(vec3(1.0, 0.5, 0.25)));
        assert_eq!(Vec3::ZERO, scaled.center());
        assert_eq!(vec3(2.0, 2.0, 2.0), scaled.extents());
        let moved = bbox.transform(Affine3A::from_translation(vec3(1.0, 2.0, 3.0)));
        assert_eq!(vec3(1.0, 2.0, 3.0), moved.center());
        assert_eq!(vec3(2.0, 4.0, 8.0), moved.extents());
    }

    #[test]
    fn bbox_contains_point() {
        let bbox = BoundingBox::from_extent(Vec3::ZERO, vec3(2.0, 4.0, 8.0));
        assert!(bbox.contains_point3(Vec3::ZERO));
        assert!(bbox.contains_point3(vec3(1.0, 2.0, 4.0)));
        assert!(bbox.contains_point3(vec3(-1.0, 2.0, 4.0)));
        assert!(bbox.contains_point3(vec3(1.0, -2.0, 4.0)));
        assert!(bbox.contains_point3(vec3(-1.0, -2.0, 4.0)));
        assert!(bbox.contains_point3(vec3(1.0, 2.0, -4.0)));
        assert!(bbox.contains_point3(vec3(1.0, -2.0, -4.0)));
        assert!(bbox.contains_point3(vec3(-1.0, -2.0, -4.0)));
        assert!(!bbox.contains_point3(vec3(-2.0, -3.0, -5.0)));
        assert!(!bbox.contains_point3(vec3(2.0, -3.0, -5.0)));
        assert!(!bbox.contains_point3(vec3(-2.0, 3.0, -5.0)));
        assert!(!bbox.contains_point3(vec3(-2.0, -3.0, -5.0)));
        assert!(!bbox.contains_point3(vec3(2.0, -3.0, 5.0)));
        assert!(!bbox.contains_point3(vec3(-2.0, 3.0, 5.0)));
        assert!(!bbox.contains_point3(vec3(-2.0, -3.0, 5.0)));
    }

    #[test]
    fn bbox_ray_intersection_direct() {
        let bbox = BoundingBox::from_extent(vec3(5.0, 5.0, 5.0), vec3(2.0, 2.0, 2.0));
        // Rays into
        assert_eq!(
            Some(1.0),
            bbox.intersects_ray(Ray::new(vec3(3.0, 5.0, 5.0), Vec3::X))
        );
        assert_eq!(
            Some(1.0),
            bbox.intersects_ray(Ray::new(vec3(5.0, 3.0, 5.0), Vec3::Y))
        );
        assert_eq!(
            Some(1.0),
            bbox.intersects_ray(Ray::new(vec3(5.0, 5.0, 3.0), Vec3::Z))
        );
        assert_eq!(
            Some(1.0),
            bbox.intersects_ray(Ray::new(vec3(7.0, 5.0, 5.0), Vec3::NEG_X))
        );
        assert_eq!(
            Some(1.0),
            bbox.intersects_ray(Ray::new(vec3(5.0, 7.0, 5.0), Vec3::NEG_Y))
        );
        assert_eq!(
            Some(1.0),
            bbox.intersects_ray(Ray::new(vec3(5.0, 5.0, 7.0), Vec3::NEG_Z))
        );
    }

    #[test]
    fn bbox_ray_no_intersection_opposite() {
        let bbox = BoundingBox::from_extent(vec3(5.0, 5.0, 5.0), vec3(2.0, 2.0, 2.0));
        // Rays directed outside
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(3.0, 5.0, 5.0), Vec3::NEG_X))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(5.0, 3.0, 5.0), Vec3::NEG_Y))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(5.0, 5.0, 3.0), Vec3::NEG_Z))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(7.0, 5.0, 5.0), Vec3::X))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(5.0, 7.0, 5.0), Vec3::Y))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(5.0, 5.0, 7.0), Vec3::Z))
        );
    }

    #[test]
    fn bbox_ray_no_intersect_outside() {
        let bbox = BoundingBox::from_extent(vec3(5.0, 5.0, 5.0), vec3(2.0, 2.0, 2.0));
        // Rays outside
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(3.0, 10.0, 10.0), Vec3::X))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(10.0, 3.0, 10.0), Vec3::Y))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(10.0, 10.0, 3.0), Vec3::Z))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(7.0, 10.0, 10.0), Vec3::NEG_X))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(10.0, 7.0, 10.0), Vec3::NEG_Y))
        );
        assert_eq!(
            None,
            bbox.intersects_ray(Ray::new(vec3(10.0, 10.0, 7.0), Vec3::NEG_Z))
        );
    }

    #[test]
    fn bbox_outside_when_fully_outside() {
        assert!(
            !BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).inside(
                BoundingBox::from_extent(vec3(3.0, 3.0, 3.0), vec3(1.0, 1.0, 1.0))
            )
        );
    }

    #[test]
    fn bbox_outside_if_partially_intersects() {
        assert!(
            !BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).inside(
                BoundingBox::from_extent(vec3(2.0, 2.0, 2.0), vec3(2.0, 2.0, 2.0))
            )
        );
    }

    #[test]
    fn bbox_inside_when_smaller_and_inside() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).inside(
                BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0))
            )
        );
    }

    #[test]
    fn bbox_inside_when_exact_same() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).inside(
                BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0))
            )
        );
    }

    #[test]
    fn bbox_not_intersects_when_fully_outside() {
        assert!(
            !BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).intersects_bbox(
                BoundingBox::from_extent(vec3(3.0, 3.0, 3.0), vec3(1.0, 1.0, 1.0))
            )
        );
    }

    #[test]
    fn bbox_intesects_when_partially_intersects() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).intersects_bbox(
                BoundingBox::from_extent(vec3(2.0, 2.0, 2.0), vec3(2.0, 2.0, 2.0))
            )
        );
    }

    #[test]
    fn bbox_intersects_when_smaller_and_inside() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).intersects_bbox(
                BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0))
            )
        );
    }

    #[test]
    fn bbox_intersects_when_exact_same() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0)).intersects_bbox(
                BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0))
            )
        );
    }

    #[test]
    fn bbox_sphere_intersects() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0))
                .interesects_sphere(BoundingSphere::new(vec3(-1.0, -1.0, -1.0), 2.5))
        );
    }

    #[test]
    fn bbox_sphere_not_intersects() {
        assert!(
            !BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0))
                .interesects_sphere(BoundingSphere::new(vec3(-3.0, -3.0, -3.0), 2.0))
        );
    }

    #[test]
    fn bbox_sphere_intersects_sphere_inside() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0))
                .interesects_sphere(BoundingSphere::new(vec3(1.0, 1.0, 1.0), 1.0))
        );
    }

    #[test]
    fn bbox_sphere_intersects_box_inside() {
        assert!(
            BoundingBox::from_extent(vec3(1.0, 1.0, 1.0), vec3(2.0, 2.0, 2.0))
                .interesects_sphere(BoundingSphere::new(vec3(1.0, 1.0, 1.0), 5.0))
        );
    }

    #[test]
    fn bbox_on_forward_plane() {
        let plane = Plane::new(vec3(0.0, 0.0, 1.0), Vec3::Z);
        assert!(
            BoundingBox::from_extent(vec3(0.0, 0.0, 3.0), vec3(1.0, 1.0, 1.0))
                .is_on_or_forward_plane(plane)
        );
    }

    #[test]
    fn bbox_not_on_forward_plane() {
        let plane = Plane::new(vec3(0.0, 0.0, 1.0), Vec3::Z);
        assert!(
            !BoundingBox::from_extent(vec3(0.0, 0.0, -3.0), vec3(1.0, 1.0, 1.0))
                .is_on_or_forward_plane(plane)
        );
    }

    #[test]
    fn bbox_in_frustum() {
        let frustum = PerspectiveCamera::new(
            vec3(0.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Y,
            FRAC_PI_2,
            1.0,
            1.0,
            10.0,
        )
        .frustum();
        assert!(BoundingBox::from_extent(vec3(0.0, 0.0, 5.0), Vec3::ONE).is_visible(&frustum));
        assert!(
            BoundingBox::from_extent(vec3(0.0, 0.0, 0.0), vec3(2.0, 2.0, 2.0)).is_visible(&frustum)
        );
        assert!(
            BoundingBox::from_extent(vec3(0.0, 0.0, 10.0), vec3(2.0, 2.0, 2.0))
                .is_visible(&frustum)
        );
        assert!(
            BoundingBox::from_extent(vec3(3.5, 0.0, 3.5), vec3(1.0, 1.0, 1.0)).is_visible(&frustum)
        );
    }

    #[test]
    fn bbox_not_in_frustum() {
        let frustum = PerspectiveCamera::new(
            vec3(0.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Y,
            FRAC_PI_2,
            1.0,
            1.0,
            10.0,
        )
        .frustum();
        assert!(!BoundingBox::from_extent(vec3(0.0, 0.0, -1.0), Vec3::ONE).is_visible(&frustum));
        assert!(!BoundingBox::from_extent(vec3(0.0, 0.0, 12.0), Vec3::ONE).is_visible(&frustum));
        assert!(!BoundingBox::from_extent(vec3(0.0, 7.0, 5.0), Vec3::ONE).is_visible(&frustum));
        assert!(!BoundingBox::from_extent(vec3(0.0, -7.0, 5.0), Vec3::ONE).is_visible(&frustum));
        assert!(!BoundingBox::from_extent(vec3(7.0, 0.0, 5.0), Vec3::ONE).is_visible(&frustum));
        assert!(!BoundingBox::from_extent(vec3(-7.0, 0.0, 5.0), Vec3::ONE).is_visible(&frustum));
    }
}
