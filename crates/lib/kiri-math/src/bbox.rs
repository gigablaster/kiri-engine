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

    pub fn from_points(points: &[Vec3]) -> Self {
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

    pub fn center(self) -> Vec3 {
        self.min.midpoint(self.max).into()
    }

    pub fn extents(self) -> Vec3 {
        (self.max - self.min).into()
    }

    pub fn transform(self, transform: Affine3A) -> Self {
        let min = transform.transform_point3a(self.min);
        let max = transform.transform_point3a(self.max);
        let mi = min.min(max);
        let ma = min.max(max);
        Self { min: mi, max: ma }
    }

    pub fn contains_point3(self, point: Vec3) -> bool {
        self.contains_point3a(point.into())
    }

    pub fn contains_point3a(self, point: Vec3A) -> bool {
        let min = self.min.cmple(point);
        let max = self.max.cmpge(point);
        (min & max).all()
    }
}

#[cfg(test)]
mod test {
    use glam::{vec3, Affine3A, Vec3};

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
}
