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
mod plane;
mod ray;

pub use bbox::*;
pub use bsphere::*;
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
}
