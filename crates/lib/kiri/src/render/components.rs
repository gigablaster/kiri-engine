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

use std::sync::Arc;

use bevy_ecs::component::Component;
use kiri_gfx::RenderModel;
use kiri_math::Vec3;
use kiri_resources::ModelHandle;

#[derive(Debug, Component)]
pub struct Model(pub Arc<RenderModel>);

#[derive(Debug, Component)]
pub struct PendingModel(pub ModelHandle);

#[derive(Debug, Default, Clone, Copy, Component)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
}

#[derive(Debug, Default, Clone, Copy, Component)]
pub struct PerspectiveCamera {
    pub fov: f32,
    pub znear: f32,
    pub zfar: f32,
}
