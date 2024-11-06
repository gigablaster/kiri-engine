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

mod components;
mod systems;

use std::sync::Arc;

use bevy_ecs::system::Resource;
pub use components::*;
use kiri_backend::Swapchain;
use kiri_gfx::{RenderTargetPool, Renderer};
use kiri_math::Vec3;
use kiri_resources::ResourceManager;
pub use systems::*;

#[derive(Debug, Resource)]
pub struct ResourceManagerWrapper(pub Arc<ResourceManager>);

#[derive(Debug, Resource)]
pub struct RendererWrapper(pub Arc<Renderer>);

#[derive(Debug, Default, Clone, Copy, Resource)]
pub struct HemisphericalLight {
    pub top: Vec3,
    pub middle: Vec3,
    pub bottom: Vec3,
}

#[derive(Debug, Resource)]
pub struct RenderTargetPoolWrapper(pub RenderTargetPool);

#[derive(Resource)]
pub struct SwapchainWrapper(pub Arc<Swapchain>);

#[derive(Debug, Resource)]
pub struct Postprocess {
    pub expouse: f32,
}

impl Default for Postprocess {
    fn default() -> Self {
        Self { expouse: 1.0 }
    }
}
