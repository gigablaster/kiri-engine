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

mod runner;

use std::{error::Error, sync::Arc};

use kiri_common::{GameAppConfig, GameTime};
use kiri_gfx::{RenderContext, RenderTargetPool, Renderer};
use kiri_resources::PipelineCache;
pub use runner::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameTickState {
    Continue,
    Exit,
}

#[derive(Debug, thiserror::Error)]
pub enum GameError<E: Error> {
    GameFailure(E),
    BackendFailure(#[from] kiri_backend::Error),
    GfxError(#[from] kiri_gfx::Error),
    EngineError(#[from] kiri::Error),
    ResourceError(#[from] kiri_resources::Error),
    LoopError(String),
}

pub trait GameClient<E: Error>: Sized + Send + Sync {
    fn create(
        renderer: Arc<Renderer>,
        pipeline_cache: Arc<PipelineCache>,
    ) -> Result<Self, GameError<E>>;
    fn config() -> &'static GameAppConfig<'static>;
    fn update(&mut self, time: GameTime) -> Result<GameTickState, E>;
    fn render(
        &mut self,
        time: GameTime,
        context: &RenderContext,
        pool: &RenderTargetPool,
    ) -> Result<(), kiri_gfx::Error>;
}
