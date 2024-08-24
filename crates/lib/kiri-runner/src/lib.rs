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

use kiri_common::GameTime;
use kiri_gfx::{ImageHandle, RenderContext, Renderer};
pub use runner::*;

pub enum GameTickState {
    Continue,
    Exit,
}

#[derive(Debug, thiserror::Error)]
pub enum GameError<E: Error> {
    GameFailure(E),
    BackendFailure(#[from] kiri_backend::Error),
    GfxError(kiri_gfx::Error),
    EngineError(kiri::Error),
    LoopError(String),
}

impl<E: Error> From<kiri::Error> for GameError<E> {
    fn from(value: kiri::Error) -> Self {
        match value {
            kiri::Error::BackendError(err) => Self::BackendFailure(err),
            kiri::Error::RendererError(err) => Self::GfxError(err),
            err => Self::EngineError(err),
        }
    }
}

impl<E: Error> From<kiri_gfx::Error> for GameError<E> {
    fn from(value: kiri_gfx::Error) -> Self {
        match value {
            kiri_gfx::Error::BackendError(err) => Self::BackendFailure(err),
            err => Self::GfxError(err),
        }
    }
}

pub trait GameClient<E: Error>: Sized + Send + Sync {
    fn new(renderer: &Arc<Renderer>) -> Result<Self, GameError<E>>;
    fn title(&self) -> &str;
    fn update(&mut self, time: GameTime) -> Result<GameTickState, E>;
    fn swapchain_created(&mut self) -> Result<(), GameError<E>>;
    fn render(
        &self,
        time: GameTime,
        context: &RenderContext,
    ) -> Result<ImageHandle, kiri_gfx::Error>;
    fn resumed(&mut self) -> Result<(), GameError<E>> {
        Ok(())
    }
    fn suspended(&mut self) -> Result<(), GameError<E>> {
        Ok(())
    }
}
