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

use kiri_backend::{RenderContext, RenderDevice};
use kiri_common::GameTime;
pub use runner::*;

pub enum GameTickState {
    Continue,
    Exit,
}

#[derive(Debug, thiserror::Error)]
pub enum GameError<E: Error> {
    GameFailure(E),
    GraphicsFailure(kiri_backend::Error),
    LoopError(String),
    GfxError(kiri_gfx::Error),
}

impl<E: Error> From<kiri_backend::Error> for GameError<E> {
    fn from(value: kiri_backend::Error) -> Self {
        Self::GraphicsFailure(value)
    }
}

impl<E: Error> From<String> for GameError<E> {
    fn from(value: String) -> Self {
        Self::LoopError(value)
    }
}

impl<E: Error> From<kiri_gfx::Error> for GameError<E> {
    fn from(value: kiri_gfx::Error) -> Self {
        match value {
            kiri_gfx::Error::BackendError(err) => Self::GraphicsFailure(err),
            err => Self::GfxError(err),
        }
    }
}

pub trait GameClient<E: Error>: Sized + Send + Sync {
    fn new(render_device: &Arc<RenderDevice>) -> Result<Self, GameError<E>>;
    fn info() -> (&'static str, &'static str, &'static str);
    fn title(&self) -> &str;
    fn update(&mut self, time: GameTime) -> Result<GameTickState, E>;
    fn draw(&self, time: GameTime, context: &RenderContext) -> Result<(), kiri_backend::Error>;
    fn resumed(&mut self) -> Result<(), GameError<E>> {
        Ok(())
    }
    fn suspended(&mut self) -> Result<(), GameError<E>> {
        Ok(())
    }
}
