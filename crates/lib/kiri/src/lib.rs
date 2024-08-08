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

use std::error::Error;

use kiri_common::GameTime;
use kiri_gfx::FrameRecorder;
pub use runner::*;

pub enum GameTickState {
    Continue,
    Exit,
}

pub trait GameClient<E: Error>: Default {
    fn info(&self) -> (&str, &str, &str);
    fn title(&self) -> &str;
    fn update(&mut self, time: GameTime) -> Result<GameTickState, E>;
    fn draw<'a>(&self, time: GameTime, context: &FrameRecorder<'a>) -> Result<(), kiri_gfx::Error>;
}
