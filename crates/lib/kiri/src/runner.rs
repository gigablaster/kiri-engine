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

use std::error::Error;

use bevy_tasks::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool, TaskPoolBuilder};
use kiri_common::TimeFilter;
use kiri_gfx::{InstanceBuilder, PhysicalDeviceType, Surface, Swapchain};
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};
use sdl2::{event::Event, video::WindowBuildError};

use crate::{GameClient, GameTickState};

#[derive(Debug, thiserror::Error)]
pub enum GameError<E: Error> {
    GameFailure(E),
    GraphicsFailure(kiri_gfx::Error),
    SdlError(String),
}

impl<E: Error> From<kiri_gfx::Error> for GameError<E> {
    fn from(value: kiri_gfx::Error) -> Self {
        Self::GraphicsFailure(value)
    }
}

impl<E: Error> From<String> for GameError<E> {
    fn from(value: String) -> Self {
        Self::SdlError(value)
    }
}

impl<E: Error> From<WindowBuildError> for GameError<E> {
    fn from(value: WindowBuildError) -> Self {
        Self::SdlError(value.to_string())
    }
}

impl<E: Error> From<HandleError> for GameError<E> {
    fn from(value: HandleError) -> Self {
        Self::SdlError(value.to_string())
    }
}

pub fn run_game<E: Error, G: GameClient<E>>(game: G) -> Result<(), GameError<E>> {
    let mut game = game;
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let window = video
        .window(game.title(), 1280, 720)
        .position_centered()
        .vulkan()
        .build()?;
    let instance = InstanceBuilder::new(window.display_handle()?.as_raw());
    #[cfg(debug_assertions)]
    let instance = instance.debug(true);
    let instance = instance.build()?;
    let surface = Surface::new(&instance, window.window_handle()?.as_raw())?;
    let context = instance.create_context(
        &surface,
        &[PhysicalDeviceType::Discrete, PhysicalDeviceType::Integrated],
    )?;
    let timer = sdl.timer()?;
    let last_time = timer.performance_counter();
    let mut swapchain = None;
    let mut pump = sdl.event_pump()?;
    let mut game_time = TimeFilter::default();
    ComputeTaskPool::get_or_init(|| TaskPoolBuilder::new().build());
    AsyncComputeTaskPool::get_or_init(|| TaskPoolBuilder::new().build());
    IoTaskPool::get_or_init(|| TaskPoolBuilder::new().build());
    'main: loop {
        for event in pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'main,
                _ => {}
            }
        }
        let new_time = timer.performance_counter();
        let dt =
            game_time.sample((new_time - last_time) as f64 / timer.performance_frequency() as f64);
        match game.update(dt).map_err(|e| GameError::GameFailure(e))? {
            GameTickState::Exit => break 'main,
            _ => {}
        }
        let size = window.vulkan_drawable_size();
        if size.0 > 0 && size.1 > 0 {
            let swapchain_frame =
                swapchain.get_or_insert(Swapchain::new(&context, &surface, [size.0, size.1])?);
            match context.frame(&swapchain_frame, |context| Ok(game.draw(dt, context)?))? {
                kiri_gfx::FrameState::NeedRecreateSwapchain => swapchain = None,
                _ => {}
            }
        }
    }
    Ok(())
}
