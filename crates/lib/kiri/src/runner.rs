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
use kiri_backend::{FrameState, InstanceBuilder, PhysicalDeviceType, Surface, Swapchain};
use kiri_common::TimeFilter;
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};
use sdl2::{
    event::Event,
    keyboard::{Keycode, Mod},
    video::{FullscreenType, WindowBuildError},
};

use crate::{GameClient, GameTickState};

#[derive(Debug, thiserror::Error)]
pub enum GameError<E: Error> {
    GameFailure(E),
    GraphicsFailure(kiri_backend::Error),
    SdlError(String),
}

impl<E: Error> From<kiri_backend::Error> for GameError<E> {
    fn from(value: kiri_backend::Error) -> Self {
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
    let mut window = video
        .window(game.title(), 1280, 720)
        .allow_highdpi()
        .position_centered()
        .vulkan()
        .build()?;
    let instance = InstanceBuilder::new(window.display_handle()?.as_raw(), game.info());
    #[cfg(debug_assertions)]
    let instance = instance.debug(true);
    let instance = instance.build()?;
    let surface = Surface::new(&instance, window.window_handle()?.as_raw())?;
    let var_name = [PhysicalDeviceType::Discrete, PhysicalDeviceType::Integrated];
    let context = instance.create_context(&surface, &var_name)?;
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
                Event::KeyDown {
                    keycode: Some(Keycode::RETURN),
                    keymod: Mod::LALTMOD,
                    ..
                } => {
                    if window.fullscreen_state() == FullscreenType::Desktop {
                        window.set_fullscreen(FullscreenType::Off)?;
                    } else {
                        window.set_fullscreen(FullscreenType::Desktop)?;
                    }
                }
                _ => {}
            }
        }
        let new_time = timer.performance_counter();
        let dt =
            game_time.sample((new_time - last_time) as f64 / timer.performance_frequency() as f64);
        if let GameTickState::Exit = game.update(dt).map_err(|e| GameError::GameFailure(e))? {
            break 'main;
        }
        let size = window.vulkan_drawable_size();
        if size.0 > 0 && size.1 > 0 {
            if let Some(swapchain_frame) = &swapchain {
                if let FrameState::NeedRecreateSwapchain =
                    context.frame(swapchain_frame, |context| game.draw(dt, context))?
                {
                    swapchain = None
                }
            } else {
                swapchain = Some(Swapchain::new(&context, &surface, [size.0, size.1])?);
            }
        }
    }
    Ok(())
}
