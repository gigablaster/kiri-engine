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

use std::{
    error::Error,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::{GameClient, GameError, GameTickState};
use bevy_tasks::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool, TaskPool};
use kiri_backend::{InstanceBuilder, PhysicalDeviceType, RenderDevice, Surface, Swapchain};
use kiri_common::{GameTime, TimeFilter};
use kiri_gfx::{FrameState, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::{Keycode, Mod},
    video::{FullscreenType, Window, WindowBuildError},
};

impl<E: Error> From<String> for GameError<E> {
    fn from(value: String) -> Self {
        Self::LoopError(value)
    }
}

impl<E: Error> From<WindowBuildError> for GameError<E> {
    fn from(value: WindowBuildError) -> Self {
        Self::LoopError(value.to_string())
    }
}

fn main_loop<E: Error, G: GameClient<E>>(
    sdl: &sdl2::Sdl,
    device: &Arc<RenderDevice>,
    surface: &Surface,
    window: &mut Window,
) -> Result<(), GameError<E>> {
    let mut swapchain = None;
    let renderer = Renderer::new(device)?;
    let mut game = G::new(&renderer)?;
    let mut event_pump = sdl.event_pump()?;
    let mut last_time = Instant::now();
    let mut time_filter = TimeFilter::new();
    'main: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'main,
                Event::KeyDown {
                    keycode: Some(Keycode::RETURN),
                    keymod: Mod::LALTMOD,
                    ..
                } => match window.fullscreen_state() {
                    FullscreenType::Off => window.set_fullscreen(FullscreenType::Desktop)?,
                    FullscreenType::Desktop => window.set_fullscreen(FullscreenType::Off)?,
                    _ => {}
                },
                Event::Window {
                    win_event: WindowEvent::Resized(..),
                    ..
                } => swapchain = None,
                _ => {}
            };
        }
        let (w, h) = window.vulkan_drawable_size();
        let now = Instant::now();
        time_filter.sample(now - last_time);
        last_time = now;
        if game
            .update(time_filter.game_time())
            .map_err(|err| GameError::GameFailure(err))?
            == GameTickState::Exit
        {
            break 'main;
        }
        if window.title() != game.title() {
            window
                .set_title(game.title())
                .map_err(|x| GameError::LoopError(x.to_string()))?;
        }
        if w > 0 && h > 0 {
            if swapchain.is_none() {
                swapchain = Some(Swapchain::new(device, surface, [w, h])?);
                game.swapchain_created()?;
            }
            let current_swapchain = swapchain.as_ref().unwrap();
            if FrameState::NeedRecreateSwapchain
                == renderer.render(current_swapchain, |context| {
                    game.render(GameTime::default(), context)
                })?
            {
                swapchain = None;
            }
        } else {
            // Sleep for a while to let OS to do other things
            thread::sleep(Duration::from_millis(30));
        }
    }
    Ok(())
}

pub fn run_game<E: Error, G: GameClient<E>>() -> Result<(), GameError<E>> {
    ComputeTaskPool::get_or_init(TaskPool::new);
    AsyncComputeTaskPool::get_or_init(TaskPool::new);
    IoTaskPool::get_or_init(TaskPool::new);
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let mut window = video
        .window("Engine", 1280, 720)
        .allow_highdpi()
        .position_centered()
        .vulkan()
        .build()?;
    let instance = InstanceBuilder::new(window.display_handle().unwrap().as_raw())
        .debug(true)
        .build()?;
    let surface = Surface::new(&instance, window.window_handle().unwrap().as_raw())?;
    let device = RenderDevice::new(
        &instance,
        &surface,
        &[PhysicalDeviceType::Discrete, PhysicalDeviceType::Integrated],
    )?;
    main_loop::<E, G>(&sdl, &device, &surface, &mut window)
}
