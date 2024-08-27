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

use std::{error::Error, marker::PhantomData, sync::Arc, time::Instant};

use bevy_tasks::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool, TaskPool};
use kiri_backend::{InstanceBuilder, PhysicalDeviceType, RenderDevice, Surface, Swapchain};
use kiri_common::TimeFilter;
use kiri_gfx::{FrameState, Renderer};
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    error::{EventLoopError, OsError},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowButtons, WindowId},
};

use crate::{GameClient, GameError, GameTickState};

struct InnerData<E: Error, G: GameClient<E>> {
    window: Window,
    swapchain: Option<Swapchain>,
    surface: Surface,
    device: Arc<RenderDevice>,
    renderer: Arc<Renderer>,
    _marker1: PhantomData<E>,
    _marker2: PhantomData<G>,
}

impl<E: Error> From<HandleError> for GameError<E> {
    fn from(value: HandleError) -> Self {
        Self::LoopError(value.to_string())
    }
}

impl<E: Error> From<OsError> for GameError<E> {
    fn from(value: OsError) -> Self {
        Self::LoopError(value.to_string())
    }
}

impl<E: Error> From<EventLoopError> for GameError<E> {
    fn from(value: EventLoopError) -> Self {
        Self::LoopError(value.to_string())
    }
}

impl<E: Error, G: GameClient<E>> Drop for InnerData<E, G> {
    fn drop(&mut self) {
        self.swapchain = None;
    }
}

impl<E: Error, G: GameClient<E>> InnerData<E, G> {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, GameError<E>> {
        let window = event_loop.create_window(
            WindowAttributes::default()
                .with_inner_size(PhysicalSize::new(1280, 720))
                .with_enabled_buttons(WindowButtons::CLOSE | WindowButtons::MINIMIZE),
        )?;
        let instance = InstanceBuilder::new(window.display_handle()?.display_handle()?.into())
            .debug(true)
            .build()?;
        let surface = Surface::new(&instance, window.window_handle()?.window_handle()?.into())?;
        let device = RenderDevice::new(
            &instance,
            &surface,
            &[PhysicalDeviceType::Discrete, PhysicalDeviceType::Integrated],
        )?;
        let renderer = Renderer::new(&device)?;
        Ok(Self {
            window,
            renderer,
            device,
            surface,
            swapchain: None,
            _marker1: PhantomData,
            _marker2: PhantomData,
        })
    }
}

struct GameApp<E: Error, G: GameClient<E>> {
    game: Option<G>,
    inner: Option<InnerData<E, G>>,
    last_time: Instant,
    game_time: TimeFilter,
    _marker: PhantomData<E>,
}

impl<E: Error, G: GameClient<E>> Default for GameApp<E, G> {
    fn default() -> Self {
        Self {
            game: None,
            inner: None,
            last_time: Instant::now(),
            game_time: Default::default(),
            _marker: PhantomData,
        }
    }
}

impl<E: Error, G: GameClient<E>> ApplicationHandler for GameApp<E, G> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let internal = self
            .inner
            .get_or_insert(InnerData::new(event_loop).unwrap());
        let game = self.game.get_or_insert(G::new(&internal.renderer).unwrap());
        game.resumed().unwrap();
        internal.window.set_title(game.title());
        self.last_time = Instant::now();
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(game) = self.game.as_mut() {
            game.suspended().unwrap();
        }
        if let Some(inner) = self.inner.as_mut() {
            inner.swapchain = None;
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(inner) = &mut self.inner {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                    self.inner = None;
                }
                WindowEvent::Resized(_) => inner.swapchain = None,
                WindowEvent::RedrawRequested => {
                    let dims = [
                        inner.window.inner_size().width,
                        inner.window.inner_size().height,
                    ];
                    let game = self.game.as_mut().unwrap();
                    let now = Instant::now();
                    self.game_time.sample(now - self.last_time);
                    self.last_time = now;
                    if let GameTickState::Exit = game.update(self.game_time.game_time()).unwrap() {
                        event_loop.exit();
                    }
                    if dims[0] > 0 && dims[1] > 0 {
                        if inner.swapchain.is_none() {
                            inner.swapchain =
                                Some(Swapchain::new(&inner.device, &inner.surface, dims).unwrap());
                            inner.renderer.invalidate_fbos();
                            game.swapchain_created().unwrap();
                        }
                        let swapchain = inner.swapchain.as_ref().unwrap();
                        if let FrameState::NeedRecreateSwapchain = inner
                            .renderer
                            .render(swapchain, |context| {
                                game.render(self.game_time.game_time(), context)
                            })
                            .unwrap()
                        {
                            inner.swapchain = None;
                        }

                        inner.window.request_redraw();
                    } else {
                        self.last_time = Instant::now();
                    }
                }
                _ => {}
            }
        }
    }
}

pub fn run_game<E: Error, G: GameClient<E>>() -> Result<(), GameError<E>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    ComputeTaskPool::get_or_init(TaskPool::new);
    AsyncComputeTaskPool::get_or_init(TaskPool::new);
    IoTaskPool::get_or_init(TaskPool::new);
    let mut app = GameApp::<E, G>::default();
    Ok(event_loop.run_app(&mut app)?)
}
