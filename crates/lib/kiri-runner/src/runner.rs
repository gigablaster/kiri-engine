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
    marker::PhantomData,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::{GameClient, GameError, GameTickState};
use bevy_tasks::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool, TaskPool};
use kiri_backend::{InstanceBuilder, PhysicalDeviceType, RenderDevice, Surface, Swapchain};
use kiri_common::TimeFilter;
use kiri_gfx::{FrameState, RenderTargetPool, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowButtons},
};

impl<E: Error> From<String> for GameError<E> {
    fn from(value: String) -> Self {
        Self::LoopError(value)
    }
}

struct RenderSystem<E: Error> {
    window: Window,
    surface: Surface,
    renderer: Arc<Renderer>,
    pool: RenderTargetPool,
    _phantom: PhantomData<E>,
}

impl<E: Error> RenderSystem<E> {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, GameError<E>> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_inner_size(PhysicalSize::new(1280, 720))
                    .with_enabled_buttons(WindowButtons::CLOSE | WindowButtons::MINIMIZE),
            )
            .map_err(|x| GameError::LoopError(x.to_string()))?;
        let instance = InstanceBuilder::new(window.display_handle().unwrap().as_raw())
            .debug(true)
            .build()?;
        let surface = Surface::new(&instance, window.window_handle().unwrap().as_raw()).unwrap();
        let device = RenderDevice::new(
            &instance,
            &surface,
            &[PhysicalDeviceType::Discrete, PhysicalDeviceType::Integrated],
        )?;
        let renderer = Renderer::new(&device)?;
        let pool = RenderTargetPool::new(&renderer);
        Ok(Self {
            window,
            surface,
            renderer,
            pool,
            _phantom: PhantomData,
        })
    }
}

struct GameApp<G, E>
where
    G: GameClient<E>,
    E: Error,
{
    game: Option<G>,
    swapchain: Option<Swapchain>,
    render_system: Option<RenderSystem<E>>,
    time: TimeFilter,
    last_timestamp: Instant,
    _phantom: PhantomData<E>,
}

impl<G, E> ApplicationHandler for GameApp<G, E>
where
    G: GameClient<E>,
    E: Error,
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.render_system.is_some() {
            self.last_timestamp = Instant::now();
            return;
        }
        let render_system = RenderSystem::new(event_loop).unwrap();

        self.game = Some(G::create(&render_system.renderer).unwrap());
        self.render_system = Some(render_system);
        self.time = TimeFilter::default();
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(..) => self.swapchain = None,
            WindowEvent::RedrawRequested => {
                let render_system = self.render_system.as_ref().unwrap();
                let game = self.game.as_mut().unwrap();
                let timestamp = Instant::now();
                self.time.sample(timestamp - self.last_timestamp);
                let dt = self.time.game_time();
                self.last_timestamp = timestamp;
                if game.update(dt).unwrap() == GameTickState::Exit {
                    event_loop.exit();
                    return;
                }
                let width = render_system.window.inner_size().width;
                let height = render_system.window.inner_size().height;
                if width > 0 && height > 0 {
                    let swapchain = self.swapchain.get_or_insert_with(|| {
                        render_system.pool.purge();
                        Swapchain::new(
                            &render_system.renderer.device,
                            &render_system.surface,
                            [width, height],
                        )
                        .unwrap()
                    });
                    if FrameState::NeedRecreateSwapchain
                        == render_system
                            .renderer
                            .render(swapchain, |context| {
                                game.render(dt, context, &render_system.pool)
                            })
                            .unwrap()
                    {
                        self.swapchain = None;
                    }
                } else {
                    thread::sleep(Duration::from_millis(30));
                }
                render_system.window.request_redraw();
            }
            _ => (),
        }
    }

    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.swapchain = None;
        self.render_system = None;
        self.game = None;
    }
}

impl<G, E> Default for GameApp<G, E>
where
    G: GameClient<E>,
    E: Error,
{
    fn default() -> Self {
        Self {
            game: None,
            render_system: None,
            swapchain: None,
            time: TimeFilter::default(),
            last_timestamp: Instant::now(),
            _phantom: PhantomData,
        }
    }
}

pub fn run_game<E: Error, G: GameClient<E>>() {
    ComputeTaskPool::get_or_init(TaskPool::new);
    AsyncComputeTaskPool::get_or_init(TaskPool::new);
    IoTaskPool::get_or_init(TaskPool::new);

    let even_loop = EventLoop::new().unwrap();
    even_loop.run_app(&mut GameApp::<G, E>::default()).unwrap();
}
