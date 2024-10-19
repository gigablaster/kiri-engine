#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc, thread, time::Duration};

use kiri_gfx::{RenderContext, Renderer};
use kiri_resources::{ResourceLoader, ResourceManager};
use kiri_runner::{run_game, GameClient, GameError, GameTickState};

#[derive(Debug)]
struct Loop {
    resources: Arc<ResourceManager>,
}

#[derive(Debug)]
enum LoopError {}

impl Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LoopError")
    }
}

impl Error for LoopError {}

impl GameClient<LoopError> for Loop {
    fn new(renderer: &Arc<Renderer>) -> Result<Self, GameError<LoopError>> {
        let resources = ResourceManager::new(renderer)?;
        resources.get_or_load_model("FlightHelmet/FlightHelmet.gltf");
        Ok(Self { resources })
    }
    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, _time: kiri_common::GameTime) -> Result<GameTickState, LoopError> {
        self.resources.tick();
        thread::sleep(Duration::from_millis(16));
        Ok(GameTickState::Continue)
    }

    fn render(&self, _time: kiri_common::GameTime, _context: &RenderContext) {
        self.resources.tick();
    }

    fn swapchain_created(&mut self) -> Result<(), GameError<LoopError>> {
        Ok(())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game::<LoopError, Loop>().unwrap();
}
