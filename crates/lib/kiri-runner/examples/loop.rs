#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc};

use ash::vk;
use kiri::ResourceManager;
use kiri_backend::{Image, ImageCreateDesc};
use kiri_gfx::{BindlessManager, RenderContext};
use kiri_runner::{run_game, GameClient, GameError, GameTickState};

#[derive(Debug)]
struct Loop {
    image: Arc<Image>,
    resource_manager: Arc<ResourceManager>,
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
    fn new(renderer: &Arc<BindlessManager>) -> Result<Self, GameError<LoopError>> {
        let resource_manager = ResourceManager::new(renderer)?;
        resource_manager.get_or_load_static_mesh("PBR/gun.gltf#Mesh")?;
        resource_manager.get_or_load_scene("ABeautifulGame/ABeautifulGame.gltf")?;
        Ok(Self {
            image: Arc::new(Image::new(
                &renderer.device,
                ImageCreateDesc::new(vk::Format::R8G8B8A8_UNORM, [1280, 720])
                    .trasfer_source()
                    .sampled()
                    .samples(vk::SampleCountFlags::TYPE_1),
            )?),
            resource_manager,
        })
    }
    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, _time: kiri_common::GameTime) -> Result<GameTickState, LoopError> {
        Ok(GameTickState::Continue)
    }

    fn render(
        &self,
        _time: kiri_common::GameTime,
        _context: RenderContext,
    ) -> Result<Arc<Image>, kiri_gfx::Error> {
        Ok(self.image.clone())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game::<LoopError, Loop>().unwrap();
}
