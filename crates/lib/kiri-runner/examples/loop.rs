#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc};

use ash::vk;
use kiri::{ResourceCache, ResourceLoader};
use kiri_backend::AttachmentClearValue;
use kiri_gfx::{ImageBarrierType, RenderContext, RenderTarget, Renderer};
use kiri_runner::{run_game, GameClient, GameError, GameTickState};

#[derive(Debug)]
struct Loop {
    _cache: Arc<ResourceCache>,
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
        let cache = ResourceCache::new(renderer)?;
        cache.get_or_load_scene("PBR/gun.gltf")?;
        cache.get_or_load_scene("ABeautifulGame/ABeautifulGame.gltf")?;
        Ok(Self { _cache: cache })
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
        context: &RenderContext,
    ) -> Result<(), kiri_gfx::Error> {
        let mut pass = context.create_render_pass(
            &[RenderTarget::color(context.target)
                .clear(AttachmentClearValue::Color([0.1, 0.1, 0.9, 1.0]))],
            None,
        );
        pass.image_barrier(
            context.target,
            ImageBarrierType::DiscardToWriteColor,
            vk::ImageAspectFlags::COLOR,
        );
        context.submit(pass.build());
        Ok(())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game::<LoopError, Loop>().unwrap();
}
