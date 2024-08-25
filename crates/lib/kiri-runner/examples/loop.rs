#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc};

use ash::vk;
use kiri::{ImagePool, ResourceCache, ResourceLoader};
use kiri_backend::RenderTargetClearValue;
use kiri_gfx::{ImageBarrierType, ImageHandle, RenderContext, RenderTarget, Renderer};
use kiri_runner::{run_game, GameClient, GameError, GameTickState};

#[derive(Debug)]
struct Loop {
    cache: Arc<ResourceCache>,
    pool: ImagePool,
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
        Ok(Self {
            cache,
            pool: ImagePool::new(renderer),
        })
    }
    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, _time: kiri_common::GameTime) -> Result<GameTickState, LoopError> {
        self.cache.tick();
        Ok(GameTickState::Continue)
    }

    fn render(
        &self,
        _time: kiri_common::GameTime,
        context: &RenderContext,
    ) -> Result<ImageHandle, kiri_gfx::Error> {
        let target = self.pool.get(
            vk::Format::A2R10G10B10_UNORM_PACK32,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            context.backbuffer_dims,
        )?;
        let mut pass = context.create_render_pass(
            "main",
            &[RenderTarget::color(target.handle)
                .clear(RenderTargetClearValue::Color([0.1, 0.1, 0.9, 1.0]))],
            None,
        );
        pass.image_barrier(
            target.handle,
            ImageBarrierType::DiscardToWriteColor,
            vk::ImageAspectFlags::COLOR,
        );
        context.submit(pass.build());
        Ok(target.handle)
    }

    fn swapchain_created(&mut self) -> Result<(), GameError<LoopError>> {
        self.pool.purge();
        Ok(())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game::<LoopError, Loop>().unwrap();
}
