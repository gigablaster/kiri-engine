#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc};

use ash::vk;
use kiri::{ImagePool, ResourceCache, ResourceLoader};
use kiri_backend::{ImageAttachmentDesc, RenderPassLayout, SubpassLayout};
use kiri_gfx::{ImageHandle, RenderContext, RenderPassHandle, RenderTarget, Renderer};
use kiri_runner::{run_game, GameClient, GameError, GameTickState};

#[derive(Debug)]
struct Loop {
    cache: Arc<ResourceCache>,
    pool: ImagePool,
    pass: RenderPassHandle,
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
        // cache.get_or_load_scene("PBR/gun.gltf")?;
        // cache.get_or_load_scene("ABeautifulGame/ABeautifulGame.gltf")?;
        let layout = RenderPassLayout {
            color: &[ImageAttachmentDesc::new(vk::Format::A2R10G10B10_UNORM_PACK32).clear_input()],
            depth: None,
            subpasses: &[SubpassLayout {
                depth_write: false,
                depth_read: false,
                color_writes: &[0],
                color_reads: &[],
            }],
        };
        Ok(Self {
            cache,
            pool: ImagePool::new(renderer),
            pass: renderer.create_render_pass(layout)?,
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
            context.backbuffer.desc.dims,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let pass = context.create_rasterizer_pass(
            "main",
            self.pass,
            &[RenderTarget::new(target.handle).clear_color([0.2, 0.2, 0.8, 1.0])],
            None,
            None,
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
