#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc};

use kiri::{run_game, GameClient};
use kiri_assets::GltfSceneSource;
use kiri_backend::{
    ClearRenderTarget, Format, ImageLayout, RenderDevice, RenderPassHandle, RenderPassLayout,
    RenderTarget, RenderTargetDesc, SubpassLayout,
};
use kiri_gfx::ResourceManager;

#[derive(Debug)]
struct Loop {
    render_pass: RenderPassHandle,
    _cache: Arc<ResourceManager>,
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
    fn new(render_device: &Arc<RenderDevice>) -> Result<Self, kiri::GameError<LoopError>> {
        let layout = RenderPassLayout::default()
            .color_target(
                RenderTargetDesc::new(Format::BGRA8_UNORM)
                    .clear_input()
                    .store_output()
                    .initial_layout(ImageLayout::Undefined)
                    .final_layout(ImageLayout::Present),
            )
            .subpass(SubpassLayout::default().color_write(&[0]));
        let cache = ResourceManager::new(render_device)?;
        cache.get_or_load_scene(GltfSceneSource::new("PBR/gun.gltf"))?;
        cache.get_or_load_scene(GltfSceneSource::new("ABeautifulGame/ABeautifulGame.gltf"))?;
        Ok(Self {
            render_pass: render_device.create_render_pass(layout)?,
            _cache: cache,
        })
    }
    fn info() -> (&'static str, &'static str, &'static str) {
        ("com", "gigablaster", "kiri-demo-loop")
    }

    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, _time: kiri_common::GameTime) -> Result<kiri::GameTickState, LoopError> {
        Ok(kiri::GameTickState::Continue)
    }

    fn draw(
        &self,
        _time: kiri_common::GameTime,
        context: &kiri_backend::RenderContext,
    ) -> Result<(), kiri_backend::Error> {
        let targets = [RenderTarget::color(context.backbuffer)
            .clear(ClearRenderTarget::Color([0.25, 0.25, 0.75, 1.0]))];
        let recorder = context.record(self.render_pass, 0, &targets);
        recorder.finish();
        Ok(())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game::<LoopError, Loop>().unwrap();
}
