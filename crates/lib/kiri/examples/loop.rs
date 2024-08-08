use std::{error::Error, fmt::Display};

use ash::vk;
use kiri::{run_game, GameClient};
use kiri_gfx::{ClearRenderTarget, RenderPass, RenderPassLayout, RenderTarget};

#[derive(Debug, Default)]
struct Loop {}

#[derive(Debug)]
enum LoopError {
    Dummy,
}

impl Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LoopError")
    }
}
impl Error for LoopError {}

const PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color: &[vk::Format::B8G8R8A8_UNORM],
    depth: None,
};

impl GameClient<LoopError> for Loop {
    fn info(&self) -> (&str, &str, &str) {
        (&"com", &"gigablaster", "kiri-demo-loop")
    }

    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, time: kiri_common::GameTime) -> Result<kiri::GameTickState, LoopError> {
        Ok(kiri::GameTickState::Continue)
    }

    fn draw<'a>(
        &self,
        time: kiri_common::GameTime,
        context: &kiri_gfx::FrameRecorder<'a>,
    ) -> Result<(), kiri_gfx::Error> {
        let pass = RenderPass::default().color(
            RenderTarget::color(context.backbuffer)
                .clear(ClearRenderTarget::Color([1.0, 0.0, 0.0, 1.0])),
        );
        context.record(pass).finish();
        Ok(())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game(Loop::default()).unwrap();
}
