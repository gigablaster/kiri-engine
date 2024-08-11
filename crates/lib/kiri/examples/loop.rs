#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display};

use kiri::{run_game, GameClient};
use kiri_backend::{ClearRenderTarget, ImageBarrier, ImageBarrierType, RenderPass, RenderTarget};

#[derive(Debug, Default)]
struct Loop {}

#[derive(Debug)]
enum LoopError {}

impl Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LoopError")
    }
}
impl Error for LoopError {}

impl GameClient<LoopError> for Loop {
    fn info(&self) -> (&str, &str, &str) {
        (&"com", &"gigablaster", "kiri-demo-loop")
    }

    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, _time: kiri_common::GameTime) -> Result<kiri::GameTickState, LoopError> {
        Ok(kiri::GameTickState::Continue)
    }

    fn draw<'a>(
        &self,
        _time: kiri_common::GameTime,
        context: &kiri_backend::FrameRecorder<'a>,
    ) -> Result<(), kiri_backend::Error> {
        let pass = RenderPass::default().color(
            RenderTarget::color(context.backbuffer)
                .clear(ClearRenderTarget::Color([1.0, 0.0, 0.0, 1.0])),
        );
        let recorder = context.record(pass);
        recorder.barriers(&[ImageBarrier::new(
            context.backbuffer,
            ImageBarrierType::DiscardRenderTarget,
        )]);
        recorder.finish();
        Ok(())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game(Loop::default()).unwrap();
}
