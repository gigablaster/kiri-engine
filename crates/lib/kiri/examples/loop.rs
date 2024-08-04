use std::{error::Error, fmt::Display};

use kiri::{run_game, GameClient};

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
        context: &kiri_gfx::FrameRecordContext<'a>,
    ) -> Result<(), kiri_gfx::Error> {
        Ok(())
    }
}
fn main() {
    let logger = simple_logger::init().unwrap();
    run_game(Loop::default()).unwrap();
}
