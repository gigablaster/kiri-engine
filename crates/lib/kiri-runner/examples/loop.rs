#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc, thread, time::Duration};

use bevy_ecs::world::World;
use kiri::{
    render::{
        render_world, DirectionalLight, HemisphericalLight, PendingModel, PerspectiveCamera,
        Postprocess,
    },
    Transform,
};
use kiri_gfx::{RenderContext, RenderTargetPool, Renderer};
use kiri_math::{vec3, Affine3A, Vec3};
use kiri_resources::{ResourceLoader, ResourceManager};
use kiri_runner::{run_game, GameClient, GameError, GameTickState};

#[derive(Debug)]
struct Loop {
    resources: Arc<ResourceManager>,
    world: World,
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
        let mut world = World::new();
        world.spawn((
            Transform::default(),
            PendingModel(resources.get_or_load_model("FlightHelmet/FlightHelmet.gltf")),
        ));
        world.spawn(DirectionalLight {
            direction: Vec3::Z,
            color: vec3(1.0, 1.0, 1.0),
        });
        world.spawn((
            PerspectiveCamera {
                fov: 1.0,
                znear: 0.1,
                zfar: 100.0,
            },
            Transform(Affine3A::look_at_lh(
                vec3(1.0, 1.0, 1.0),
                Vec3::ZERO,
                Vec3::Y,
            )),
        ));
        world.insert_resource(HemisphericalLight::default());
        world.insert_resource(Postprocess::default());
        Ok(Self { resources, world })
    }
    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, _time: kiri_common::GameTime) -> Result<GameTickState, LoopError> {
        self.resources.tick();
        thread::sleep(Duration::from_millis(16));
        Ok(GameTickState::Continue)
    }

    fn render(
        &mut self,
        _time: kiri_common::GameTime,
        context: &RenderContext,
        pool: &RenderTargetPool,
    ) -> Result<(), kiri_gfx::Error> {
        render_world(&mut self.world, context, &self.resources, pool)
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game::<LoopError, Loop>().unwrap();
}
