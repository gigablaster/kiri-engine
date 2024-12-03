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
    fn create(renderer: &Arc<Renderer>) -> Result<Self, GameError<LoopError>> {
        let resources = ResourceManager::new(renderer)?;
        let mut world = World::new();
        for x in -5..5 {
            for y in -5..5 {
                for z in -5..5 {
                    world.spawn((
                        Transform(Affine3A::from_translation(vec3(
                            0.75 * x as f32,
                            0.75 * y as f32,
                            0.75 * z as f32,
                        ))),
                        PendingModel(resources.get_or_load_model("FlightHelmet/FlightHelmet.gltf")),
                    ));
                }
            }
        }
        world.spawn(DirectionalLight {
            direction: Vec3::Z,
            color: vec3(15.0, 10.0, 13.0),
        });
        world.spawn((
            PerspectiveCamera {
                fov: 1.0,
                znear: 0.1,
                zfar: 100.0,
            },
            Transform(Affine3A::look_at_lh(
                vec3(0.0, 0.5, 2.25),
                Vec3::ZERO,
                Vec3::Y,
            )),
        ));
        world.insert_resource(HemisphericalLight {
            top: vec3(1.0, 1.0, 1.5),
            middle: vec3(1.0, 0.5, 0.5),
            bottom: vec3(0.5, 1.0, 0.5),
        });
        world.insert_resource(Postprocess { expouse: 0.2 });
        Ok(Self { resources, world })
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
    let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    eprintln!("Serving demo profile data on {server_addr}. Run `puffin_viewer` to view it.");
    puffin::set_scopes_on(true);

    run_game::<LoopError, Loop>();
}
