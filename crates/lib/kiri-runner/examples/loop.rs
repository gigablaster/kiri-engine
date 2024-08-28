#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{error::Error, fmt::Display, sync::Arc};

use glam::{vec3, Affine3A, Mat4, Vec3};
use kiri::{Camera, NodeHandle, PipelineCache, ResourceCache, Scene, SceneRenderer};
use kiri_common::Handle;
use kiri_gfx::{ImageHandle, RenderContext, Renderer};
use kiri_runner::{run_game, GameClient, GameError, GameTickState};

#[derive(Debug)]
struct Loop {
    resources: Arc<ResourceCache>,
    _pipelines: Arc<PipelineCache>,
    render: SceneRenderer,
    scene: Scene,
    root: NodeHandle,
    time: f32,
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
        let resources = ResourceCache::new(renderer)?;
        let _pipelines = PipelineCache::new(renderer);
        let test = resources.get_or_load_scene("FlightHelmet/FlightHelmet.gltf")?;
        let mut scene = Scene::default();
        let root = scene.add_node(
            Handle::default(),
            kiri::NodeData::Scene(test),
            glam::Affine3A::IDENTITY,
        );
        scene.add_node(
            root,
            kiri::NodeData::Scene(test),
            Affine3A::from_translation(Vec3::new(-0.5, 0.0, 0.0)),
        );
        scene.add_node(
            root,
            kiri::NodeData::Scene(test),
            Affine3A::from_translation(Vec3::new(0.5, 0.0, 0.0)),
        );
        let render = SceneRenderer::new(&resources, &_pipelines)?;
        Ok(Self {
            resources,
            _pipelines,
            scene,
            render,
            root,
            time: 0.0,
        })
    }
    fn title(&self) -> &str {
        "Loop Demo"
    }

    fn update(&mut self, time: kiri_common::GameTime) -> Result<GameTickState, LoopError> {
        self.scene
            .update_node_transform(self.root, Affine3A::from_rotation_y(self.time * 0.5));
        self.time += time.delta_time;
        self.scene.update(&self.resources.resolve());

        Ok(GameTickState::Continue)
    }

    fn render(&self, _time: kiri_common::GameTime, context: &RenderContext) -> ImageHandle {
        self.resources.tick().unwrap();
        let camera = Camera {
            view: Mat4::look_at_lh(vec3(0.0, 0.5, 1.5), Vec3::Y * 0.25, Vec3::NEG_Y),
            projection: Mat4::perspective_lh(1.0, context.backbuffer.desc.aspect(), 0.001, 10.0),
        };
        self.render.render(&self.scene, camera, context).unwrap()
    }

    fn swapchain_created(&mut self) -> Result<(), GameError<LoopError>> {
        self.render.swapchain_changed();
        Ok(())
    }
}
fn main() {
    simple_logger::init().unwrap();
    run_game::<LoopError, Loop>().unwrap();
}
