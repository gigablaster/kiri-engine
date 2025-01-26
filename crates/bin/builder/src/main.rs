use std::collections::HashSet;
use std::io::Cursor;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::{
    fs::{self, File},
    io::{self, Write},
    path::Path,
    sync::{atomic::AtomicBool, Arc},
    thread,
    time::Duration,
};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use clap::{Arg, ArgAction};
use kiri_asset_pipeline::{
    AssetPipelineContext, AssetSource, GlslShaderSource, ImageSource, ImportAsset, ModelSource,
};
use kiri_assets::{save_asset, Asset, CompiledAssetPath};
use kiri_vfs::{COMPILED_ASSETS_PATH, SOURCE_ASSETS_PATH};
use log::{error, info};
use notify::{RecursiveMode, Watcher};
use parking_lot::Mutex;

struct ContentProcessor {
    images: Mutex<HashSet<ImageSource>>,
    scenes: Mutex<HashSet<ModelSource>>,
    shaders: Mutex<HashSet<GlslShaderSource>>,
}

unsafe impl Send for ContentProcessor {}
unsafe impl Sync for ContentProcessor {}

fn compiled_asset_change_time(path: &CompiledAssetPath) -> Option<SystemTime> {
    let path = Path::new(COMPILED_ASSETS_PATH).join(path);
    if let Ok(metadata) = path.metadata() {
        if let Ok(changed) = metadata.modified() {
            Some(changed)
        } else if let Ok(created) = metadata.created() {
            Some(created)
        } else {
            None
        }
    } else {
        None
    }
}

fn asset_need_rebuild(asset: &impl AssetSource) -> bool {
    if let Ok(compiled) = asset.source().compiled() {
        if let Some(timestamp) = compiled_asset_change_time(&compiled) {
            return asset.changed(timestamp);
        }
    }
    true
}

fn save(path: &CompiledAssetPath, data: &[u8]) -> io::Result<()> {
    let path = Path::new(COMPILED_ASSETS_PATH).join(path);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut file = File::create(path)?;
    file.write_all(data)?;
    Ok(())
}

impl AssetPipelineContext for ContentProcessor {
    fn import_image(&self, image: ImageSource) -> CompiledAssetPath {
        let compiled = image.source.compiled().unwrap();
        self.images.lock().insert(image);
        compiled
    }
}

impl ContentProcessor {
    pub fn new() -> Self {
        Self {
            images: Default::default(),
            scenes: Default::default(),
            shaders: Default::default(),
        }
    }

    fn import_scene(&self, source: ModelSource) {
        self.scenes.lock().insert(source);
    }

    fn import_shader(&self, source: GlslShaderSource) {
        self.shaders.lock().insert(source);
    }

    async fn build_scene(&self, scene: ModelSource) {
        info!("Building scene {:?}", scene);
        if let Err(err) = self.build_asset(scene.clone()) {
            error!("Failed to build scene {:?}: {}", scene.source(), err);
        }
    }

    async fn build_image(&self, image: ImageSource) {
        info!("Building image {:?}", image);
        if let Err(err) = self.build_asset(image.clone()) {
            error!("Failed to build image {:?}: {}", image.source(), err);
        }
    }

    async fn build_shader(&self, shader: GlslShaderSource) {
        info!("Compile shader {:?}", shader);
        if let Err(err) = self.build_asset(shader.clone()) {
            error!("Failed to compiled shader {:?}:\n{}", shader.source(), err);
        }
    }

    pub fn process(self) {
        AsyncComputeTaskPool::get().scope(|s| {
            for scene in self.scenes.lock().iter() {
                if self.asset_need_rebuild(scene) {
                    s.spawn(self.build_scene(scene.clone()))
                }
            }
        });

        AsyncComputeTaskPool::get().scope(|s| {
            for image in self.images.lock().iter() {
                if self.asset_need_rebuild(image) {
                    s.spawn(self.build_image(image.clone()));
                }
            }
        });

        AsyncComputeTaskPool::get().scope(|s| {
            for shader in self.shaders.lock().iter() {
                if self.asset_need_rebuild(shader) {
                    s.spawn(self.build_shader(shader.clone()));
                }
            }
        });
    }

    fn build_asset<T: Asset, U: AssetSource + ImportAsset<T>>(
        &self,
        source: U,
    ) -> Result<(), io::Error> {
        self.write_asset(source.source().compiled()?, source.import(self)?)?;
        Ok(())
    }

    fn write_asset<T: Asset>(&self, path: CompiledAssetPath, asset: T) -> io::Result<()> {
        let mut cursor = Cursor::new(Vec::new());
        save_asset(&mut cursor, &asset)?;
        save(&path, &cursor.into_inner())?;
        Ok(())
    }

    fn asset_need_rebuild<T: AssetSource>(&self, asset: &T) -> bool {
        asset_need_rebuild(asset)
    }
}

fn collect(processor: &ContentProcessor, root: &Path) -> io::Result<()> {
    for path in fs::read_dir(root)? {
        let path = path?;
        if path.path().is_dir() {
            collect(processor, &path.path())?
        } else {
            let path = path
                .path()
                .strip_prefix(SOURCE_ASSETS_PATH)
                .unwrap()
                .to_owned();
            let path_str = path.to_str().unwrap();
            if path_str.ends_with(".gltf") {
                processor.import_scene(ModelSource::new(path));
            } else if path_str.ends_with(".vert") {
                processor.import_shader(GlslShaderSource::vertex(path));
            } else if path_str.ends_with(".frag") {
                processor.import_shader(GlslShaderSource::fragment(path));
            }
        }
    }

    Ok(())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = clap::Command::new("builder")
        .version("0.2.0")
        .author("gigablaster <gigakek@protonmail.com>")
        .about("Asset builder for kiri engine")
        .arg(
            Arg::new("watch")
                .long("watch")
                .required(false)
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("bundle")
                .long("bundle")
                .required(false)
                .action(ArgAction::SetTrue)
                .conflicts_with("watch"),
        )
        .get_matches();
    AsyncComputeTaskPool::get_or_init(TaskPool::new);

    let processor = ContentProcessor::new();
    collect(&processor, Path::new(SOURCE_ASSETS_PATH)).unwrap();
    processor.process();

    let need_reimport = Arc::new(AtomicBool::new(false));

    if args.get_flag("watch") {
        info!("Watching for changes...");
        let need_reimport2 = need_reimport.clone();
        let mut watcher = notify::recommended_watcher(move |_| {
            need_reimport2.store(true, Ordering::Release);
        })
        .unwrap();
        loop {
            watcher
                .watch(Path::new(SOURCE_ASSETS_PATH), RecursiveMode::Recursive)
                .unwrap();
            thread::sleep(Duration::from_secs(1));
            if need_reimport.load(Ordering::Acquire) {
                let processor = ContentProcessor::new();
                collect(&processor, Path::new(SOURCE_ASSETS_PATH)).unwrap();
                processor.process();
                need_reimport.store(false, Ordering::Release);
            }
        }
    }
}
