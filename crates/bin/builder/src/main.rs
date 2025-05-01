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
use kiri_asset_pipeline::{AssetPipelineContext, ImportAsset};
use kiri_assets::{
    save_asset, Asset, AssetSource, ImageAssetSource, ModelAssetSource, ShaderAssetSource,
};
use kiri_vfs::{AssetReference, COMPILED_ASSETS_PATH, SOURCE_ASSETS_PATH};
use log::{error, info};
use notify::{RecursiveMode, Watcher};
use parking_lot::Mutex;

struct ContentProcessor {
    images: Mutex<HashSet<ImageAssetSource>>,
    scenes: Mutex<HashSet<ModelAssetSource>>,
    shaders: Mutex<HashSet<ShaderAssetSource>>,
}

unsafe impl Send for ContentProcessor {}
unsafe impl Sync for ContentProcessor {}

fn compiled_asset_change_time(reference: AssetReference) -> Option<SystemTime> {
    let path = Path::new(COMPILED_ASSETS_PATH).join(format!("{}.asset", reference));
    if let Ok(metadata) = path.metadata() {
        if let Ok(changed) = metadata.modified() {
            Some(changed)
        } else {
            metadata.created().ok()
        }
    } else {
        None
    }
}

fn asset_need_rebuild(asset: &impl AssetSource) -> bool {
    if let Some(timestamp) = compiled_asset_change_time(asset.reference()) {
        asset.changed(timestamp)
    } else {
        true
    }
}

fn save(reference: AssetReference, data: &[u8]) -> io::Result<()> {
    let path = Path::new(COMPILED_ASSETS_PATH).join(format!("{}.asset", reference));
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut file = File::create(path)?;
    file.write_all(data)?;
    Ok(())
}

impl AssetPipelineContext for ContentProcessor {
    fn import_image(&self, image: ImageAssetSource) -> AssetReference {
        let reference = image.reference();
        self.images.lock().insert(image);
        reference
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

    fn import_scene(&self, source: ModelAssetSource) {
        self.scenes.lock().insert(source);
    }

    fn import_shader(&self, source: ShaderAssetSource) {
        self.shaders.lock().insert(source);
    }

    async fn build_scene(&self, scene: ModelAssetSource) {
        info!("Building scene {:?}", scene);
        if let Err(err) = self.build_asset(scene.clone()) {
            error!("Failed to build scene {:?}: {}", scene, err);
        }
    }

    async fn build_image(&self, image: ImageAssetSource) {
        info!("Building image {:?}", image);
        if let Err(err) = self.build_asset(image.clone()) {
            error!("Failed to build image {:?}: {}", image, err);
        }
    }

    async fn build_shader(&self, shader: ShaderAssetSource) {
        info!("Compile effect {:?}", shader);
        if let Err(err) = self.build_asset(shader.clone()) {
            error!("Failed to compiled effect {:?}:\n{}", shader, err);
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
        self.write_asset(source.reference(), source.import(self)?)?;
        Ok(())
    }

    fn write_asset<T: Asset>(&self, reference: AssetReference, asset: T) -> io::Result<()> {
        let mut cursor = Cursor::new(Vec::new());
        save_asset(&mut cursor, &asset)?;
        save(reference, &cursor.into_inner())?;
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
                processor.import_scene(ModelAssetSource::new(path_str));
            } else if path_str.ends_with(".vert") {
                processor.import_shader(ShaderAssetSource::vertex(path_str));
            } else if path_str.ends_with(".frag") {
                processor.import_shader(ShaderAssetSource::fragment(path_str));
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
