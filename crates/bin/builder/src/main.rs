use std::collections::HashSet;
use std::io::Cursor;
use std::sync::atomic::Ordering;
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
use kiri_assets::{
    get_compiled_asset_change_time, get_compiled_asset_path, save_asset, Asset, AssetReference,
    AssetSource, ImageAsset, ImageData, ImageSource, ImportAsset, ModelSource, ShaderAsset,
    ShaderAssetSource,
};
use kiri_vfs::ROOT_SOURCE_ASSETS_PATH;
use log::{error, info};
use notify::{RecursiveMode, Watcher};
use parking_lot::Mutex;

struct ContentProcessor {
    images: Mutex<HashSet<ImageSource>>,
    scenes: Mutex<HashSet<ModelSource>>,
    shaders: Mutex<HashSet<ShaderAssetSource>>,
}

unsafe impl Send for ContentProcessor {}
unsafe impl Sync for ContentProcessor {}

fn asset_need_rebuild(asset: &impl AssetSource) -> bool {
    let reference = asset.reference();
    if let Some(last_update) = get_compiled_asset_change_time(reference) {
        asset.changed(last_update)
    } else {
        true
    }
}

fn save(reference: AssetReference, data: &[u8]) -> io::Result<()> {
    let path = get_compiled_asset_path(reference)?;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut file = File::create(path)?;
    file.write_all(data)?;
    Ok(())
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

    fn import_shader(&self, source: ShaderAssetSource) {
        self.shaders.lock().insert(source);
    }

    async fn build_scene(&self, scene: ModelSource) {
        info!("Building scene {:?}", scene);
        if let Err(err) = self.build_scene_impl(scene.clone()) {
            error!("Failed to build scene {:?}: {}", scene, err);
        }
    }

    fn build_scene_impl(&self, scene: ModelSource) -> Result<(), io::Error> {
        let asset = scene.import()?;
        let mut images = self.images.lock();
        asset
            .collect_dependencies()
            .iter()
            .cloned()
            .for_each(|source| {
                // Don't export self-contained images
                if let ImageData::Path(_) = source.data {
                    images.insert(source.clone());
                }
            });
        Ok(self.write_asset(scene.reference(), asset)?)
    }

    async fn build_image(&self, image: ImageSource) {
        info!("Building image {:?}", image);
        if let Err(err) = self.build_asset::<ImageAsset, ImageSource>(image.clone()) {
            error!("Failed to build image {:?}: {}", image, err);
        }
    }

    async fn build_shader(&self, shader: ShaderAssetSource) {
        info!("Compile shader {:?}", shader);
        if let Err(err) = self.build_asset::<ShaderAsset, ShaderAssetSource>(shader.clone()) {
            error!("Failed to compiled shader {:?}:\n{}", shader, err);
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
        self.write_asset(source.reference(), source.import()?)?;
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
                .strip_prefix(ROOT_SOURCE_ASSETS_PATH)
                .unwrap()
                .to_owned();
            let path_str = path.to_str().unwrap().replace('\\', "/");
            if path_str.ends_with(".gltf") {
                processor.import_scene(ModelSource::new(&path_str));
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
        .version("0.1.0")
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
    collect(&processor, Path::new(ROOT_SOURCE_ASSETS_PATH)).unwrap();
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
                .watch(Path::new(ROOT_SOURCE_ASSETS_PATH), RecursiveMode::Recursive)
                .unwrap();
            thread::sleep(Duration::from_secs(1));
            if need_reimport.load(Ordering::Acquire) {
                let processor = ContentProcessor::new();
                collect(&processor, Path::new(ROOT_SOURCE_ASSETS_PATH)).unwrap();
                processor.process();
                need_reimport.store(false, Ordering::Release);
            }
        }
    }
}
