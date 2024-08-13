use std::sync::atomic::Ordering;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, Write},
    path::Path,
    sync::{atomic::AtomicBool, Arc},
    thread,
    time::{Duration, SystemTime},
};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use clap::{Arg, ArgAction};
use kiri_assets::{
    get_cached_asset_path, Asset, AssetImportContext, AssetReference, AssetSource, Error,
    GltfAsset, GltfMeshSource, GltfSceneSource, ImageAsset, ImageAssetSource, ImportAsset,
    MeshAssetBuilder, ROOT_DATA_PATH,
};
use log::{error, info};
use notify::{RecursiveMode, Watcher};
use parking_lot::Mutex;

#[derive(Debug, Default)]
struct ContentProcessor {
    images: Mutex<HashMap<AssetReference, ImageAssetSource>>,
    meshes: Mutex<HashMap<AssetReference, (GltfMeshSource, MeshAssetBuilder)>>,
    scenes: Mutex<HashMap<AssetReference, GltfSceneSource>>,
}

unsafe impl Send for ContentProcessor {}
unsafe impl Sync for ContentProcessor {}

impl AssetImportContext for ContentProcessor {
    fn import_image(&self, source: ImageAssetSource) -> AssetReference {
        let reference = source.reference();
        self.images.lock().entry(reference).or_insert(source);
        reference
    }

    fn import_static_mesh(&self, source: GltfMeshSource, data: MeshAssetBuilder) -> AssetReference {
        let referene = source.reference();
        self.meshes.lock().entry(referene).or_insert((source, data));
        referene
    }

    fn import_scene(&self, source: GltfSceneSource) -> AssetReference {
        let reference = source.reference();
        self.scenes.lock().entry(reference).or_insert(source);
        reference
    }
}

fn get_cached_asset_change_time(reference: AssetReference) -> Option<SystemTime> {
    let path = get_cached_asset_path(reference);
    if path.exists() {
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                return Some(modified);
            }
            if let Ok(created) = metadata.created() {
                return Some(created);
            }
        }
    }
    None
}

fn asset_need_rebuild<T: AssetSource>(asset: &T) -> bool {
    let reference = asset.reference();
    if let Some(last_update) = get_cached_asset_change_time(reference) {
        asset.changed(last_update)
    } else {
        true
    }
}

impl ContentProcessor {
    async fn build_scene(&self, scene: GltfSceneSource) {
        info!("Building scene {:?}", scene);
        if let Err(err) = self.build_asset::<GltfAsset, GltfSceneSource>(scene.clone()) {
            error!("Failed to build scene {:?}: {}", scene, err);
        }
    }

    async fn build_mesh(&self, source: GltfMeshSource, data: MeshAssetBuilder) {
        info!("Building mesh {:?}", source);
        let mesh = data.build();
        if let Err(err) = self.write_asset(source.reference(), mesh) {
            error!("Failed to write mesh {:?}: {}", source, err);
        }
    }

    async fn build_image(&self, image: ImageAssetSource) {
        info!("Building image {:?}", image);
        if let Err(err) = self.build_asset::<ImageAsset, ImageAssetSource>(image.clone()) {
            error!("Failed to build image {:?}: {}", image, err);
        }
    }

    pub fn process(&self) {
        AsyncComputeTaskPool::get().scope(|s| {
            for (_, scene) in self.scenes.lock().iter() {
                if asset_need_rebuild(scene) {
                    s.spawn(self.build_scene(scene.clone()))
                }
            }
        });

        AsyncComputeTaskPool::get().scope(|s| {
            for (_, (source, data)) in self.meshes.lock().drain() {
                s.spawn(self.build_mesh(source, data))
            }
        });

        AsyncComputeTaskPool::get().scope(|s| {
            for (_, image) in self.images.lock().iter() {
                if asset_need_rebuild(image) {
                    s.spawn(self.build_image(image.clone()));
                }
            }
        });
    }

    fn build_asset<T: ImportAsset<U>, U: AssetSource>(&self, source: U) -> Result<(), Error> {
        self.write_asset(source.reference(), T::import(source, self)?)?;
        Ok(())
    }

    fn write_asset<T: Asset>(&self, reference: AssetReference, asset: T) -> io::Result<()> {
        let mut file = File::create(get_cached_asset_path(reference))?;
        file.write_all(&asset.save()?)?;
        Ok(())
    }
}

fn collect(processor: &ContentProcessor, root: &Path) -> io::Result<()> {
    for path in fs::read_dir(root)? {
        let path = path?;
        if path.path().is_dir() {
            collect(processor, &path.path())?
        } else {
            let path = path.path().strip_prefix(ROOT_DATA_PATH).unwrap().to_owned();
            let path_str = path.to_str().unwrap().replace('\\', "/");
            if path_str.ends_with(".gltf") {
                processor.import_scene(GltfSceneSource::new(path_str));
            }
            // else if path_str.ends_with("_ps.hlsl") {
            //     processor.import(Box::new(ShaderSource::fragment(path_str)));
            // } else if path_str.ends_with("_vs.hlsl") {
            //     processor.import(Box::new(ShaderSource::vertex(path_str)));
            // } else if path_str.ends_with("_cs.hlsl") {
            //     processor.import(Box::new(ShaderSource::compute(path_str)));
            // }
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
        .get_matches();
    AsyncComputeTaskPool::get_or_init(TaskPool::new);
    let processor = ContentProcessor::default();
    collect(&processor, Path::new(ROOT_DATA_PATH)).unwrap();
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
                .watch(Path::new(ROOT_DATA_PATH), RecursiveMode::Recursive)
                .unwrap();
            thread::sleep(Duration::from_secs(1));
            if need_reimport.load(Ordering::Acquire) {
                let processor = ContentProcessor::default();
                collect(&processor, Path::new(ROOT_DATA_PATH)).unwrap();
                processor.process();
                need_reimport.store(false, Ordering::Release);
            }
        }
    }
}
