#![allow(clippy::doc_lazy_continuation)]
// Copyright (C) 2024 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
use glam::Affine3A;
use kiri_assets::NodeIndex;
use kiri_common::{Handle, HotColdPool};

use crate::{Bounds, RenderScene, SceneHandle, StaticMeshHandle, StaticRenderMesh};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeData {
    Empty,
    StaticMesh(StaticMeshHandle),
    Scene(SceneHandle),
}

impl NodeData {
    pub fn bounds<T: MeshResolver>(self, resolver: &T) -> Option<Bounds> {
        match self {
            Self::StaticMesh(handle) => resolver.resolve_static_mesh(handle).map(|x| x.bounds),
            _ => None,
        }
    }
}

pub type NodeHandle = Handle<SceneNode>;
type NodePool = HotColdPool<SceneNode, NodeIndex>;

/// Клиентские данные о ноде
#[derive(Debug, Clone, Copy)]
pub struct SceneNode {
    data: NodeData,
    parent: NodeHandle,
    transform: glam::Affine3A,
}

/// Интерфейс для получения данных из сцены
pub trait SceneCuller: Send + Sync {
    fn cull(&self, bounds: Bounds) -> bool;
}

const MAX_SCENE_NODES: usize = 0xfffff;

#[derive(Debug)]
pub struct Scene {
    nodes: NodePool,
    data: Vec<NodeData>,
    parents: Vec<NodeIndex>,
    local_transforms: Vec<glam::Affine3A>,
    world_transforms: Vec<glam::Affine3A>,
    bounds: Vec<Bounds>,
    rebuild_scene: bool,
    recalculate_transforms: bool,
    update_bounds: Vec<NodeHandle>,
}

/// Интерфейс для доступа к данным меша
pub trait MeshResolver {
    fn resolve_static_mesh(&self, handle: StaticMeshHandle) -> Option<&StaticRenderMesh>;
    fn resolve_scene(&self, handle: SceneHandle) -> Option<&RenderScene>;
}

#[derive(Debug)]
pub struct CullResult {
    pub static_meshes: Vec<(Affine3A, StaticMeshHandle)>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            nodes: NodePool::new(MAX_SCENE_NODES),
            data: Default::default(),
            parents: Default::default(),
            local_transforms: Default::default(),
            world_transforms: Default::default(),
            bounds: Default::default(),
            rebuild_scene: Default::default(),
            recalculate_transforms: Default::default(),
            update_bounds: Default::default(),
        }
    }
}

impl Scene {
    pub fn add_node(
        &mut self,
        parent: NodeHandle,
        data: NodeData,
        transform: Affine3A,
    ) -> NodeHandle {
        let handle = self.nodes.push(
            SceneNode {
                data,
                parent,
                transform,
            },
            NodeIndex::default(),
        );
        self.rebuild_scene = true;
        handle
    }

    pub fn remove_node(&mut self, handle: NodeHandle) {
        if self.nodes.remove(handle).is_some() {
            self.rebuild_scene = true;
        }
    }

    pub fn update_node_transform(&mut self, handle: NodeHandle, transform: Affine3A) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.transform = transform;
            if let Some(index) = self.nodes.get_cold(handle).unwrap().index() {
                self.local_transforms[index as usize] = transform;
                self.recalculate_transforms = true;
            }
        }
    }

    pub fn update_node_data(&mut self, handle: NodeHandle, data: NodeData) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.data = data;
            if let Some(index) = self.nodes.get_cold(handle).unwrap().index() {
                self.data[index as usize] = data;
                self.update_bounds.push(handle);
            }
        }
    }

    pub fn update<T: MeshResolver>(&mut self, resolver: &T) {
        puffin::profile_function!();
        if self.rebuild_scene {
            puffin::profile_scope!("Rebuild scene");
            let mut nodes_by_parent = self
                .nodes
                .enumerate()
                .map(|(handle, node, _)| (handle, *node))
                .collect::<Vec<_>>();
            nodes_by_parent.sort_by(|a, b| a.1.parent.cmp(&b.1.parent));
            let mut last_parent_handle = Handle::default();
            let mut last_parent = NodeIndex::default();
            self.parents.clear();
            self.local_transforms.clear();
            self.world_transforms.clear();
            self.bounds.clear();
            self.data.clear();
            let mut garbage = Vec::new();
            let mut index = 0u32;
            for (handle, node) in nodes_by_parent.into_iter() {
                let current = NodeIndex::new(index);
                if node.parent != last_parent_handle {
                    if garbage.contains(&node.parent) {
                        garbage.push(handle);
                        continue;
                    }
                    last_parent = if let Some(node_index) = self.nodes.get_cold(node.parent) {
                        *node_index
                    } else {
                        garbage.push(handle);
                        continue;
                    };
                    last_parent_handle = node.parent;
                }
                self.parents.push(last_parent);
                self.local_transforms.push(node.transform);
                self.data.push(node.data);
                self.bounds
                    .push(node.data.bounds(resolver).unwrap_or_default());
                self.world_transforms.push(Affine3A::default());
                self.nodes.replace_cold(handle, current);
                index += 1;
            }
        } else if !self.update_bounds.is_empty() {
            self.update_bounds.sort();
            self.update_bounds.dedup();
            self.update_bounds.drain(..).for_each(|handle| {
                let index = self.nodes.get_cold(handle).unwrap().index().unwrap() as usize;
                self.bounds[index] = self
                    .nodes
                    .get(handle)
                    .unwrap()
                    .data
                    .bounds(resolver)
                    .unwrap_or_default();
            })
        }
        if self.rebuild_scene || self.recalculate_transforms {
            puffin::profile_scope!("Update transforms");
            for i in 0..self.parents.len() {
                let parent_transform = self.parents[i]
                    .index()
                    .map(|x| self.world_transforms[x as usize])
                    .unwrap_or_default();
                self.world_transforms[i] = parent_transform * self.local_transforms[i];
            }
        }
        self.rebuild_scene = false;
        self.recalculate_transforms = false;
    }

    pub fn cull<T: SceneCuller, U: MeshResolver>(&self, culler: T, resolver: &U) -> CullResult {
        assert!(
            !self.rebuild_scene && self.update_bounds.is_empty() && !self.recalculate_transforms,
            "Scene must be updated before culling"
        );
        let mut static_meshes = Vec::new();
        self.data
            .iter()
            .enumerate()
            .for_each(|(index, data)| match data {
                NodeData::StaticMesh(handle) => {
                    let transform = self.world_transforms[index];
                    if culler.cull(self.bounds[index].transform(transform)) {
                        static_meshes.push((transform, *handle));
                    }
                }
                NodeData::Scene(handle) => {
                    if let Some(scene) = resolver.resolve_scene(*handle) {
                        let parent_transform = self.world_transforms[index];
                        scene
                            .node_to_mesh
                            .iter()
                            .copied()
                            .for_each(|(node_index, mesh_index)| {
                                let tranform =
                                    parent_transform * scene.world_transforms[node_index as usize];
                                if culler
                                    .cull(scene.bounds[mesh_index as usize].transform(tranform))
                                {
                                    static_meshes
                                        .push((tranform, scene.mesh_handles[mesh_index as usize]));
                                }
                            });
                    }
                }
                _ => {}
            });
        CullResult { static_meshes }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Default)]
    struct DummyResolver {}

    impl MeshResolver for DummyResolver {
        fn resolve_static_mesh(&self, _handle: StaticMeshHandle) -> Option<&StaticRenderMesh> {
            None
        }

        fn resolve_scene(&self, _handle: SceneHandle) -> Option<&RenderScene> {
            None
        }
    }

    #[test]
    fn build_scene() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeData::Empty, Affine3A::default());
        let handle1_1 = scene.add_node(
            handle1,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2_1 = scene.add_node(
            handle2,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(2.0, 2.0, 2.0)),
        );
        scene.update(&DummyResolver::default());
        assert_eq!(NodeIndex::new(0), *scene.nodes.get_cold(handle1).unwrap());
        assert_eq!(NodeIndex::new(1), *scene.nodes.get_cold(handle2).unwrap());
        assert_eq!(NodeIndex::new(2), *scene.nodes.get_cold(handle1_1).unwrap());
        assert_eq!(NodeIndex::new(3), *scene.nodes.get_cold(handle2_1).unwrap());
        assert_eq!(NodeIndex::default(), scene.parents[0]);
        assert_eq!(NodeIndex::default(), scene.parents[1]);
        assert_eq!(NodeIndex::new(0), scene.parents[2]);
        assert_eq!(NodeIndex::new(1), scene.parents[3]);
        assert_eq!(
            glam::Vec3A::default(),
            scene.local_transforms[0].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[1].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[2].translation
        );
        assert_eq!(
            glam::Vec3A::new(2.0, 2.0, 2.0),
            scene.local_transforms[3].translation
        );
        assert_eq!(
            glam::Vec3A::default(),
            scene.world_transforms[0].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[1].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[2].translation
        );
        assert_eq!(
            glam::Vec3A::new(3.0, 3.0, 3.0),
            scene.world_transforms[3].translation
        );
    }

    #[test]
    fn remove_node() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeData::Empty, Affine3A::default());
        let handle1_1 = scene.add_node(
            handle1,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2_1 = scene.add_node(
            handle2,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(2.0, 2.0, 2.0)),
        );
        scene.update(&DummyResolver::default());
        scene.remove_node(handle1_1);
        scene.update(&DummyResolver::default());
        assert_eq!(NodeIndex::new(0), *scene.nodes.get_cold(handle1).unwrap());
        assert_eq!(NodeIndex::new(1), *scene.nodes.get_cold(handle2).unwrap());
        assert_eq!(NodeIndex::new(2), *scene.nodes.get_cold(handle2_1).unwrap());
        assert_eq!(NodeIndex::default(), scene.parents[0]);
        assert_eq!(NodeIndex::default(), scene.parents[1]);
        assert_eq!(NodeIndex::new(1), scene.parents[2]);
        assert_eq!(
            glam::Vec3A::default(),
            scene.local_transforms[0].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[1].translation
        );
        assert_eq!(
            glam::Vec3A::new(2.0, 2.0, 2.0),
            scene.local_transforms[2].translation
        );
        assert_eq!(
            glam::Vec3A::default(),
            scene.world_transforms[0].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[1].translation
        );
        assert_eq!(
            glam::Vec3A::new(3.0, 3.0, 3.0),
            scene.world_transforms[2].translation
        );
    }

    #[test]
    fn move_node() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeData::Empty, Affine3A::default());
        let handle1_1 = scene.add_node(
            handle1,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2_1 = scene.add_node(
            handle2,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(2.0, 2.0, 2.0)),
        );
        scene.update(&DummyResolver::default());
        scene.update_node_transform(
            handle1,
            glam::Affine3A::from_translation(glam::Vec3::new(10.0, 10.0, 10.0)),
        );
        scene.update(&DummyResolver::default());
        assert_eq!(NodeIndex::new(0), *scene.nodes.get_cold(handle1).unwrap());
        assert_eq!(NodeIndex::new(1), *scene.nodes.get_cold(handle2).unwrap());
        assert_eq!(NodeIndex::new(2), *scene.nodes.get_cold(handle1_1).unwrap());
        assert_eq!(NodeIndex::new(3), *scene.nodes.get_cold(handle2_1).unwrap());
        assert_eq!(NodeIndex::default(), scene.parents[0]);
        assert_eq!(NodeIndex::default(), scene.parents[1]);
        assert_eq!(NodeIndex::new(0), scene.parents[2]);
        assert_eq!(NodeIndex::new(1), scene.parents[3]);
        assert_eq!(
            glam::Vec3A::new(10.0, 10.0, 10.0),
            scene.local_transforms[0].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[1].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[2].translation
        );
        assert_eq!(
            glam::Vec3A::new(2.0, 2.0, 2.0),
            scene.local_transforms[3].translation
        );
        assert_eq!(
            glam::Vec3A::new(10.0, 10.0, 10.0),
            scene.world_transforms[0].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[1].translation
        );
        assert_eq!(
            glam::Vec3A::new(11.0, 11.0, 11.0),
            scene.world_transforms[2].translation
        );
        assert_eq!(
            glam::Vec3A::new(3.0, 3.0, 3.0),
            scene.world_transforms[3].translation
        );
    }

    #[test]
    fn remove_parent_node_removes_children() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeData::Empty, Affine3A::default());
        let _handle1_1 = scene.add_node(
            handle1,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(1.0, 1.0, 1.0)),
        );
        let _handle1_2 = scene.add_node(
            handle1,
            NodeData::Empty,
            Affine3A::from_translation(glam::Vec3::new(2.0, 2.0, 2.0)),
        );
        scene.update(&DummyResolver::default());
        scene.remove_node(handle1);
        scene.update(&DummyResolver::default());
        assert_eq!(1, scene.data.len());
        assert_eq!(NodeIndex::new(0), *scene.nodes.get_cold(handle2).unwrap());
        assert_eq!(NodeIndex::default(), scene.parents[0]);
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[0].translation
        );
        assert_eq!(
            glam::Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[0].translation
        );
    }
}
