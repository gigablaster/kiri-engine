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

use kiri_common::{Handle, HotColdPool, NodeIndex};
use kiri_math::{Affine3A, BoundingBox, Bounds};
use kiri_resources::{ModelHandle, ResourceResolver, StaticRenderMesh};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeValue {
    Empty,
    StaticMesh(ModelHandle, u32),
    Model(ModelHandle),
}

impl NodeValue {
    pub fn bounds<T: ResourceResolver>(self, resolver: &T) -> Option<BoundingBox> {
        match self {
            Self::StaticMesh(handle, index) => resolver
                .resolve_static_mesh(handle, index)
                .map(|x| x.bounds),
            _ => None,
        }
    }
}

pub type NodeHandle = Handle<SceneNode>;
type NodePool = HotColdPool<SceneNode, NodeIndex>;

/// Клиентские данные о ноде
#[derive(Debug, Clone, Copy)]
pub struct SceneNode {
    data: NodeValue,
    parent: NodeHandle,
    transform: Affine3A,
}

/// Интерфейс для получения данных из сцены
pub trait SceneCuller: Send + Sync {
    fn visible(&self, bounds: BoundingBox) -> bool;
}

const MAX_SCENE_NODES: usize = 0xfffff;

#[derive(Debug)]
pub struct Scene {
    nodes: NodePool,
    data: Vec<NodeValue>,
    parents: Vec<NodeIndex>,
    local_transforms: Vec<Affine3A>,
    world_transforms: Vec<Affine3A>,
    bounds: Vec<BoundingBox>,
    rebuild_scene: bool,
    recalculate_transforms: bool,
    update_bounds: Vec<NodeHandle>,
}

#[derive(Debug)]
pub struct CullResult<'a> {
    pub static_meshes: Vec<(Affine3A, &'a StaticRenderMesh)>,
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

pub struct SceneCullIterator<'a, T: ResourceResolver, C: SceneCuller> {
    resolver: &'a T,
    culler: &'a C,
    nodes: &'a [NodeValue],
    bounds: &'a [BoundingBox],
    transforms: &'a [Affine3A],
    index: usize,
    model_node_to_mesh_index: usize,
}

impl<'a, T: ResourceResolver, C: SceneCuller> SceneCullIterator<'a, T, C> {
    fn new(
        nodes: &'a [NodeValue],
        bounds: &'a [BoundingBox],
        transforms: &'a [Affine3A],
        resolver: &'a T,
        culler: &'a C,
    ) -> Self {
        Self {
            resolver,
            nodes,
            bounds,
            transforms,
            culler,
            index: 0,
            model_node_to_mesh_index: 0,
        }
    }
}

impl<'a, T: ResourceResolver, C: SceneCuller> Iterator for SceneCullIterator<'a, T, C> {
    type Item = (Affine3A, &'a StaticRenderMesh);

    fn next(&mut self) -> Option<Self::Item> {
        // Check current node, skip empty nodes and process nodes with content
        while self.index < self.nodes.len() {
            let current_index = self.index;
            match self.nodes[self.index] {
                NodeValue::Empty => {
                    // Just skip it
                    self.index += 1;
                }
                NodeValue::StaticMesh(model, index) => {
                    self.index += 1;
                    // It's a mesh. Check if it's really exist, cull and return
                    if let Some(mesh) = self.resolver.resolve_static_mesh(model, index) {
                        if self.culler.visible(self.bounds[current_index]) {
                            return Some((self.transforms[current_index], mesh));
                        }
                    }
                }
                NodeValue::Model(model) => {
                    // First, we get model from resources.
                    if let Some(model) = self.resolver.resolve_model(model) {
                        // Check if we stll have unprocessed meshes in model.
                        while self.model_node_to_mesh_index < model.node_to_mesh.len() {
                            // Get current submesh, transform bounds into world space and cull
                            let (node_index, mesh_index) =
                                model.node_to_mesh[self.model_node_to_mesh_index];
                            self.model_node_to_mesh_index += 1;
                            let node_index = node_index as usize;
                            let mesh_index = mesh_index as usize;
                            let transform =
                                self.transforms[current_index] * model.world_transforms[node_index];
                            let bounds = model.bounds_per_mesh[mesh_index].transform(transform);
                            if self.culler.visible(bounds) {
                                let mesh = &model.meshes[mesh_index];
                                return Some((transform, mesh));
                            }
                        }
                    }
                    // Next node
                    self.model_node_to_mesh_index = 0;
                    self.index += 1;
                }
            }
        }
        // List is over
        None
    }
}

impl Scene {
    pub fn add_node(
        &mut self,
        parent: NodeHandle,
        data: NodeValue,
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

    pub fn update_node_data(&mut self, handle: NodeHandle, data: NodeValue) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.data = data;
            if let Some(index) = self.nodes.get_cold(handle).unwrap().index() {
                self.data[index as usize] = data;
                self.update_bounds.push(handle);
            }
        }
    }

    pub fn update<T: ResourceResolver>(&mut self, resolver: &T) {
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

    pub fn cull<'a, T: SceneCuller, U: ResourceResolver>(
        &'a self,
        culler: &'a T,
        resolver: &'a U,
    ) -> impl Iterator<Item = (Affine3A, &'a StaticRenderMesh)> {
        puffin::profile_function!();
        assert!(
            !self.rebuild_scene && self.update_bounds.is_empty() && !self.recalculate_transforms,
            "Scene must be updated before culling"
        );
        SceneCullIterator::new(
            &self.data,
            &self.bounds,
            &self.world_transforms,
            resolver,
            culler,
        )
    }
}

#[cfg(test)]
mod test {
    use kiri_math::{Vec3, Vec3A};
    use kiri_resources::{RenderModel, StaticRenderMesh};

    use super::*;

    #[derive(Default)]
    struct DummyResolver {}

    impl ResourceResolver for DummyResolver {
        fn resolve_static_mesh(
            &self,
            _handle: ModelHandle,
            _index: u32,
        ) -> Option<&StaticRenderMesh> {
            None
        }

        fn resolve_model(&self, _handle: ModelHandle) -> Option<&RenderModel> {
            None
        }
    }

    #[test]
    fn build_scene() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeValue::Empty, Affine3A::default());
        let handle1_1 = scene.add_node(
            handle1,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2_1 = scene.add_node(
            handle2,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(2.0, 2.0, 2.0)),
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
        assert_eq!(Vec3A::default(), scene.local_transforms[0].translation);
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[1].translation
        );
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[2].translation
        );
        assert_eq!(
            Vec3A::new(2.0, 2.0, 2.0),
            scene.local_transforms[3].translation
        );
        assert_eq!(Vec3A::default(), scene.world_transforms[0].translation);
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[1].translation
        );
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[2].translation
        );
        assert_eq!(
            Vec3A::new(3.0, 3.0, 3.0),
            scene.world_transforms[3].translation
        );
    }

    #[test]
    fn remove_node() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeValue::Empty, Affine3A::default());
        let handle1_1 = scene.add_node(
            handle1,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2_1 = scene.add_node(
            handle2,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(2.0, 2.0, 2.0)),
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
        assert_eq!(Vec3A::default(), scene.local_transforms[0].translation);
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[1].translation
        );
        assert_eq!(
            Vec3A::new(2.0, 2.0, 2.0),
            scene.local_transforms[2].translation
        );
        assert_eq!(Vec3A::default(), scene.world_transforms[0].translation);
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[1].translation
        );
        assert_eq!(
            Vec3A::new(3.0, 3.0, 3.0),
            scene.world_transforms[2].translation
        );
    }

    #[test]
    fn move_node() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeValue::Empty, Affine3A::default());
        let handle1_1 = scene.add_node(
            handle1,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2_1 = scene.add_node(
            handle2,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(2.0, 2.0, 2.0)),
        );
        scene.update(&DummyResolver::default());
        scene.update_node_transform(
            handle1,
            Affine3A::from_translation(Vec3::new(10.0, 10.0, 10.0)),
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
            Vec3A::new(10.0, 10.0, 10.0),
            scene.local_transforms[0].translation
        );
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[1].translation
        );
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[2].translation
        );
        assert_eq!(
            Vec3A::new(2.0, 2.0, 2.0),
            scene.local_transforms[3].translation
        );
        assert_eq!(
            Vec3A::new(10.0, 10.0, 10.0),
            scene.world_transforms[0].translation
        );
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[1].translation
        );
        assert_eq!(
            Vec3A::new(11.0, 11.0, 11.0),
            scene.world_transforms[2].translation
        );
        assert_eq!(
            Vec3A::new(3.0, 3.0, 3.0),
            scene.world_transforms[3].translation
        );
    }

    #[test]
    fn remove_parent_node_removes_children() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeValue::Empty, Affine3A::default());
        let _handle1_1 = scene.add_node(
            handle1,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let handle2 = scene.add_node(
            Handle::default(),
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        let _handle1_2 = scene.add_node(
            handle1,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(2.0, 2.0, 2.0)),
        );
        scene.update(&DummyResolver::default());
        scene.remove_node(handle1);
        scene.update(&DummyResolver::default());
        assert_eq!(1, scene.data.len());
        assert_eq!(NodeIndex::new(0), *scene.nodes.get_cold(handle2).unwrap());
        assert_eq!(NodeIndex::default(), scene.parents[0]);
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.local_transforms[0].translation
        );
        assert_eq!(
            Vec3A::new(1.0, 1.0, 1.0),
            scene.world_transforms[0].translation
        );
    }

    #[test]
    fn move_attached_objects() {
        let mut scene = Scene::default();
        let handle1 = scene.add_node(Handle::default(), NodeValue::Empty, Affine3A::default());
        scene.add_node(
            handle1,
            NodeValue::Empty,
            Affine3A::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        scene.update(&DummyResolver::default());
        scene.update_node_transform(
            handle1,
            Affine3A::from_translation(Vec3::new(-1.0, -1.0, -1.0)),
        );
        scene.update(&DummyResolver::default());
        assert_eq!(
            Vec3A::new(-1.0, -1.0, -1.0),
            scene.world_transforms[0].translation
        );
        assert_eq!(
            Vec3A::new(0.0, 0.0, 0.0),
            scene.world_transforms[1].translation
        );
    }
}
