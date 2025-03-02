// Copyright (C) 2025 gigablaster

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

use std::{collections::HashSet, sync::Arc};

use kiri_common::{Handle, Pool};
use kiri_gfx::{RenderMesh, RenderModel};
use kiri_math::{vec3, Affine3A, BoundingBox, Bounds, Vec3};
use nohash_hasher::BuildNoHashHasher;
use thiserror::Error;

const MAX_NODES: usize = 262144;
const MAX_SCENE_DEPTH: usize = 32;

#[derive(Debug, Clone, Copy, Default)]
pub struct DirectionalLight {
    pub color: Vec3,
    pub power: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PointLight {
    pub color: Vec3,
    pub power: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Node {
    parent: NodeHandle,
    first_child: NodeHandle,
    next_sibling: NodeHandle,
    level: u32,
}

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("Node not found: {0}")]
    NodeNotFound(NodeHandle),
    #[error("Exceeded max scene depth")]
    ExceededSceneDepth,
}

#[derive(Debug, Clone, Default)]
pub enum NodeData {
    #[default]
    Empty,
    Mesh(Arc<RenderMesh>),
    Model(Arc<RenderModel>),
    DirectionalLight(DirectionalLight),
    PointLight(PointLight),
}

impl NodeData {
    fn cull<'a, C: Culler>(&'a self, transform: Affine3A, culler: &C, out: &mut CullResult<'a>) {
        match self {
            Self::Mesh(mesh) => {
                let bounds = mesh.bounds.transform(transform);
                if culler.cull(bounds) {
                    out.meshes.push((mesh, transform));
                }
            }
            Self::Model(model) => {
                let bounds = model.bounds.transform(transform);
                if culler.cull(bounds) {
                    for (node_index, mesh_index) in model
                        .node_to_mesh
                        .iter()
                        .map(|(x, y)| (*x as usize, *y as usize))
                    {
                        let transform = model.world_transforms[node_index] * transform;
                        if culler.cull(model.bounds_per_mesh[mesh_index].transform(transform)) {
                            out.meshes.push((&model.meshes[mesh_index], transform));
                        }
                    }
                }
            }
            Self::Empty => {}
            Self::DirectionalLight(directional_light) => {
                out.directional_lights.push((*directional_light, transform))
            }
            Self::PointLight(point_light) => {
                let bounds = BoundingBox::from_extent(
                    transform.translation.into(),
                    vec3(point_light.radius, point_light.radius, point_light.radius),
                );
                if culler.cull(bounds) {
                    out.point_lights.push((*point_light, transform));
                }
            }
        }
    }
}

/// Culler interface.
pub trait Culler {
    /// Returns true if bbox passes the cull test
    fn cull(&self, bounds: BoundingBox) -> bool;
}

pub type NodeHandle = Handle<Node>;

#[derive(Debug)]
pub struct CullResult<'a> {
    pub meshes: Vec<(&'a RenderMesh, Affine3A)>,
    pub directional_lights: Vec<(DirectionalLight, Affine3A)>,
    pub point_lights: Vec<(PointLight, Affine3A)>,
}

#[derive(Debug)]
pub struct Scene {
    local_transforms: Vec<Affine3A>,
    world_transforms: Vec<Affine3A>,
    data: Vec<NodeData>,
    nodes: Pool<Node>,
    to_remove: HashSet<NodeHandle, BuildNoHashHasher<NodeHandle>>,
    changed: [HashSet<NodeHandle, BuildNoHashHasher<NodeHandle>>; MAX_SCENE_DEPTH],
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            local_transforms: vec![Affine3A::default(); MAX_NODES],
            world_transforms: vec![Affine3A::default(); MAX_NODES],
            data: vec![NodeData::Empty; MAX_NODES],
            nodes: Pool::new(MAX_NODES),
            to_remove: Default::default(),
            changed: Default::default(),
        }
    }
}

impl Scene {
    pub fn create(
        &mut self,
        root: NodeHandle,
        local_transform: Affine3A,
        data: NodeData,
    ) -> Result<NodeHandle, SceneError> {
        if root.is_valid() {
            // Add child node
            let mut root_node = self
                .nodes
                .get(root)
                .copied()
                .ok_or(SceneError::NodeNotFound(root))?;
            if root_node.level + 1 >= MAX_SCENE_DEPTH as u32 {
                return Err(SceneError::ExceededSceneDepth);
            }
            let node = self.nodes.push(Node {
                parent: root,
                first_child: NodeHandle::default(),
                next_sibling: NodeHandle::default(),
                level: root_node.level + 1,
            });
            let index = node.index() as usize;
            self.data[index] = data;
            self.local_transforms[index] = local_transform;
            // Root node has child
            if let Some(mut first_child) = self.nodes.get(root_node.first_child).copied() {
                // Find last sibling node
                if let Some(last_sibling) = self.find_last_sibling(first_child.next_sibling) {
                    // Add created node to the chain
                    let last_sibling = self.nodes.get_mut(last_sibling).unwrap();
                    last_sibling.next_sibling = node;
                } else {
                    // Otherwise create said chain
                    first_child.next_sibling = node;
                }
                self.nodes.replace(root_node.first_child, first_child);
            } else {
                // No children - create one
                root_node.first_child = node;
            }
            self.nodes.replace(root, root_node);
            self.changed[(root_node.level + 1) as usize].insert(node);
            Ok(node)
        } else {
            // Add root node
            let handle = self.nodes.push(Node {
                parent: NodeHandle::default(),
                first_child: NodeHandle::default(),
                next_sibling: NodeHandle::default(),
                level: 0,
            });
            self.local_transforms[handle.index() as usize] = local_transform;
            self.changed[0].insert(handle);
            Ok(handle)
        }
    }

    pub fn remove(&mut self, handle: NodeHandle) {
        self.to_remove.insert(handle);
    }

    pub fn update_transform(&mut self, handle: NodeHandle, local_transform: Affine3A) {
        if let Some(node) = self.nodes.get(handle) {
            self.local_transforms[handle.index() as usize] = local_transform;
            self.changed[node.level as usize].insert(handle);
        }
    }

    pub fn update(&mut self) {
        puffin::profile_function!();
        self.delete_nodes();
        self.update_hierarchy();
    }

    pub fn cull<C: Culler>(&self, culler: C) -> CullResult {
        puffin::profile_function!();
        let mut result = CullResult {
            meshes: Vec::with_capacity(64536),
            directional_lights: Vec::with_capacity(16),
            point_lights: Vec::with_capacity(256),
        };
        for (handle, _) in self.nodes.enumerate() {
            let index = handle.index() as usize;
            self.data[index].cull(self.world_transforms[index], &culler, &mut result);
        }
        result
    }

    pub fn get_world_transform(&self, handle: NodeHandle) -> Result<Affine3A, SceneError> {
        debug_assert!(
            self.changed.iter().all(|x| x.is_empty()),
            "All changes must be applied"
        );
        if !self.nodes.is_handle_valid(handle) {
            return Err(SceneError::NodeNotFound(handle));
        }
        Ok(self.world_transforms[handle.index() as usize])
    }

    fn find_last_sibling(&self, handle: NodeHandle) -> Option<NodeHandle> {
        let mut handle = handle;
        while let Some(node) = self.nodes.get(handle) {
            if !node.next_sibling.is_valid() {
                return Some(handle);
            }
            handle = node.next_sibling;
        }
        None
    }

    fn delete_nodes(&mut self) {
        let to_remove = self.to_remove.drain().collect::<Vec<_>>();
        let mut collected = HashSet::with_hasher(BuildNoHashHasher::default());
        // For every marked node we collect all of it's children
        for handle in &to_remove {
            self.collect_all_children(*handle, &mut collected);
        }
        // Then we remove marked node from hierarachy
        for handle in to_remove {
            self.remove_single_node(handle);
        }
        // And then remove all collected children from node list
        for handle in collected {
            self.nodes.remove(handle);
        }
    }

    fn remove_single_node(&mut self, handle: NodeHandle) {
        if let Some(node) = self.nodes.remove(handle) {
            // Go to the root.
            if let Some(mut parent) = self.nodes.get(node.parent).copied() {
                // Get child chain
                // If node is first child then move it to next sibling
                if parent.first_child == handle {
                    parent.first_child = node.next_sibling;
                    self.nodes.replace(node.parent, parent);
                } else {
                    // Otherwise traverse the chain until we find the node
                    let mut current = parent.first_child;
                    while let Some(mut current_node) = self.nodes.get(current).copied() {
                        if current_node.next_sibling == handle {
                            // Replace previous node next sibling to our next sibling
                            current_node.next_sibling = node.next_sibling;
                            self.nodes.replace(current, current_node);
                            return;
                        }
                        current = current_node.next_sibling;
                    }
                }
            }
            // No need to do anything for root nodes
        }
    }

    fn collect_all_children(
        &mut self,
        handle: NodeHandle,
        list: &mut HashSet<NodeHandle, BuildNoHashHasher<NodeHandle>>,
    ) {
        if let Some(node) = self.nodes.get(handle) {
            let mut siblings = HashSet::with_hasher(BuildNoHashHasher::default());
            self.collect_siblings(node.first_child, &mut siblings);
            siblings.iter().for_each(|handle| {
                list.insert(*handle);
            });
            for sibling in siblings {
                self.collect_all_children(sibling, list);
            }
        }
    }

    fn update_hierarchy(&mut self) {
        let mut to_update =
            HashSet::with_capacity_and_hasher(MAX_NODES, BuildNoHashHasher::default());
        for level in 0..MAX_SCENE_DEPTH {
            // Add eveything that changed on this level to list from previous level
            // First, collect all valid changed nodes. Then add them to update list.
            // This list already contains nodes filled from previous level.
            for handle in self.changed[level]
                .drain()
                .filter(|handle| self.nodes.is_handle_valid(*handle))
            {
                to_update.insert(handle);
            }
            // Actually update transformations
            for handle in &to_update {
                let node = self.nodes.get(*handle).unwrap();
                if node.parent.is_valid() {
                    self.world_transforms[handle.index() as usize] = self.local_transforms
                        [handle.index() as usize]
                        * self.world_transforms[node.parent.index() as usize];
                } else {
                    self.world_transforms[handle.index() as usize] =
                        self.local_transforms[handle.index() as usize];
                }
            }
            // Clear update list and add all children for dirty nodes on this level so it will be updated on next level
            let for_next_level = to_update.drain().collect::<Vec<_>>();
            for handle in for_next_level {
                if let Some(node) = self.nodes.get(handle) {
                    self.collect_siblings(node.first_child, &mut to_update);
                }
            }
        }
    }

    fn collect_siblings(
        &self,
        handle: NodeHandle,
        list: &mut HashSet<NodeHandle, BuildNoHashHasher<NodeHandle>>,
    ) {
        let mut handle = handle;
        while let Some(node) = self.nodes.get(handle) {
            list.insert(handle);
            handle = node.next_sibling;
        }
    }
}

#[cfg(test)]
mod test {
    use kiri_common::Handle;
    use kiri_math::{vec3, Affine3A};

    use crate::NodeHandle;

    use super::{NodeData, Scene};

    #[test]
    fn add_remove_root_node() {
        let mut scene = Scene::default();
        let transform1 = Affine3A::from_translation(vec3(10.0, 10.0, 10.0));
        let handle = scene
            .create(Handle::default(), transform1, NodeData::Empty)
            .unwrap();
        scene.update();
        assert_eq!(transform1, scene.get_world_transform(handle).unwrap());
        scene.remove(handle);
        scene.update();
        assert!(scene.get_world_transform(handle).is_err());
    }

    #[test]
    fn add_remove_child_nodes() {
        let mut scene = Scene::default();
        let transform1 = Affine3A::from_translation(vec3(10.0, 0.0, 0.0));
        let transform2 = Affine3A::from_translation(vec3(0.0, 10.0, 0.0));
        let transform3 = Affine3A::from_translation(vec3(-10.0, -10.0, 0.0));
        let handle1 = scene
            .create(Handle::default(), transform1, NodeData::Empty)
            .unwrap();
        let handle2 = scene.create(handle1, transform2, NodeData::Empty).unwrap();
        let handle3 = scene.create(handle2, transform3, NodeData::Empty).unwrap();
        let handle4 = scene.create(handle2, transform1, NodeData::Empty).unwrap();
        let handle5 = scene
            .create(NodeHandle::default(), transform2, NodeData::Empty)
            .unwrap();
        let handle6 = scene.create(handle1, transform1, NodeData::Empty).unwrap();
        scene.update();
        assert_eq!(transform1, scene.get_world_transform(handle1).unwrap());
        assert_eq!(
            Affine3A::from_translation(vec3(10.0, 10.0, 0.0)),
            scene.get_world_transform(handle2).unwrap()
        );
        assert_eq!(
            Affine3A::default(),
            scene.get_world_transform(handle3).unwrap()
        );
        assert_eq!(
            Affine3A::from_translation(vec3(20.0, 10.0, 0.0)),
            scene.get_world_transform(handle4).unwrap()
        );
        assert_eq!(transform2, scene.get_world_transform(handle5).unwrap());
        assert_eq!(
            Affine3A::from_translation(vec3(20.0, 0.0, 0.0)),
            scene.get_world_transform(handle6).unwrap()
        );
        scene.remove(handle2);
        scene.update();
        assert!(scene.get_world_transform(handle1).is_ok());
        assert!(scene.get_world_transform(handle5).is_ok());
        assert!(scene.get_world_transform(handle6).is_ok());
        assert!(scene.get_world_transform(handle2).is_err());
        assert!(scene.get_world_transform(handle3).is_err());
        assert!(scene.get_world_transform(handle4).is_err());
    }

    #[test]
    fn move_root_node() {
        let mut scene = Scene::default();
        let transform1 = Affine3A::from_translation(vec3(10.0, 10.0, 10.0));
        let transform2 = Affine3A::from_translation(vec3(0.0, 10.0, 0.0));
        let handle = scene
            .create(Handle::default(), transform1, NodeData::Empty)
            .unwrap();
        scene.update();
        assert_eq!(transform1, scene.get_world_transform(handle).unwrap());
        scene.update_transform(handle, transform2);
        scene.update();
        assert_eq!(transform2, scene.get_world_transform(handle).unwrap());
    }

    #[test]
    fn move_child_node() {
        let mut scene = Scene::default();
        let transform1 = Affine3A::from_translation(vec3(10.0, 0.0, 0.0));
        let transform2 = Affine3A::from_translation(vec3(0.0, 10.0, 0.0));
        let transform3 = Affine3A::from_translation(vec3(-10.0, -10.0, 0.0));
        let handle1 = scene
            .create(Handle::default(), transform1, NodeData::Empty)
            .unwrap();
        let handle2 = scene.create(handle1, transform2, NodeData::Empty).unwrap();
        let handle3 = scene.create(handle2, transform3, NodeData::Empty).unwrap();
        let handle4 = scene.create(handle2, transform1, NodeData::Empty).unwrap();
        let handle5 = scene
            .create(NodeHandle::default(), transform2, NodeData::Empty)
            .unwrap();
        let handle6 = scene.create(handle1, transform1, NodeData::Empty).unwrap();
        scene.update();
        scene.update_transform(handle2, transform1);
        scene.update();
        assert_eq!(transform1, scene.get_world_transform(handle1).unwrap());
        assert_eq!(transform2, scene.get_world_transform(handle5).unwrap());
        assert_eq!(
            Affine3A::from_translation(vec3(20.0, 0.0, 0.0)),
            scene.get_world_transform(handle6).unwrap()
        );
        assert_eq!(
            Affine3A::from_translation(vec3(20.0, 0.0, 0.0)),
            scene.get_world_transform(handle2).unwrap()
        );
        assert_eq!(
            Affine3A::from_translation(vec3(10.0, -10.0, 0.0)),
            scene.get_world_transform(handle3).unwrap()
        );
        assert_eq!(
            Affine3A::from_translation(vec3(30.0, 0.0, 0.0)),
            scene.get_world_transform(handle4).unwrap()
        );
    }
}
