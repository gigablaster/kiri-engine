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

use std::mem;

use crate::{
    MeshAssetMaterial, MeshSurfaceAsset, MeshVertexAttributes, MeshVertexPositions, StaticMeshAsset,
};

#[derive(Debug)]
pub struct MeshSurfaceBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 4]>,
    uv1: Vec<[f32; 2]>,
    uv2: Vec<[f32; 2]>,
    indices: Vec<u32>,
    material: MeshAssetMaterial,
}

#[derive(Debug, Default)]
pub struct MeshAssetBuilder {
    surfaces: Vec<(Vec<FullVertex>, Vec<u16>, MeshAssetMaterial)>,
}

#[derive(Debug, Default, Clone, Copy)]
struct FullVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uvs: [[f32; 2]; 2],
}

impl MeshSurfaceBuilder {
    pub fn new(material: MeshAssetMaterial) -> Self {
        Self {
            material,
            positions: Default::default(),
            normals: Default::default(),
            tangents: Default::default(),
            uv1: Default::default(),
            uv2: Default::default(),
            indices: Default::default(),
        }
    }

    pub fn push_position(&mut self, positions: &[[f32; 3]]) {
        debug_assert!(!positions.is_empty());
        positions.iter().for_each(|x| self.positions.push(*x));
    }

    pub fn push_normals(&mut self, normals: &[[f32; 3]]) {
        debug_assert!(!normals.is_empty());
        normals.iter().for_each(|x| self.normals.push(*x));
    }

    pub fn push_tangents(&mut self, tangents: &[[f32; 4]]) {
        debug_assert!(!tangents.is_empty());
        tangents.iter().for_each(|x| self.tangents.push(*x));
    }

    pub fn push_uv1(&mut self, uvs: &[[f32; 2]]) {
        debug_assert!(!uvs.is_empty());
        uvs.iter().for_each(|x| self.uv1.push(*x));
    }

    pub fn push_uv2(&mut self, uvs: &[[f32; 2]]) {
        debug_assert!(!uvs.is_empty());
        uvs.iter().for_each(|x| self.uv2.push(*x));
    }

    pub fn push_indices(&mut self, indices: &[u32]) {
        debug_assert!(!indices.is_empty());
        indices.iter().for_each(|x| self.indices.push(*x));
    }

    fn build_static(self) -> (Vec<FullVertex>, Vec<u16>, MeshAssetMaterial) {
        let (mut tangents, has_tangents) = if self.positions.len() == self.tangents.len() {
            (self.tangents, true)
        } else {
            (vec![[0.0, 1.0, 0.0, 0.0]; self.positions.len()], false)
        };
        let (normals, has_normals) = if self.positions.len() == self.normals.len() {
            (self.normals, true)
        } else {
            (vec![[1.0, 0.0, 0.0]; self.positions.len()], false)
        };
        let (uv1, has_uvs) = if self.uv1.len() == self.positions.len() {
            (self.uv1, true)
        } else {
            ((vec![[0.0, 0.0]; self.positions.len()]), false)
        };
        let uv2 = if self.uv2.len() == self.positions.len() {
            self.uv2
        } else {
            vec![[0.0, 0.0]; self.positions.len()]
        };
        if has_uvs && has_normals && !has_tangents {
            mikktspace::generate_tangents(&mut TangentCalcContext {
                indices: &self.indices,
                positions: &self.positions,
                normals: &normals,
                uvs: &uv1,
                tangents: &mut tangents,
            });
        }

        let mut indices = self.indices;
        let mut vertices = Vec::with_capacity(self.positions.len());
        for index in 0..self.positions.len() {
            let tangent = tangents[index];
            let vertex = FullVertex {
                position: self.positions[index],
                normal: normals[index],
                tangent,
                uvs: [uv1[index], uv2[index]],
            };
            vertices.push(vertex);
        }
        let (total_vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, Some(&indices));
        vertices = meshopt::remap_vertex_buffer(&vertices, total_vertex_count, &remap);
        indices = meshopt::remap_index_buffer(Some(&indices), total_vertex_count, &remap);
        let remap = meshopt::optimize_vertex_fetch_remap(&indices, total_vertex_count);
        vertices = meshopt::remap_vertex_buffer(&vertices, total_vertex_count, &remap);
        indices = meshopt::remap_index_buffer(Some(&indices), total_vertex_count, &remap);

        let indices = indices.into_iter().map(|x| x as u16).collect::<Vec<_>>();
        (vertices, indices, self.material)
    }
}

struct TangentCalcContext<'a> {
    indices: &'a [u32],
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    uvs: &'a [[f32; 2]],
    tangents: &'a mut [[f32; 4]],
}

impl<'a> mikktspace::Geometry for TangentCalcContext<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[self.indices[face * 3 + vert] as usize] = tangent;
    }
}

fn calculate_bounding_sphere(vertices: &[FullVertex]) -> ([f32; 3], f32) {
    debug_assert!(!vertices.is_empty());
    let mut middle = glam::Vec3::from_array(vertices[0].position);
    for vertex in vertices.iter().skip(1) {
        middle += glam::Vec3::from_array(vertex.position);
    }
    middle /= vertices.len() as f32;
    let mut radius = 0.0;
    for vertex in vertices {
        let distance = middle.distance(glam::Vec3::from_array(vertex.position));
        if distance > radius {
            radius = distance;
        }
    }
    (middle.to_array(), radius)
}

fn find_limit_value<T, const N: usize, F: Fn(&T) -> [f32; N]>(values: &[T], f: F) -> f32 {
    debug_assert!(!values.is_empty());
    let values = values.iter().map(f).collect::<Vec<_>>();
    let mut max = values[0][0];
    for value in values {
        for value in value {
            max = max.max(value);
        }
    }
    max
}

fn quantize_position(value: [f32; 3], max: f32) -> [i16; 3] {
    [
        quantize_float(value[0], max),
        quantize_float(value[1], max),
        quantize_float(value[2], max),
    ]
}

fn quantize_normalized(value: [f32; 3]) -> [i16; 3] {
    [
        quantize_float(value[0], 1.0),
        quantize_float(value[1], 1.0),
        quantize_float(value[2], 1.0),
    ]
}

unsafe fn to_10bits(value: [i16; 3], sign: f32) -> u32 {
    let x = (mem::transmute::<i16, u16>(value[0]) >> 6) as u32; // 16 - 10
    let y = (mem::transmute::<i16, u16>(value[1]) >> 6) as u32; // 16 - 10
    let z = (mem::transmute::<i16, u16>(value[2]) >> 6) as u32; // 16 - 10
    let sign = if sign > 0.0 { 1 } else { 0 };
    // (w & 3) | (x & 0x3ff) << 2 | (y & 0x3ff) << 12 | (z & 0x3ff) << 22
    (z & 0x3ff) | (y & 0x3ff) << 10 | (x & 0x3ff) << 20 | (sign & 3) << 22
}

fn quantize_float(value: f32, max: f32) -> i16 {
    let value = (value / max).clamp(-1.0, 1.0);
    let value = if value > 0.0 {
        value * 32767.0 + 0.5
    } else {
        value * 32767.0 - 0.5
    };
    value.clamp(-32768.0, 32767.0) as i16
}

impl MeshAssetBuilder {
    pub fn push(&mut self, surface: MeshSurfaceBuilder) {
        self.surfaces.push(surface.build_static());
    }

    pub fn build(
        self,
        vertex_positions: &mut Vec<MeshVertexPositions>,
        vertex_attributes: &mut Vec<MeshVertexAttributes>,
        indices: &mut Vec<u16>,
        materials: &mut Vec<MeshAssetMaterial>,
    ) -> StaticMeshAsset {
        let first_vertex = vertex_positions.len() as u64;
        let first_index = indices.len() as u64;
        let mut mesh_vertices = Vec::new();
        let mut mesh_indices = Vec::new();
        let mut mesh_surfaces = Vec::new();
        for (mut vertices, mut indices, material) in self.surfaces {
            let material_index = materials
                .iter()
                .enumerate()
                .find_map(|(index, x)| (*x == material).then_some(index));
            let material_index = if let Some(index) = material_index {
                index
            } else {
                let index = materials.len();
                materials.push(material);
                index
            } as u32;
            mesh_surfaces.push(MeshSurfaceAsset {
                first_index: mesh_indices.len() as u32,
                index_count: indices.len() as u32,
                material: material_index,
            });
            mesh_vertices.append(&mut vertices);
            mesh_indices.append(&mut indices);
        }
        let bounds = calculate_bounding_sphere(&mesh_vertices);
        let position_scale = (find_limit_value(&mesh_vertices, |x| x.position).max(1.0) as u32)
            .next_power_of_two() as f32;
        let mut quantized_vertex_positions = mesh_vertices
            .iter()
            .copied()
            .map(|x| MeshVertexPositions {
                position: quantize_position(x.position, position_scale),
            })
            .collect::<Vec<_>>();
        let mut quantized_vertex_attrubutes = mesh_vertices
            .iter()
            .copied()
            .map(|x| MeshVertexAttributes {
                normal_packed: unsafe { to_10bits(quantize_normalized(x.normal), 1.0) },
                tangent_packed: unsafe {
                    to_10bits(
                        quantize_normalized([x.tangent[0], x.tangent[1], x.tangent[2]]),
                        x.tangent[1],
                    )
                },
                uv: x.uvs[0],
            })
            .collect::<Vec<_>>();
        vertex_positions.append(&mut quantized_vertex_positions);
        vertex_attributes.append(&mut quantized_vertex_attrubutes);
        indices.append(&mut mesh_indices);
        StaticMeshAsset {
            surfaces: mesh_surfaces,
            positon_scale: position_scale,
            bounds,
            first_vertex,
            first_index,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn quantize() {
        assert_eq!(0, quantize_float(0.0, 1.0));
        assert_eq!(-32767, quantize_float(-1.0, 1.0));
        assert_eq!(i16::MAX, quantize_float(1.0, 1.0));
        assert_eq!(0, quantize_float(0.0, 100.0));
        assert_eq!(-32767, quantize_float(-100.0, 100.0));
        assert_eq!(i16::MAX, quantize_float(100.0, 100.0));
        assert_eq!(16384, quantize_float(50.0, 100.0));
        assert_eq!(-16384, quantize_float(-50.0, 100.0));
    }
}
