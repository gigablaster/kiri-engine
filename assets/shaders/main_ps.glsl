#version 450
#extension GL_GOOGLE_include_directive : enable

#include "include/pbr_material.glsl"

layout (location = 0) in vec2 uv;

layout (location = 0) out vec4 out_color;

void main() {
    out_color = texture(base_color, uv);
}