#version 450
#extension GL_GOOGLE_include_directive : enable

#include "include/pbr_material.glsl"
#include "include/shared.glsl"

layout(location = 0) in vec2 uv;

layout(constant_id = 0) const int USE_DISCARD = 0;

void main() {
    if (USE_DISCARD != 0 && texture(base_color, uv).a <= material.alpha_cutoff) {
        discard;
    }
}
