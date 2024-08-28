#version 450
#extension GL_GOOGLE_include_directive : enable

#include "include/shared.glsl"
#include "include/layout_static.glsl"

layout (location = 0) out vec2 out_uv;

void main() {
    mat4 model = dynamic.instances[gl_InstanceIndex].model;

    // vec3 world_pos = vec4(position, 1.0) * model).xyz;
    out_uv = uv;
    gl_Position = per_pass.projection * per_pass.view * model * vec4(position, 1.0);
}