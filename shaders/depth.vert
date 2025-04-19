#version 450
#extension GL_GOOGLE_include_directive : enable

#include "include/shared.glsl"
#include "include/pass_main.glsl"
#include "include/layout_static.glsl"

void main() {
    mat4 model = dynamic.instances[gl_InstanceIndex].model;
    vec4 position = vec4(in_position, 1.0);
    vec3 world_pos = (model * position).xyz;

    gl_Position = per_pass.view_projection * vec4(world_pos, 1.0);
}
