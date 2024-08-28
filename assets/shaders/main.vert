#version 450
#extension GL_GOOGLE_include_directive : enable

#include "include/shared.glsl"
#include "include/layout_static.glsl"

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_world_pos;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out mat3 out_tbn;

void main() {
    mat4 model = dynamic.instances[gl_InstanceIndex].model;

    vec3 world_pos = (model * position).xyz;
    vec3 bitangent = normalize(cross(normal, tangent));
    vec3 T = normalize(vec3(model * vec4(tangent, 0.0)));
    vec3 B = normalize(vec3(model * vec4(bitangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(normal, 0.0)));

    out_uv = uv;
    out_world_pos = world_pos;
    out_normal = normal;
    out_tbn = mat3(T, B, N);

    gl_Position = per_pass.view_projection * vec4(world_pos, 1.0);
}
