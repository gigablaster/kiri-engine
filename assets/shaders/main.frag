#version 450
#extension GL_GOOGLE_include_directive : enable

#include "include/pbr_material.glsl"
#include "include/shared.glsl"
#include "include/bdrf.glsl"

layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 world_pos;
layout(location = 2) in vec3 normal;
layout(location = 3) in mat3 tbn;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 base_color = texture(base_color, uv).rgb;
    vec3 normal = unpack_normal(texture(normals, uv));
    vec3 mr = texture(metallic_roughness, uv).rgb;
    float ao = texture(occlusion, uv).r;
    vec3 emissive = texture(emissive, uv).rgb;
    float metallic = mr.b;
    float roughness = mr.g;

    roughness = max(roughness, 0.0001);

    vec3 N = normalize(tbn * normal);
    vec3 V = normalize(per_pass.eye - world_pos);

    vec3 f0 = vec3(0.04, 0.04, 0.04);
    f0 = mix(f0, base_color, metallic);

    vec3 Lo = vec3(0, 0, 0);
    for (int i = 0; i < 3; i++) {
        Lo += bdrf(N, V, base_color, f0, metallic, roughness, ao, vec3(1.0, -1.0, 1.0) * per_pass.lights[i].direction, per_pass.lights[i].color);
    }
    vec3 diffuse_ambient = ambient_light(N, per_pass.ambient);
    vec3 specular_ambient = ambient_light(-reflect(V, N), per_pass.ambient);
    vec3 ambient = mix(diffuse_ambient * base_color * ao, mix(specular_ambient, diffuse_ambient, roughness * roughness) * f0, metallic);

    out_color = vec4(Lo + ambient, 1.0);
}
