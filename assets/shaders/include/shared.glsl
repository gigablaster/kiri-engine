struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

struct HemisphericalAmbient {
    vec3 top;
    vec3 middle;
    vec3 bottom;
};

layout(set = 0, binding = 0) uniform PerPass {
    mat4 view;
    mat4 projection;
    mat4 view_projection;
    vec3 eye;
    DirectionalLight lights[3];
    HemisphericalAmbient ambient;
} per_pass;

struct Instance {
    mat4 model;
};

layout(set = 3, binding = 0) buffer Instances {
    Instance instances[];
} dynamic;

vec3 unpack_normal(vec4 tx) {
    vec2 normal_xy = tx.xy * 2.0 - 1.0;
    float normal_z = sqrt(clamp(1.0 - dot(normal_xy, normal_xy), 0.0, 1.0));
    return vec3(normal_xy, normal_z);
}
