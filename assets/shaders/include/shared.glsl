struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

struct HemisphericalAmbient {
    vec3 top;
    vec3 middle;
    vec3 bottom;
};

struct Instance {
    mat4 model;
    float uv_scale;
};

layout(set = 3, binding = 0) readonly buffer Instances {
    Instance instances[];
} dynamic;

vec3 unpack_normal(vec4 tx) {
    vec2 normal_xy = tx.xy * 2.0 - 1.0;
    float normal_z = sqrt(clamp(1.0 - dot(normal_xy, normal_xy), 0.0, 1.0));
    return vec3(normal_xy, normal_z);
}
