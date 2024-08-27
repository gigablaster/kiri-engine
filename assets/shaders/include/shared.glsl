layout(set = 0, binding = 0) uniform PerPass {
    mat4 view;
    mat4 projection;
    mat4 view_projection;
    vec3 eye;
} per_pass;

struct Instance {
    mat4 model;
};

layout(set = 3, binding = 0) buffer Instances {
    Instance instances[];
} dynamic;