layout(set = 2, binding = 0) uniform MaterialData {
    float emissie_power;
    float alpha_cutoff;
} material;

layout(set = 2, binding = 1) uniform sampler2D base_color;
layout(set = 2, binding = 2) uniform sampler2D normals;
layout(set = 2, binding = 3) uniform sampler2D metallic_roughness;
layout(set = 2, binding = 4) uniform sampler2D occlusion;
layout(set = 2, binding = 5) uniform sampler2D emissive;
