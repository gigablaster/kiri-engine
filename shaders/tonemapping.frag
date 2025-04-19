#version 450

layout(set = 0, binding = 0) uniform sampler2D screen;
layout(set = 0, binding = 1) uniform Params {
    float expouse;
} params;

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 in_uv;

float luminance(vec3 v) {
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 reinhard_jodie(vec3 v)
{
    float l = luminance(v);
    vec3 tv = v / (1.0f + v);
    return mix(v / (1.0f + l), tv, tv);
}

void main() {
    vec3 hdr_color = texture(screen, in_uv).rgb;
    vec3 expoused = 1.0 - exp(-hdr_color * params.expouse);
    vec3 ldr_color = reinhard_jodie(expoused);
    out_color = vec4(ldr_color, 1.0);
}