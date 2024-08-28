const float PI = 3.1415926;

float distribution_ggx(float NdotH, float roughness) {
    float a = roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float geometry_shlick_ggx(float cos_theta, float k) {
    float num = cos_theta;
    float denom = cos_theta * (1.0 - k) + k;

    return num / denom;
}

vec3 frensel_shlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(2, (-5.55473 * cos_theta - 6.98316 * cos_theta));
}

float geometry_smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggx1 = geometry_shlick_ggx(NdotV, k);
    float ggx2 = geometry_shlick_ggx(NdotL, k);

    return ggx1 * ggx2;
}

vec3 ambient_light(vec3 dir, HemisphericalAmbient ambient) {
    vec3 target = dir.y < 0.0 ? ambient.bottom : ambient.top;
    return mix(ambient.middle, target, abs(dir.y));
}

vec3 bdrf(vec3 N, vec3 V, vec3 base_color, vec3 f0, float metallic, float roughness, float ao, vec3 light_dir, vec3 light_color) {
    vec3 L = normalize(light_dir);
    vec3 H = normalize(L + V);

    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float HdotV = clamp(dot(H, V), 0.0, 1.0);

    vec3 F = frensel_shlick(HdotV, f0);
    float NDF = distribution_ggx(NdotH, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);

    vec3 num = NDF * G * F;
    float denom = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = num / denom;

    vec3 kS = f0;
    vec3 kD = vec3(1.0, 1.0, 1.0) - kS;

    kD *= 1.0 - metallic;

    return (kD * base_color * NdotL * ao + specular) * light_color;
}
