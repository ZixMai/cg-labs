#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec3 f_color;

layout (location = 0) out vec4 final_color;

layout(binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
    vec3 view_position;
	vec3 ambient_light_intensity;
	vec3 camera_light_direction;
	vec3 camera_light_color;
    uint spot_light_count;
    uint enable_camera_light;
};


layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	vec3 specular_color;
    float shininess;
    uint is_light_source;
};

struct SpotLight {
    vec4 position_radius;
    vec4 direction_angle;
    vec3 color;
    uint enabled;
};

layout(binding = 2, std430) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout (set = 1, binding = 0) uniform sampler2D specular_texture;
layout (set = 1, binding = 1) uniform sampler2D emissive_texture;

void main() {
    vec2 mapped_uv = vec2(f_uv.x + sin(f_uv.y * 3.14 / 2 + 0.5) * 0.2, f_uv.y);
    // ----- wave‑distortion of texture coordinates --------------------------
//    const float wave_freq = 10.0;// number of waves across the quad
//    const float wave_amp  = 0.05;// maximum offset in UV space
//    mapped_uv = f_uv +
//    vec2(sin(f_uv.y * wave_freq) * wave_amp, // horizontal wobble
//    cos(f_uv.x * wave_freq) * wave_amp);// vertical wobble

    vec3 normal = normalize(f_normal);
    vec3 color = ambient_light_intensity;

    vec3 view_dir = normalize(view_position - f_position);

    // ---------- 1. базовый и «световой» цвета ---------------------------------
    vec3 base_col = texture(specular_texture, mapped_uv).rgb * albedo_color;
    vec3 camera_diffuse = texture(emissive_texture,  mapped_uv).rgb * albedo_color;

    // ---------- 2. считаем общее диффузное освещение --------------------------
    float light_factor = 0.0;

    /* --- камера ------------------------------------------------------------ */
    if (enable_camera_light == 1) {
        vec3 half_vector = normalize(view_dir - camera_light_direction);

        float camera_shade = max(0.0f, -dot(camera_light_direction, normal));
//        vec3 camera_diffuse = texture(emissive_texture, mapped_uv).rgb;
        //        f_color * albedo_color
        vec3 camera_specular = base_col * pow(max(0.0f, dot(normal, half_vector)), shininess);
        vec3 camera_light_intensity = camera_shade * camera_light_color *
        (camera_diffuse + camera_specular) ;

        color += camera_light_intensity;

        // NOTE gpt idea
        light_factor += camera_shade;
    }

    /* --- прожекторы -------------------------------------------------------- */
    for (uint i = 0; i < spot_light_count; ++i) {
        SpotLight light = spot_lights[i];

        if (light.enabled != 1) {
            continue;
        }

        vec3 light_position = light.position_radius.xyz;
        float light_radius = light.position_radius.w;

        vec3 light_direction = light.direction_angle.xyz;
        float light_angle = light.direction_angle.w;

        vec3 spot_direction = normalize(light_position - f_position);
        float spot_angle = -dot(light_direction, spot_direction);

        if (spot_angle > light_angle) {
            // Прибавляем код освещения по Блинн-Фонгу
            vec3 h_vector = normalize(view_dir + spot_direction);
            vec3 light_view = light_position - f_position;
            vec3 light_view_normal = normalize(light_view);
            float light_shade = max(0.0f, dot(normal, spot_direction));
            light_factor += light_shade;
            float light_falloff = light_radius * light_radius / dot(light_view, light_view);
            vec3 light_spec = texture(specular_texture, mapped_uv).rgb * pow(max(0.0f, dot(normal, h_vector)), shininess);

            color += light_shade * light_falloff * (light.color * albedo_color + light_spec);
        }
    }

    light_factor = clamp(light_factor, 0.0, 1.0);  // нормируем

    // ---------- 3. смешиваем две текстуры ------------------------------------
    vec3 surface = mix(base_col, camera_diffuse, light_factor);

    // ---------- 4. добавляем спекуляры (по второй текстуре) -------------------
    vec3 spec_col = vec3(0.0);
    if (enable_camera_light == 1) {
        vec3 half_v = normalize(view_dir - camera_light_direction);
        spec_col = texture(emissive_texture, mapped_uv).rgb *
        pow(max(0.0, dot(normal, half_v)), shininess);
    }

    // ---------- 5. финальный цвет --------------------------------------------
    final_color = vec4(surface + spec_col, 1.0);

    // куб-«лампочка» оставляем белым
    if (is_light_source == 1)
    final_color = vec4(1.0);
}