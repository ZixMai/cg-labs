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


void main() {
	vec3 normal = normalize(f_normal);
    vec3 color = ambient_light_intensity;

    vec3 view_dir = normalize(view_position - f_position);
    vec3 half_vector = normalize(view_dir - camera_light_direction);

    float camera_shade = max(0.0f, -dot(camera_light_direction, normal));
    vec3 camera_diffuse =  albedo_color;
    vec3 camera_specular = specular_color *
                        pow(max(0.0f, dot(normal, half_vector)),
                            shininess);
    vec3 camera_light_intensity = camera_shade * camera_light_color *
                               (camera_diffuse + camera_specular) ;

    if (enable_camera_light == 1) {
        color += camera_light_intensity;
    }

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

        float epsilon = 0.05;
        float intensity = smoothstep(light_angle, light_angle + epsilon, spot_angle);

        if (intensity > 0.0) {
            vec3 light_view = light_position - f_position;
            vec3 h_vector = normalize(view_dir + spot_direction);
            float light_shade = max(0.0f, dot(normal, spot_direction));
            float light_falloff = light_radius * light_radius / dot(light_view, light_view);
            vec3 light_spec = specular_color *
            pow(max(0.0f, dot(normal, h_vector)), shininess);

            color += intensity * light_shade * light_falloff *
            (light.color * albedo_color + light_spec);
        }
    }

    final_color = vec4(color , 1.0f);


    if (is_light_source == 1) {
        final_color = vec4(1.0f);
    }
}
