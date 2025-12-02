#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;
layout (location = 3) in vec3 f_color;

layout(binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
    vec3 view_position;
	vec3 ambient_light_intensity;
	vec3 sun_light_direction;
	vec3 sun_light_color;
};


layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	vec3 specular_color;
    float shininess;
};

void main() {
	vec3 normal = normalize(f_normal);

    vec3 view_dir = normalize(view_position - f_position);
    vec3 half_vector = normalize(view_dir - sun_light_direction);

    float sun_shade = max(0.0f, -dot(sun_light_direction, normal));
    vec3 sun_diffuse =  albedo_color;
    vec3 sun_specular = specular_color *
                        pow(max(0.0f, dot(normal, half_vector)),
                            shininess);
    vec3 sun_light_intensity = sun_shade * sun_light_color *
                               (sun_diffuse + sun_specular) ;

    vec3 color = ambient_light_intensity + sun_light_intensity;
    final_color = vec4(f_color, 1.0f);

}
