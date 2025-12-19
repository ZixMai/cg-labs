#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	veekay::vec3 color;
};

veekay::vec4 white = {1.0f, 1.0f, 1.0f, 1.0f};
veekay::vec4 black = {0.0f, 0.0f, 0.0f, 0.0f};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec3 view_position; float _pad0;

	veekay::vec3 ambient_light_intensity; float _pad1;

	veekay::vec3 camera_light_direction; float _pad2;
	veekay::vec3 camera_light_color;
	uint32_t spot_lights_count = 1;
	uint32_t enabled_camera_light = false;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
	veekay::vec3 specular_color;
	float shininess;
	uint32_t is_light_source;
};

struct SpotLight {
	veekay::vec3 position;
	float radius;
	veekay::vec3 direction;
	float angle; // Косинус угла
	veekay::vec3 color;
	uint32_t enabled;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Material {
	veekay::vec3 albedo_color;
	veekay::vec3 specular_color;
	float shininess;
};

class MaterialNew {
public:
	std::shared_ptr<veekay::graphics::Texture> specular_texture;
	std::shared_ptr<veekay::graphics::Texture> emissive_texture;
	VkSampler specular_sampler;
	VkSampler emissive_sampler;
	VkDescriptorSet material_descriptor_set;
	float shininess;

	MaterialNew(veekay::graphics::Texture* specular_texture_,
				veekay::graphics::Texture* emissive_texture_,
				float shininess_);
};

std::vector<MaterialNew> materials;

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	veekay::vec3 offset;
	bool is_light_source;
	Material material;
	uint32_t material_id;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;
	constexpr static int default_fps = 120;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;
	float speed = 1.;
	bool is_animation_frozen = false;
	bool reverse_animation = false;
	bool enable_camera_light = true;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
inline namespace {
	// Index of the cube that visualises (and follows) the spotlight
	size_t spotlight_model_index = SIZE_MAX;

	Camera camera{
		.position = {0.0f, -0.5f, -3.0f}
	};

	std::vector<Model> models;
	std::array<bool, 3> animated;
	std::array<bool, 3> inverted;
	std::array<float, 3> speed{
		1, 1, 1
	};
	std::vector<SpotLight> spot_lights{
	 {
	 	 .position = {0, -5, 0},
		 .radius = 3,
		 .direction = {0, -1, 0},
		 .angle = std::cosf(M_PI / 6),
		 .color = {1, 1, 1},
		 .enabled = true,
		}
	};
}

// NOTE: Vulkan objects
inline namespace {
	constexpr uint32_t max_point_lights = 16;
	constexpr uint32_t max_spot_lights = 16;

	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkDescriptorPool    material_pool;
	VkDescriptorSetLayout material_set_layout;
	VkDescriptorSet       material_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
	veekay::graphics::Buffer* spot_lights_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * static_cast<float>(M_PI) / 180.0f;
}

MaterialNew::MaterialNew(veekay::graphics::Texture* specular_texture_,
					veekay::graphics::Texture* emissive_texture_,
					float shininess_) :  specular_texture(specular_texture_),
					emissive_texture(emissive_texture_), shininess(shininess_) {

	VkDevice &device = veekay::app.vk_device;
	// NOTE: Allocating descriptor set for material
	{
		VkDescriptorSetAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = material_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &material_set_layout,
		};

		if (vkAllocateDescriptorSets(device, &info, &material_descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan descriptor set\n";
			veekay::app.running = false;
			return;
		}
	}

	// NOTE: Making samplers
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR, // Фильтрация если плотность текселей меньше
			.minFilter = VK_FILTER_LINEAR, // Фильтрация если плотность больше
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, // Фильтрация мип-мапов
			// Что делать, если по какой-то из осей вышли за границы текстурных коорд-т
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.anisotropyEnable = true, // Включить анизотропную фильтрацию?
			.maxAnisotropy = 16.0f,   // Кол-во сэмплов анизотропной фильтрации
			.minLod = 0.0f, // Минимальный уровень мипа
			.maxLod = VK_LOD_CLAMP_NONE, // Максимальный уровень мипа (тут бескоченость)
		};

		if (vkCreateSampler(device, &info, nullptr, &specular_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture specular sampler\n";
			veekay::app.running = false;
			return;
		}

		if (vkCreateSampler(device, &info, nullptr, &emissive_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture emissive sampler\n";
			veekay::app.running = false;
			return;
		}
	}

	// NOTE: Making working it
	{
		VkDescriptorImageInfo image_infos[] = {
			{
				.sampler = specular_sampler,
				.imageView = specular_texture->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			},
			{
				.sampler = emissive_sampler,
				.imageView = emissive_texture->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			}
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_infos[1],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
					   write_infos, 0, nullptr);
	}
}

veekay::mat4 Transform::matrix() const {
	const auto scaling_mtx = veekay::mat4::scaling(scale);
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, rotation.x);
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, rotation.y);
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, rotation.z);
	auto t = veekay::mat4::translation(position);

	return scaling_mtx * rot_mtx_x * rot_mtx_y * rot_mtx_z * t;
}

veekay::mat4 Camera::view() const {
	const auto t = veekay::mat4::translation(-position);
	const auto rot_mtx_x = veekay::mat4::rotation({1., .0, .0}, toRadians(rotation.x));
	const auto rot_mtx_y = veekay::mat4::rotation({.0, -1., .0}, toRadians(rotation.y));
	const auto rot_mtx_z = veekay::mat4::rotation({.0, .0, 1.}, toRadians(rotation.z));

	// return veekay::mat4::transpose(t * rot_mtx_x * rot_mtx_y * rot_mtx_z);
	return t * veekay::mat4::transpose(rot_mtx_x * rot_mtx_y * rot_mtx_z);

}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// NOTE: Loads shader byte code from file
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("../../shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("../../shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
			{
				.location = 3,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			}
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				},
			};

			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Material pool initialisation
		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8
				}
			};

			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 10,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
									   &material_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
									   }
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Material descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				}
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
											&material_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
											}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		//NOTE Material descriptor set allocation
		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = material_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &material_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &material_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		const VkDescriptorSetLayout setLayouts[] = {
			descriptor_set_layout, material_set_layout,
		};

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 2,
			.pSetLayouts = setLayouts,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}

		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	spot_lights_buffer = new veekay::graphics::Buffer(
		max_spot_lights * sizeof(SpotLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	// NOTE: Loading all materials
	{
		{
			materials.emplace_back(
				new veekay::graphics::Texture(
					 cmd, 1, 1,
					 VK_FORMAT_R32G32B32A32_SFLOAT,
					 &white
					 ),
				 new veekay::graphics::Texture(
					 cmd, 1, 1,
					 VK_FORMAT_R32G32B32A32_SFLOAT,
					 &black
					 ),
					 32.0f);
		}

		{
			uint32_t width1, height1;
			uint32_t width2, height2;
			std::vector<uint8_t> pixels1;
			std::vector<uint8_t> pixels2;

			lodepng::decode(pixels1, width1, height1, "../../assets/depresuha.png");
			lodepng::decode(pixels2, width2, height2, "../../assets/skebob.png");

			materials.emplace_back(
			new veekay::graphics::Texture(
				cmd, width1, height1,
				VK_FORMAT_R8G8B8A8_UNORM,
				pixels1.data()),
				new veekay::graphics::Texture(
				cmd, width2, height2,
				VK_FORMAT_R8G8B8A8_UNORM,
				pixels2.data()),
				32.0f
			);
		}
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
			.buffer = spot_lights_buffer->buffer,
			.offset = 0,
			.range = max_spot_lights * sizeof(SpotLight)}
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {1,1,1}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {1,1,1}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {1,1,1}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}, {1,1,1}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{
			.position = {0, 1.0f, 0},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material = Material{
			.albedo_color = {},
			.specular_color = {},
			.shininess = 64.0f
		},
		.material_id = 1
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-0.5f, -1.5f, -0.5f},
		},
		.albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f},
		.offset = {0, -4, -0.5f},
		.material = Material{
			.albedo_color = {},
			.specular_color = {},
			.shininess = 64.0f
		},
		.material_id = 1
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.5f, -0.5f, -0.5f},
		},
		.albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f},
		.offset = {0.5f, -3.5f, -0.5f},
		.material = Material{
			.albedo_color = {},
			.specular_color = {},
			.shininess = 64.0f
		},
		.material_id = 1
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.0f, -0.5f, 0},
		},
		.albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f},
		.offset = {0.0f, -3.5f, 0.0f},
		.material = Material{
			.albedo_color = {},
			.specular_color = {},
			.shininess = 64.0f
		},
		.material_id = 1
	});

	// Visual marker for the moving spotlight
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = spot_lights[0].position,      // start exactly at the light
			.scale    = {0.3f, 0.3f, 0.3f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 0.0f},
		.is_light_source = true
	});
	spotlight_model_index = models.size() - 1;
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	vkDestroySampler(device, texture_sampler, nullptr);
	delete texture;

	for (auto & material : materials) {
		vkDestroySampler(device, material.specular_sampler, nullptr);
		vkDestroySampler(device, material.emissive_sampler, nullptr);
		material.specular_texture.reset();
		material.emissive_texture.reset();
	}

	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	delete spot_lights_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(device, material_set_layout, nullptr);
	vkDestroyDescriptorPool(device, material_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	ImGui::Begin("Controls:");
	ImGui::SliderFloat("Rotation X", &camera.rotation.x, -180.f, 180.f);
	ImGui::SliderFloat("Rotation Y", &camera.rotation.y, -180.f, 180.f);
	ImGui::SliderFloat("Rotation Z", &camera.rotation.z, -180.f, 180.f);
	for (int i = 1; i < models.size() - 1; ++i) {
		ImGui::SliderFloat(std::format("Scale X Model {}", i + 1).c_str(), &models[i].transform.scale.x, .01, 5.f);
		ImGui::SliderFloat(std::format("Scale Y Model {}", i + 1).c_str(), &models[i].transform.scale.y, .01, 5.f);
		ImGui::SliderFloat(std::format("Scale Z Model {}", i + 1).c_str(), &models[i].transform.scale.z, .01, 5.f);
	}
	if (ImGui::Button("Reset camera rotation")) {
		camera.rotation = {0.0f, 0.0f, 0.0f};
	}
	if (ImGui::Button("Reset camera position")) {
		camera.position = {0.0f, 0.0f, 0.0f};
	}
	if (ImGui::Button("Pause animation")) {
		camera.is_animation_frozen ^= 1;
	}
	if (ImGui::Button("Reverse animation")) {
		camera.reverse_animation ^= 1;
	}
	if (ImGui::Button("Reset scaling")) {
		for (auto &model: models) {
			model.transform.scale = {1, 1, 1};
		}
	}
	ImGui::Checkbox("Camera light", &camera.enable_camera_light);
    // Toggle spotlight on/off
    bool spotlight_enabled = static_cast<bool>(spot_lights[0].enabled);
    if (ImGui::Checkbox("Spotlight ON", &spotlight_enabled)) {
        spot_lights[0].enabled = spotlight_enabled ? 1u : 0u;
    }
	ImGui::SliderFloat("Animation speed", &camera.speed, 0.01, 5);
	ImGui::End();

	if (!camera.is_animation_frozen) {
		auto speed = camera.speed * (camera.reverse_animation ? -1.0f : 1.0f);
		for (size_t i = 1; i < models.size() - 1; ++i) {
			auto& model = models[i];
			model.transform.rotation.x = static_cast<int>(time * Camera::default_fps) % 360 * speed * 2 * M_PI / 360 + i;
			model.transform.rotation.y = static_cast<int>(time * Camera::default_fps) % 360 * speed * 2 * M_PI / 360 + i;

			model.transform.position.x = speed * (cosf(time) + model.offset.x) + i;
			// model.transform.position.y = speed * ((float)((int)(time * Camera::default_fps) % 2000) / 1000 + model.offset.y) + i * 2;
			model.transform.position.z = speed * (sinf(time) + model.offset.z) + i;
		}
	}

	// ----- animate the spotlight around the centre -----
	{
		const float radius = 1.0f;   // path radius matches light fall‑off radius
		const float angle  = static_cast<float>(time);   // 1 rad/sec
		veekay::vec3 new_pos{
			radius * std::cosf(angle),
			spot_lights[0].position.y,   // keep original height
			radius * std::sinf(angle)
		};

		spot_lights[0].position = new_pos;

		// Make the spotlight aim at the platform centre
		veekay::vec3 len = veekay::vec3::normalized(-new_pos);
		spot_lights[0].direction = len;

		// Keep the helper cube glued to the light
		if (spotlight_model_index != SIZE_MAX) {
			models[spotlight_model_index].transform.position = new_pos;
		}
	}
	// ----------------------------------------------------

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		if (mouse::isButtonDown(mouse::Button::left)) {
			auto move_delta = mouse::cursorDelta();

			const float rotate_x = -90 * move_delta.y / veekay::app.window_height;
			const float rotate_y = -90 * move_delta.x / veekay::app.window_width;
			camera.rotation.x += rotate_x;
			camera.rotation.y += rotate_y;
		}
		auto view = camera.view();

		veekay::vec3 right = veekay::vec3::normalized({view[0][0], view[1][0], view[2][0]});
		veekay::vec3 up = veekay::vec3::normalized({-view[0][1], -view[1][1], -view[2][1]});
		veekay::vec3 front = veekay::vec3::normalized({view[0][2], view[1][2], view[2][2]});

		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position += front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position -= front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::space))
			camera.position += up * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::z))
			camera.position -= up * 0.1f;
	}

	auto view = camera.view();
	veekay::vec3 front = veekay::vec3::normalized({view[0][2], view[1][2], view[2][2]});
	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.view_position = camera.position,
		.ambient_light_intensity = {0.05f, 0.05f, 0.05f},
		.camera_light_direction = front,
		.camera_light_color = {1.0f, 1.0f, 1.0f},  // белый свет, чтобы освещались любые цвета объектов
		.spot_lights_count = static_cast<uint32_t>(spot_lights.size()),
		.enabled_camera_light = camera.enable_camera_light
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = model.albedo_color;
		uniforms.specular_color = {0.5, 0.5, 0.5};
		auto material = materials[model.material_id];
		uniforms.shininess = material.shininess;
		uniforms.is_light_source = model.is_light_source;
	}

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}

	{
		const size_t alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(SpotLight));

		for (size_t i = 0, n = spot_lights.size(); i < n; ++i) {
			const SpotLight& spot_light = spot_lights[i];

			char* const pointer = static_cast<char*>(spot_lights_buffer->mapped_region) + i * alignment;
			*reinterpret_cast<SpotLight*>(pointer) = spot_light;
		}
	}

}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniforms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniforms_alignment;
		const VkDescriptorSet descriptor_sets[] = {
			descriptor_set,
			materials[model.material_id].material_descriptor_set
		};

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 2, descriptor_sets, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
