#pragma once

#include "particleSys.h"
#include "Shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct ParticleSystemRenderer {
	// psys data
	Shader psys_vsh, psys_fsh;
	ShaderProgram psys_sp;
	ParticleSys *psys;
	GLuint vao_psys, vbo_quad;
	float radius = 10.0;

	// cube data
	GLuint vbo_cube, vao_cube;
	Shader cube_vsh, cube_fsh;
	ShaderProgram cube_sp;
	bool show_cube = true;

	// viewing data
	glm::mat4 view, model, projection;
	glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 200.0f);
	glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

	float fov = 45.0f;
	float zoom = 1.0;
	float xrot = 0.0;
	float yrot = 0.0;
	float zrot = 0.0;

	ParticleSystemRenderer(ParticleSys *psys) : psys(psys), 
		psys_vsh(std::string("./shaders/instanced_points.vs"), GL_VERTEX_SHADER), psys_fsh(std::string("./shaders/instanced_points.fs"), GL_FRAGMENT_SHADER), psys_sp({ psys_vsh,psys_fsh }),
		cube_vsh(std::string("./shaders/basic_point.vs"), GL_VERTEX_SHADER), cube_fsh(std::string("./shaders/basic_point.vs"), GL_FRAGMENT_SHADER), cube_sp({ psys_vsh,psys_fsh }) {
		view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

		glGenVertexArrays(1, &vao_psys);
		glBindVertexArray(vao_psys);

		glBindBuffer(GL_ARRAY_BUFFER, psys->vbo_pos);
		GLCHECKERR();

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glVertexAttribDivisor(0, 1);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		float quadVertices[] = {
			-0.05f,  0.05f,
			 0.05f, -0.05f,
			-0.05f, -0.05f,

			-0.05f,  0.05f,
			 0.05f, -0.05f,
			 0.05f,  0.05f,
		};
		
		glGenBuffers(1, &vbo_quad);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);
		glBufferData(GL_ARRAY_BUFFER, 6 * 2 * sizeof(float), &quadVertices[0], GL_STATIC_DRAW);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
		GLCHECKERR();

		GLfloat cubeLines[] = {
			-1.0,-1.0,-1.0,
			-1.0,-1.0, 1.0,
			-1.0,-1.0, 1.0,
			-1.0, 1.0, 1.0,
			-1.0, 1.0, 1.0,
			-1.0, 1.0,-1.0,
			-1.0, 1.0,-1.0,
			-1.0,-1.0,-1.0,

			-1.0, 1.0,-1.0,
			 1.0, 1.0,-1.0,
			 1.0, 1.0,-1.0,
			 1.0, 1.0, 1.0,
			 1.0, 1.0, 1.0,
			-1.0, 1.0, 1.0,
			-1.0, 1.0, 1.0,
			-1.0, 1.0,-1.0,

			 1.0, 1.0,-1.0,
			 1.0,-1.0,-1.0,
			 1.0,-1.0,-1.0,
			 1.0,-1.0, 1.0,
			 1.0,-1.0, 1.0,
			 1.0, 1.0, 1.0,
			 1.0, 1.0, 1.0,
			 1.0, 1.0,-1.0,

			-1.0,-1.0,-1.0,
			 1.0,-1.0,-1.0,
			 1.0,-1.0,-1.0,
			 1.0, 1.0,-1.0,
			 1.0, 1.0,-1.0,
			-1.0, 1.0,-1.0,
			-1.0, 1.0,-1.0,
			-1.0,-1.0,-1.0,

			-1.0,-1.0,-1.0,
			 1.0,-1.0,-1.0,
			 1.0,-1.0,-1.0,
			 1.0,-1.0, 1.0,
			 1.0,-1.0, 1.0,
			-1.0,-1.0, 1.0,
			-1.0,-1.0, 1.0,
			-1.0,-1.0,-1.0,

			-1.0,-1.0, 1.0,
			 1.0,-1.0, 1.0,
			 1.0,-1.0, 1.0,
			 1.0, 1.0, 1.0,
			 1.0, 1.0, 1.0,
			-1.0, 1.0, 1.0,
			-1.0, 1.0, 1.0,
			-1.0,-1.0, 1.0
		}; // 4 lines per 6 faces = 24 lines
		// 2 vertices per 24 lines = 48 vertices

		glGenVertexArrays(1, &vao_cube);
		glGenBuffers(1, &vbo_cube);
		GLCHECKERR();

		glBindVertexArray(vao_cube);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_cube);
		GLCHECKERR();

		glBufferData(GL_ARRAY_BUFFER, 48 * 3 * sizeof(float), &cubeLines[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);
		GLCHECKERR();

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
		GLCHECKERR();
	}

	~ParticleSystemRenderer() {
		glDeleteVertexArrays(1, &vao_psys);
		glDeleteVertexArrays(1, &vao_cube);
		glDeleteBuffers(1, &vbo_quad);
		glDeleteBuffers(1, &vbo_cube);
	}

	void renderps(int width, int height) {

		projection = glm::perspective(glm::radians(fov), (float)width / (float)height, 0.1f, 300.0f);

		view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		model = glm::scale(glm::mat4(1.0), glm::vec3(zoom, zoom, zoom));
		model = glm::rotate(model, xrot, glm::vec3(1.0, 0.0, 0.0));
		model = glm::rotate(model, yrot, glm::vec3(0.0, 1.0, 0.0));
		model = glm::rotate(model, zrot, glm::vec3(0.0, 0.0, 1.0));

		psys_sp.use();
		psys_sp.setUniform(std::string("u_model"), model);
		psys_sp.setUniform(std::string("u_view"), view);
		psys_sp.setUniform(std::string("u_projection"), projection);
		psys_sp.setUniform(std::string("u_radius"), radius);

		glBindVertexArray(vao_psys);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		//glEnable(GL_BLEND);
		glDrawArraysInstanced(GL_TRIANGLES, 0, 6, psys->numParticles);
		psys_sp.unuse();

		if (show_cube) {
			cube_sp.use();

			model = glm::scale(glm::mat4(1.0), glm::vec3(50.0 * zoom, 50.0 * zoom, 50.0 * zoom));
			model = glm::rotate(model, xrot, glm::vec3(1.0, 0.0, 0.0));
			model = glm::rotate(model, yrot, glm::vec3(0.0, 1.0, 0.0));
			model = glm::rotate(model, zrot, glm::vec3(0.0, 0.0, 1.0));
			cube_sp.setUniform(std::string("u_model"), model);
			cube_sp.setUniform(std::string("u_view"), view);
			cube_sp.setUniform(std::string("u_projection"), projection);

			glBindVertexArray(vao_cube);
			glEnable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawArrays(GL_LINES, 0, 48);

			cube_sp.unuse();

			GLCHECKERR();
		}

	}
};