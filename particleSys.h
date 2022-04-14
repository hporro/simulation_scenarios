#pragma once

#include "cuda_gl_interop.h"
#include "glm/vec3.hpp"
#include "glm/glm.hpp"
#include <random>
#include "errorCheck.h"

struct ParticleSys {
	int numParticles;
	GLuint vbo_pos, vbo_vel; // Buffers with the position and velocity of the particles, for rendering porpouses
	ParticleSys(int numParticles):numParticles(numParticles){}
	virtual void update(float dt)=0;
};

__global__ void move_shaker_w_walls(int numParticles, glm::vec3* pos, glm::vec3* vel, glm::vec3* min, glm::vec3* max, float dt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		if (pos[i].x + vel[i].x > max[0].x)vel[i].x *= -1.0;
		if (pos[i].y + vel[i].y > max[0].y)vel[i].y *= -1.0;
		if (pos[i].z + vel[i].z > max[0].z)vel[i].z *= -1.0;
		if (pos[i].x + vel[i].x < min[0].x)vel[i].x *= -1.0;
		if (pos[i].y + vel[i].y < min[0].y)vel[i].y *= -1.0;
		if (pos[i].z + vel[i].z < min[0].z)vel[i].z *= -1.0;
		pos[i] += dt * vel[i];
	}
}

// Example of a ParticleSys
struct ShakerParticleSys : public ParticleSys {
	glm::vec3 *d_vel, *h_vel;
	glm::vec3 *d_pos, *h_pos;
	glm::vec3 h_min, h_max;
	glm::vec3 *d_min, *d_max;

	struct cudaGraphicsResource *vbo_pos_cuda, *vbo_vel_cuda;

	ShakerParticleSys(int numParticles, glm::vec3 min, glm::vec3 max) : ParticleSys(numParticles), h_min(min), h_max(max) {
		h_pos = new glm::vec3[numParticles];
		h_vel = new glm::vec3[numParticles];

		std::random_device dev;
		std::mt19937 rng(dev());
		std::uniform_real_distribution<> distx(h_min.x, h_max.x);
		std::uniform_real_distribution<> disty(h_min.x, h_max.x);
		std::uniform_real_distribution<> distz(h_min.x, h_max.x);
		std::uniform_real_distribution<> dist01(-3.0, 3.0); // 3 m/s

		for (int i = 0; i < numParticles; i++) {
			h_pos[i] = glm::vec3(distx(rng), disty(rng), distz(rng));
			h_vel[i] = glm::vec3(dist01(rng), dist01(rng), dist01(rng));
		}

		cudaMalloc((void**)&d_min, sizeof(glm::vec3));
		cudaMalloc((void**)&d_max, sizeof(glm::vec3));
		cudaMemcpy(d_min, &h_min, sizeof(glm::vec3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &h_max, sizeof(glm::vec3), cudaMemcpyHostToDevice);

		glGenBuffers(1, &vbo_pos);
		glGenBuffers(1, &vbo_vel);
		GLCHECKERR();

		glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
		glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::vec3), &h_pos[0], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_vel);
		glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::vec3), &h_vel[0], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		GLCHECKERR();

		gpuErrchk(cudaGraphicsGLRegisterBuffer(&vbo_pos_cuda, vbo_pos, cudaGraphicsMapFlagsNone));
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&vbo_vel_cuda, vbo_vel, cudaGraphicsMapFlagsNone));
		GLCHECKERR();

		LOG_EVENT("Particle system initialized with {} particles", numParticles);
	}

	~ShakerParticleSys() {
		delete[] h_pos;
		delete[] h_vel;
		cudaGraphicsUnregisterResource(vbo_pos_cuda);
		glDeleteBuffers(1, &vbo_pos);
		cudaGraphicsUnregisterResource(vbo_vel_cuda);
		glDeleteBuffers(1, &vbo_vel);
		LOG_EVENT("Particle system deleted");
	}

	void update(float dt) override {
		size_t bytes=0;
		gpuErrchk(cudaGraphicsMapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &bytes, vbo_pos_cuda));
		gpuErrchk(cudaGraphicsMapResources(1, &vbo_vel_cuda, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_vel, &bytes, vbo_vel_cuda));

		int blocksize = 128;
		int numBlocks = numParticles / blocksize;
		move_shaker_w_walls<<<numBlocks + 1, blocksize>>>(numParticles, d_pos, d_vel, d_min, d_max, dt);

		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_vel_cuda, 0));

		LOG_EVENT("Particle system updated");
	}

};