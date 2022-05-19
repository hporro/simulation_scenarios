#pragma once

#include "particleSys.h"
#include "helper.cuh"
#include "grid.h"
#include <algorithm>

#define EPSILON 0.000001

struct boids_sim_settings {
	float RADA = 0.6; // separation
	float RADB = 0.8; // cohesion
	float RADC = 1.5; // alignement

	float A_FORCE = 1.0;
	float B_FORCE = 0.05;
	float C_FORCE = 0.4;

	float MAX_VEL = 0.25;
	float MAX_FORCE = 0.13;
};

struct boids_neighbor_functor {
	void resetForces() {
		cudaMemset(separation, 0.0, numVertices * sizeof(glm::vec3));
		cudaMemset(cohesion, 0.0, numVertices * sizeof(glm::vec3));
		cudaMemset(alignement, 0.0, numVertices * sizeof(glm::vec3));
		cudaMemset(num_A, 0, numVertices * sizeof(int));
		cudaMemset(num_B, 0, numVertices * sizeof(int));
		cudaMemset(num_C, 0, numVertices * sizeof(int));
		//cudaMemset(vel, 0, numVertices * sizeof(glm::vec3));
	}
	inline __device__ void operator()(const int& i, const int& j, const glm::vec3& dist_vec, const float& dist) {
		if ((dist > EPSILON) && (dist < d_bss->RADA)) {
			separation[i] -= dist_vec;
			num_A[i] += 1;
		}
		if ((dist > EPSILON) && (dist < d_bss->RADB)) {
			cohesion[i] += dist_vec;
			num_B[i] += 1;
		}
		if ((dist > EPSILON) && (dist < d_bss->RADC)) {
			alignement[i] += vel[j];
			num_C[i] += 1;
		}
	}
	int numVertices;
	// this class dont own this stuff
	glm::vec3 *separation, *cohesion, *alignement; 
	glm::vec3 *pos, *vel;
	glm::vec3 offset;
	int *num_A, *num_B, *num_C;
	boids_sim_settings *d_bss;
};

__global__ void move_boids_w_walls(int numParticles, glm::vec3* pos, glm::vec3* vel, glm::vec3* min, glm::vec3* max, float dt,
	glm::vec3 *separation, glm::vec3 *cohesion, glm::vec3 *alignement, int *num_A, int *num_B, int *num_C, boids_sim_settings* bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		//for(int j=0;j<numParticles;j++){
		//	if(i==j)continue;
		//	glm::vec3 dist_vec = pos[i] - pos[j];
		//	float dist = glm::length(dist_vec);
		//	if ((dist > 0) && (dist < bss->RADA)) {
		//		separation[i] += glm::normalize(dist_vec)/dist;
		//		num_A[i] += 1;
		//	}
		//	if ((dist > 0) && (dist < bss->RADB)) {
		//		cohesion[i] += pos[j];
		//		num_B[i] += 1;
		//	}
		//	if ((dist > 0) && (dist < bss->RADC)) {
		//		alignement[i] += vel[j];
		//		num_C[i] += 1;
		//	}
		//}

		//printf("i: %d separation[i]: %f %f %f\n", i, separation[i].x, separation[i].y, separation[i].z);
		//printf("i: %d cohesion[i]: %f %f %f\n", i, cohesion[i].x, cohesion[i].y, cohesion[i].z);
		//printf("i: %d alignement[i]: %f %f %f\n", i, alignement[i].x, alignement[i].y, alignement[i].z);


		// separation
		if (num_A[i] > 0)separation[i] /= (float)num_A[i];
		if (glm::length(separation[i]) > 0) {
			separation[i] = glm::normalize(glm::normalize(separation[i]) * bss->MAX_VEL - vel[i]) * bss->MAX_FORCE;
		}
		
		// cohesion
		if (num_B[i] > 0) if (glm::length(cohesion[i]) > 0.0) {
			cohesion[i] /= (float)num_B[i];
			cohesion[i] = glm::normalize(glm::normalize(cohesion[i]) * bss->MAX_VEL - vel[i]) * bss->MAX_FORCE;
		}

		// alignement
		if (num_C[i] > 0) if(glm::length(alignement[i])>0.0) {
			alignement[i] /= (float)num_C[i];
			alignement[i] = glm::normalize(glm::normalize(alignement[i]) * bss->MAX_VEL - vel[i]) * bss->MAX_FORCE;
		}

		glm::vec3 acel = separation[i] + cohesion[i] + alignement[i];

		vel[i] += dt * acel;
		//printf("i: %d vel[i]: %f %f %f\n", i, vel[i].x, vel[i].y, vel[i].z);

		if(glm::length(vel[i])>0)
		vel[i] = glm::normalize(vel[i]) * bss->MAX_VEL;
		//printf("i: %d vel[i]: %f %f %f\n", i, vel[i].x, vel[i].y, vel[i].z);

		if (pos[i].x + dt * vel[i].x > max[0].x)vel[i].x *= -1.0;
		if (pos[i].y + dt * vel[i].y > max[0].y)vel[i].y *= -1.0;
		if (pos[i].z + dt * vel[i].z > max[0].z)vel[i].z *= -1.0;
		if (pos[i].x + dt * vel[i].x < min[0].x)vel[i].x *= -1.0;
		if (pos[i].y + dt * vel[i].y < min[0].y)vel[i].y *= -1.0;
		if (pos[i].z + dt * vel[i].z < min[0].z)vel[i].z *= -1.0;

		pos[i] += dt * vel[i];
	}
}

struct BoidsParticleSys : public ParticleSys {

	boids_sim_settings* d_bss;
	boids_sim_settings* h_bss;
	boids_neighbor_functor cff;

	glm::vec3 *d_vel, *h_vel;
	glm::vec3 *d_pos, *h_pos;
	glm::vec3  h_min,  h_max;
	glm::vec3 *d_min, *d_max;
	Grid* m_grid;

	glm::vec3 *d_separation, *d_cohesion, *d_alignement;
	int *d_num_A, *d_num_B, *d_num_C;
	
	struct cudaGraphicsResource* vbo_pos_cuda, * vbo_vel_cuda;

	BoidsParticleSys(int numParticles, glm::vec3 min, glm::vec3 max, boids_sim_settings bss) : ParticleSys(numParticles), h_min(min), h_max(max)
	{
		m_grid = new Grid(numParticles, glm::vec3(-50.0), glm::vec3(1.5), glm::ivec3(70));
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

		cudaMalloc((void**)&d_separation, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_cohesion, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_alignement, numParticles * sizeof(glm::vec3));

		cudaMalloc((void**)&d_num_A, numParticles * sizeof(int));
		cudaMalloc((void**)&d_num_B, numParticles * sizeof(int));
		cudaMalloc((void**)&d_num_C, numParticles * sizeof(int));

		cudaMalloc((void**)&d_bss, sizeof(boids_sim_settings));
		h_bss = new boids_sim_settings[1];
		h_bss[0] = bss;
		cudaMemcpy(d_bss, h_bss, sizeof(boids_sim_settings), cudaMemcpyHostToDevice);

		cudaMemset(d_separation, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_cohesion, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_alignement, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_num_A, 0, numParticles * sizeof(int));
		cudaMemset(d_num_B, 0, numParticles * sizeof(int));
		cudaMemset(d_num_C, 0, numParticles * sizeof(int));

		cudaMalloc((void**)&d_min, sizeof(glm::vec3));
		cudaMalloc((void**)&d_max, sizeof(glm::vec3));
		cudaMemcpy(d_min, &h_min, sizeof(glm::vec3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &h_max, sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cff.numVertices = numParticles;
		cff.separation = d_separation; cff.cohesion = d_cohesion; cff.alignement = d_alignement;
		cff.num_A = d_num_A; cff.num_B = d_num_B; cff.num_C = d_num_C;
		cff.d_bss = d_bss;

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

		LOG_EVENT("Boids particle system initialized with {} particles", numParticles);
	}

	~BoidsParticleSys() {
		delete[] h_pos;
		delete[] h_vel;

		cudaFree(d_separation);
		cudaFree(d_cohesion);
		cudaFree(d_alignement);

		cudaFree(d_num_A);
		cudaFree(d_num_B);
		cudaFree(d_num_C);
			
		cudaFree(d_bss);
		delete[] h_bss;

		cudaFree(d_min);
		cudaFree(d_max);
	}

	void update(float dt) override {
		size_t bytes = 0;
		gpuErrchk(cudaGraphicsMapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &bytes, vbo_pos_cuda));
		gpuErrchk(cudaGraphicsMapResources(1, &vbo_vel_cuda, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_vel, &bytes, vbo_vel_cuda));

		cff.resetForces();

		int blocksize = 1024;
		int numBlocks = numParticles / blocksize + 1;

		Timer grid_timer;

		m_grid->update(d_pos,d_vel);
		cudaDeviceSynchronize();
		LOG_TIMING("Grid update: {} ms", grid_timer.swap_time());

		cff.pos = d_pos;
		cff.vel = d_vel;
		m_grid->apply_f_frnn<boids_neighbor_functor>(cff, d_pos, 1.5);
		cudaDeviceSynchronize();
		LOG_TIMING("Grid query: {} ms", grid_timer.swap_time());

		move_boids_w_walls <<<numBlocks, blocksize >>> (numParticles, d_pos, d_vel, d_min, d_max, 0.05, d_separation, d_cohesion, d_alignement, d_num_A, d_num_B, d_num_C, d_bss);
		cudaDeviceSynchronize();
		LOG_TIMING("Integration: {} ms", grid_timer.swap_time());

		LOG_EVENT("Mean num of particles: {}", m_grid->mean_num_particle_in_cell());
		LOG_TIMING("Calc mean num of particles: {}", grid_timer.swap_time());

		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_vel_cuda, 0));

		LOG_EVENT("Boids particle system updated");
	}

};