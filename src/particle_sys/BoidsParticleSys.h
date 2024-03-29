#pragma once

#include "particleSys.h"
#include "../math/batea_math.cuh"
#include "../data_structures/gridCount.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <algorithm>

struct boids_sim_settings {
	float view_distance = 2.0;
	float view_angle = 2.08;

	float A_FORCE = 0.59 * 0.01; //separation
	float B_FORCE = 1.87 * 0.03; //cohesion
	float C_FORCE = 1.34 * 0.07; //alignement

	float MAX_VEL = 0.06;

	float map_size = 100.0;
};

struct boids_neighbor_functor {
	void resetForces() {
		cudaMemset(average_dir, 0.0, numVertices * sizeof(glm::vec3));
		cudaMemset(average_pos, 0.0, numVertices * sizeof(glm::vec3));
		cudaMemset(sep_force, 0.0, numVertices * sizeof(glm::vec3));
		cudaMemset(num_neighbors, 0, numVertices * sizeof(int));
		cudaMemset(d_clock_time, 0, numVertices * sizeof(int));
	}
	inline __device__ void operator()(const int& i, const int& j, const glm::vec3& dist_vec, const float& dist) {
		if (i == j)return;
		clock_t start_time = clock();
		const float r = glm::dot(glm::normalize(dist_vec), glm::normalize(vel[i]));
		if ((dist < d_bss->view_distance) && (acos(r) <= d_bss->view_angle)) {
			//if (dist < d_bss->view_distance) {
			num_neighbors[i] += 1;
			average_dir[i] += vel[j];
			average_pos[i] += pos[j];
			sep_force[i] += -dist_vec / dist;
		}
		clock_t stop_time = clock();
		d_clock_time[i] += (int)(stop_time - start_time);
	}
	int numVertices;
	// this class dont own this stuff
	glm::vec3* average_dir, * average_pos, * sep_force;
	glm::vec3* pos, * vel;
	glm::vec3 offset;
	int* num_neighbors;
	int* d_clock_time;
	boids_sim_settings* d_bss;
};

__global__ void integrate(int numParticles, glm::vec3* pos, glm::vec3* vel, boids_sim_settings* bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		vel[i] = glm::normalize(vel[i]);
		pos[i] += bss->MAX_VEL * vel[i];
	}
}

__global__ void move_boids_w_walls(int numParticles, glm::vec3* pos, glm::vec3* vel, glm::vec3* min, glm::vec3* max, float dt,
	glm::vec3* average_dir, glm::vec3* average_pos, glm::vec3* sep_force, int* num_neighbors, boids_sim_settings* bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {

		vel[i] += sep_force[i] * bss->A_FORCE;
		if (num_neighbors[i] > 0) {
			//alignement
			average_dir[i] /= (float)num_neighbors[i];
			vel[i] += average_dir[i] * bss->C_FORCE;
			//cohesion
			average_pos[i] /= (float)num_neighbors[i];
			vel[i] += -(pos[i] - average_pos[i]) * bss->B_FORCE;
		}

		const glm::vec3 np = pos[i] + (vel[i] * bss->MAX_VEL);
		const float mp2 = bss->map_size * 0.5;
		if (np.x < -mp2 || np.x > mp2 || np.y < -mp2 || np.y > mp2 || np.z < -mp2 || np.z > mp2) {
			const glm::vec3 to_center = glm::normalize(glm::vec3(0.0) - pos[i]);
			vel[i] += to_center * 0.2f;
		}

	}
}

struct BoidsParticleSys : public ParticleSys {

	int numParticles;
	boids_sim_settings* d_bss;
	boids_sim_settings* h_bss;
	boids_neighbor_functor cff;

	glm::vec3* d_vel, * h_vel;
	glm::vec3* d_pos, * h_pos;
	glm::vec3  h_min, h_max;
	glm::vec3* d_min, * d_max;
	GridCount* m_grid;

	glm::vec3* d_average_dir, * d_average_pos, * d_sep_force;
	int* d_num_neighbors, * d_clock_time;

	struct cudaGraphicsResource* vbo_pos_cuda, * vbo_vel_cuda;

	BoidsParticleSys(int numParticles, glm::vec3 min, glm::vec3 max, boids_sim_settings bss) : numParticles(numParticles), ParticleSys(numParticles), h_min(min), h_max(max)
	{
		cudaMalloc((void**)&d_bss, sizeof(boids_sim_settings));
		h_bss = new boids_sim_settings[1];
		h_bss[0] = bss;
		cudaMemcpy(d_bss, h_bss, sizeof(boids_sim_settings), cudaMemcpyHostToDevice);

		m_grid = new GridCount(numParticles, min, glm::vec3(h_bss->view_distance), glm::ivec3(((max.x - min.x) / (int)h_bss->view_distance) + 1));
		h_pos = new glm::vec3[numParticles];
		h_vel = new glm::vec3[numParticles];

		std::random_device dev;
		std::mt19937 rng{ dev() };;
		rng.seed(10);
		std::uniform_real_distribution<> distx(h_min.x, h_max.x);
		std::uniform_real_distribution<> disty(h_min.x, h_max.x);
		std::uniform_real_distribution<> distz(h_min.x, h_max.x);
		std::uniform_real_distribution<> dist01(-3.0, 3.0); // 3 m/s

		for (int i = 0; i < numParticles; i++) {
			h_pos[i] = glm::vec3(distx(rng), disty(rng), distz(rng));
			h_vel[i] = glm::normalize(glm::vec3(dist01(rng), dist01(rng), dist01(rng)));
			//h_vel[i] = glm::vec3(0.0f);
		}

		cudaMalloc((void**)&d_average_dir, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_average_pos, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_sep_force, numParticles * sizeof(glm::vec3));

		cudaMalloc((void**)&d_num_neighbors, numParticles * sizeof(int));
		cudaMalloc((void**)&d_clock_time, numParticles * sizeof(int));

		cudaMemset(d_average_dir, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_average_pos, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_sep_force, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_num_neighbors, 0, numParticles * sizeof(int));
		cudaMemset(d_clock_time, 0, numParticles * sizeof(int));

		cudaMalloc((void**)&d_min, sizeof(glm::vec3));
		cudaMalloc((void**)&d_max, sizeof(glm::vec3));
		cudaMemcpy(d_min, &h_min, sizeof(glm::vec3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &h_max, sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cff.numVertices = numParticles;
		cff.average_dir = d_average_dir; cff.average_pos = d_average_pos; cff.sep_force = d_sep_force;
		cff.num_neighbors = d_num_neighbors;
		cff.d_bss = d_bss;
		cff.d_clock_time = d_clock_time;

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

		cudaFree(d_average_dir);
		cudaFree(d_average_pos);
		cudaFree(d_sep_force);

		cudaFree(d_num_neighbors);

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

		m_grid->update(d_pos, d_vel);
		cudaDeviceSynchronize();
		LOG_TIMING("Grid update: {}", grid_timer.swap_time());

		cff.pos = d_pos;
		cff.vel = d_vel;
		integrate << <numBlocks, blocksize >> > (numParticles, d_pos, d_vel, d_bss);

		//m_grid->culled_f_frnn<boids_neighbor_functor>(cff, d_pos, d_vel, h_bss->view_distance, h_bss->view_angle);
		m_grid->apply_f_frnn<boids_neighbor_functor>(cff, d_pos, h_bss->view_distance);
		cudaDeviceSynchronize();
		LOG_TIMING("Grid query: {}", grid_timer.swap_time());

		// calc time on forces integration
		//thrust::device_ptr<int> clock_timer = thrust::device_pointer_cast(d_clock_time);
		//thrust::inclusive_scan(clock_timer, clock_timer + numParticles, clock_timer);
		//cudaDeviceSynchronize();
		int* res = new int[1];
		cudaMemcpy(res, &d_clock_time[numParticles - 1], sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		// 1395 MHz is the default clock time in the GPU in the lab PC
		LOG_TIMING("Num cycles on force accumulation: {: 0.3f}ms Calc time: {}ms", ((float)res[0]) / 1395.0, grid_timer.swap_time());

		move_boids_w_walls << <numBlocks, blocksize >> > (numParticles, d_pos, d_vel, d_min, d_max, 0.05, d_average_dir, d_average_pos, d_sep_force, d_num_neighbors, d_bss);
		cudaDeviceSynchronize();
		LOG_TIMING("Integration: {}", grid_timer.swap_time());

		LOG_EVENT("Mean num of particles: {}", m_grid->mean_num_particle_in_cell());
		LOG_TIMING("Calc mean num of particles: {}", grid_timer.swap_time());

		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_vel_cuda, 0));

		LOG_EVENT("Boids particle system updated");
	}

};