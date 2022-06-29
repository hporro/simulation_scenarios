#pragma once

#include "particleSys.h"
#include "../math/batea_math.cuh"
#include "../data_structures/mcleap_wrapper.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <algorithm>

struct boids_sim_settings {
	double view_distance = 2.0;
	double view_angle = 2.08;

	double A_FORCE = 0.59 * 0.01; //separation
	double B_FORCE = 1.87 * 0.03; //cohesion
	double C_FORCE = 1.34 * 0.07; //alignement

	double MAX_VEL = 0.000006;

	double map_size = 100.0;
};

struct boids_neighbor_functor {
	void resetForces() {
		cudaMemset(average_dir, 0.0, numVertices * sizeof(glm::dvec2));
		cudaMemset(average_pos, 0.0, numVertices * sizeof(glm::dvec2));
		cudaMemset(sep_force, 0.0, numVertices * sizeof(glm::dvec2));
		cudaMemset(num_neighbors, 0, numVertices * sizeof(int));
		cudaMemset(d_clock_time, 0, numVertices * sizeof(int));
	}
	inline __device__ void operator()(const int& i, const int& j, const glm::dvec2& dist_vec, const double& dist) {
		if (i == j)return;
		//clock_t start_time = clock();
		const double r = glm::dot(glm::normalize(dist_vec), glm::normalize(vel[i]));
		if ((dist < d_bss->view_distance) && (acos(r) <= d_bss->view_angle) ) {
			//if (dist < d_bss->view_distance) {
			num_neighbors[i] += 1;
			average_dir[i] += vel[j];
			average_pos[i] += pos[j];
			sep_force[i] += -dist_vec / dist;
		}
		//clock_t stop_time = clock();
		//d_clock_time[i] += (int)(stop_time - start_time);
	}
	int numVertices;
	// this class dont own this stuff
	glm::dvec2* average_dir, * average_pos, * sep_force;
	glm::dvec2* pos, * vel;
	glm::dvec2 offset;
	int* num_neighbors;
	int* d_clock_time;
	boids_sim_settings* d_bss;
};

__global__ void integrate(int numParticles, glm::dvec2* pos, glm::dvec2* vel, boids_sim_settings* bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		vel[i] = glm::normalize(vel[i]);
		pos[i] += bss->MAX_VEL * vel[i];
	}
}

__global__ void move_boids_w_walls(int numParticles, glm::dvec2* pos, glm::dvec2* vel, glm::dvec2* min, glm::dvec2* max, double dt,
	glm::dvec2* average_dir, glm::dvec2* average_pos, glm::dvec2* sep_force, int* num_neighbors, boids_sim_settings* bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {

		vel[i] += sep_force[i] * bss->A_FORCE;
		if (num_neighbors[i] > 0) {
			//alignement
			average_dir[i] /= (double)num_neighbors[i];
			vel[i] += average_dir[i] * bss->C_FORCE;
			//cohesion
			average_pos[i] /= (double)num_neighbors[i];
			vel[i] += -(pos[i] - average_pos[i]) * bss->B_FORCE;
		}

		const glm::dvec2 np = pos[i] + (vel[i] * bss->MAX_VEL);
		const double mp2 = bss->map_size * 0.5;
		if (np.x < -mp2 || np.x > mp2 || np.y < -mp2 || np.y > mp2) {
			const glm::dvec2 to_center = glm::normalize(glm::dvec2(0.0) - pos[i]);
			vel[i] += to_center * 0.2;
		}

	}
}

struct Boids2dParticleSys : public ParticleSys {

	int numParticles;
	boids_sim_settings* d_bss;
	boids_sim_settings* h_bss;
	boids_neighbor_functor cff;

	glm::dvec2* d_vel, * h_vel;
	glm::dvec2* d_pos, * h_pos;
	glm::dvec2  h_min, h_max;
	glm::dvec2* d_min, * d_max;
	triangulation2d<100,100>* m_grid;

	glm::dvec2* d_average_dir, * d_average_pos, * d_sep_force;
	int* d_num_neighbors, * d_clock_time;

	struct cudaGraphicsResource* vbo_pos_cuda, * vbo_vel_cuda;

	Boids2dParticleSys(int numParticles, glm::dvec2 min, glm::dvec2 max, boids_sim_settings bss) : numParticles(numParticles), ParticleSys(numParticles), h_min(min), h_max(max)
	{
		cudaMalloc((void**)&d_bss, sizeof(boids_sim_settings));
		h_bss = new boids_sim_settings[1];
		h_bss[0] = bss;
		cudaMemcpy(d_bss, h_bss, sizeof(boids_sim_settings), cudaMemcpyHostToDevice);

		m_grid = new triangulation2d<100,100>(numParticles);
		h_pos = new glm::dvec2[numParticles];
		h_vel = new glm::dvec2[numParticles];

		std::random_device dev;
		std::mt19937 rng{ dev() };;
		rng.seed(10);
		std::uniform_real_distribution<> distx(h_min.x, h_max.x);
		std::uniform_real_distribution<> disty(h_min.x, h_max.x);
		std::uniform_real_distribution<> distz(h_min.x, h_max.x);
		std::uniform_real_distribution<> dist01(-3.0, 3.0); // 3 m/s

		for (int i = 0; i < numParticles; i++) {
			h_pos[i] = glm::dvec2(distx(rng), disty(rng));
			h_vel[i] = glm::normalize(glm::dvec2(dist01(rng), dist01(rng)));
			//h_vel[i] = glm::dvec2(0.0f);
		}
		h_pos[0].x = -1000.0;
		h_pos[0].y = -1000.0;
		h_pos[1].x = 1000.0;
		h_pos[1].y = -1000.0;
		h_pos[2].x = 1000.0;
		h_pos[2].y = 1000.0;
		h_pos[3].x = -1000.0;
		h_pos[3].y = 1000.0;

		h_vel[0].x = 0.0;
		h_vel[0].y = 0.0;
		h_vel[1].x = 0.0;
		h_vel[1].y = 0.0;
		h_vel[2].x = 0.0;
		h_vel[2].y = 0.0;
		h_vel[3].x = 0.0;
		h_vel[3].y = 0.0;

		cudaMalloc((void**)&d_average_dir, numParticles * sizeof(glm::dvec2));
		cudaMalloc((void**)&d_average_pos, numParticles * sizeof(glm::dvec2));
		cudaMalloc((void**)&d_sep_force, numParticles * sizeof(glm::dvec2));

		cudaMalloc((void**)&d_num_neighbors, numParticles * sizeof(int));
		cudaMalloc((void**)&d_clock_time, numParticles * sizeof(int));

		cudaMemset(d_average_dir, 0.0, numParticles * sizeof(glm::dvec2));
		cudaMemset(d_average_pos, 0.0, numParticles * sizeof(glm::dvec2));
		cudaMemset(d_sep_force, 0.0, numParticles * sizeof(glm::dvec2));
		cudaMemset(d_num_neighbors, 0, numParticles * sizeof(int));
		cudaMemset(d_clock_time, 0, numParticles * sizeof(int));

		cudaMalloc((void**)&d_min, sizeof(glm::dvec2));
		cudaMalloc((void**)&d_max, sizeof(glm::dvec2));
		cudaMemcpy(d_min, &h_min, sizeof(glm::dvec2), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &h_max, sizeof(glm::dvec2), cudaMemcpyHostToDevice);

		cff.numVertices = numParticles;
		cff.average_dir = d_average_dir; cff.average_pos = d_average_pos; cff.sep_force = d_sep_force;
		cff.num_neighbors = d_num_neighbors;
		cff.d_bss = d_bss;
		cff.d_clock_time = d_clock_time;

		glGenBuffers(1, &vbo_pos);
		glGenBuffers(1, &vbo_vel);
		GLCHECKERR();

		glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
		glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::dvec2), &h_pos[0], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_vel);
		glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::dvec2), &h_vel[0], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		GLCHECKERR();

		gpuErrchk(cudaGraphicsGLRegisterBuffer(&vbo_pos_cuda, vbo_pos, cudaGraphicsMapFlagsNone));
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&vbo_vel_cuda, vbo_vel, cudaGraphicsMapFlagsNone));
		GLCHECKERR();


		size_t bytes = 0;
		gpuErrchk(cudaGraphicsMapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &bytes, vbo_pos_cuda));

		m_grid->build(h_pos);
		m_grid->apply_f_frnn_given_prev_info<boids_neighbor_functor>(cff, d_pos, h_bss->view_distance);
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));

		LOG_EVENT("Boids particle system initialized with {} particles", numParticles);
	}

	~Boids2dParticleSys() {
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
		gpuErrchk(cudaGetLastError());
		LOG_TIMING("Data structure update: {}", grid_timer.swap_time());

		cff.pos = d_pos;
		cff.vel = d_vel;
		
		m_grid->apply_f_frnn<boids_neighbor_functor>(cff, d_pos, h_bss->view_distance);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
		LOG_TIMING("Data structure query: {}", grid_timer.swap_time());

		// calc time on forces integration
		//thrust::device_ptr<int> clock_timer = thrust::device_pointer_cast(d_clock_time);
		//thrust::inclusive_scan(clock_timer, clock_timer + numParticles, clock_timer);
		//cudaDeviceSynchronize();
		//int* res = new int[1];
		//cudaMemcpy(res, &d_clock_time[numParticles - 1], sizeof(int), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		// 1395 MHz is the default clock time in the GPU in the lab PC
		//LOG_TIMING("Num cycles on force accumulation: {: 0.3f}ms Calc time: {}ms", ((double)res[0])/1395.0, grid_timer.swap_time());

		integrate << <numBlocks, blocksize >> > (numParticles, d_pos, d_vel, d_bss);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
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