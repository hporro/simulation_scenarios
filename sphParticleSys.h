#pragma once

#include "particleSys.h"
#include "grid.h"
#include <algorithm>

struct sph_sim_settings {
	float rad;
	float viscosity;
};

__device__ float smoothing_kernel(glm::vec3 r, float h) {
	float q = glm::length(r) / h;
	if (q<0 || q>1) return 0;
	else if (q <= 0.5) return 6*(q*q*q - q*q) + 1;
	else return 2*(1-q)*(1-q)*(1-q);
}

struct sph_density_functor {
	void resetVariables() {
		cudaMemset(rho, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(f_viscosity, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(f_pressure, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(v_guess, 0.0, numParticles * sizeof(glm::vec3));
	}
	__device__ void operator()(const int& i, const int& j, glm::vec3 dist_vec, double dist) {
		rho[i] += 1 * smoothing_kernel(dist_vec, d_bss->rad);
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	glm::vec3 *rho, *f_viscosity, *f_pressure, *v_guess;
	sph_sim_settings* d_bss;
};
struct sph_viscosity_functor {
	__device__ void operator()(const int& i, const int& j, glm::vec3 dist_vec, double dist) {
		//f_viscosity[i] = - (1/rho[j])*(A[i]-A[j])*(2*)
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	glm::vec3* rho, * f_viscosity, * f_pressure, * v_guess;
	sph_sim_settings* d_bss;
};
struct sph_pressure_functor {
	__device__ void operator()(const int& i, const int& j, glm::vec3 dist_vec, double dist) {}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	glm::vec3* rho, * f_viscosity, * f_pressure, * v_guess;
	sph_sim_settings* d_bss;
};

__global__ void move_sph_w_walls(int numParticles, glm::vec3* pos, glm::vec3* vel, glm::vec3* min, glm::vec3* max, float dt, sph_sim_settings* bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {

		//if (num_A[i] > 0)separation[i] = (separation[i] * bss->A_FORCE) / (float)num_A[i];
		//if (num_B[i] > 0)cohesion[i] = (cohesion[i] * bss->B_FORCE) / (float)num_B[i];
		//if (num_C[i] > 0)alignement[i] = (alignement[i] * bss->C_FORCE) / (float)num_C[i];
		//
		//glm::vec3 acel = separation[i] + cohesion[i] + alignement[i];
		//
		//vel[i] += dt * acel;
		//
		//vel[i].x = clamp(vel[i].x, -bss->MAX_VEL, bss->MAX_VEL);
		//vel[i].y = clamp(vel[i].y, -bss->MAX_VEL, bss->MAX_VEL);
		//vel[i].z = clamp(vel[i].z, -bss->MAX_VEL, bss->MAX_VEL);

		if (pos[i].x + dt * vel[i].x > max[0].x)vel[i].x *= -1.0;
		if (pos[i].y + dt * vel[i].y > max[0].y)vel[i].y *= -1.0;
		if (pos[i].z + dt * vel[i].z > max[0].z)vel[i].z *= -1.0;
		if (pos[i].x + dt * vel[i].x < min[0].x)vel[i].x *= -1.0;
		if (pos[i].y + dt * vel[i].y < min[0].y)vel[i].y *= -1.0;
		if (pos[i].z + dt * vel[i].z < min[0].z)vel[i].z *= -1.0;

		pos[i] += dt * vel[i];
	}
}

struct SphParticleSys : public ParticleSys {

	// simulation specific behaviour
	sph_sim_settings* d_bss;
	sph_sim_settings* h_bss;
	sph_density_functor   d_func;
	sph_viscosity_functor v_func;
	sph_pressure_functor  p_func;

	// parsys variables
	glm::vec3* d_vel, * h_vel;
	glm::vec3* d_pos, * h_pos;
	glm::vec3  h_min, h_max;
	glm::vec3* d_min, * d_max;
	Grid m_grid;

	// simulation specific variables
	glm::vec3* d_rho, *d_f_viscosity, *d_f_pressure, *d_v_guess;

	struct cudaGraphicsResource* vbo_pos_cuda, * vbo_vel_cuda;

	SphParticleSys(int numParticles, glm::vec3 min, glm::vec3 max, sph_sim_settings bss) : ParticleSys(numParticles), h_min(min), h_max(max),
		m_grid({ glm::ivec3(2,2,2),glm::vec3(50.0,50.0,50.0),numParticles})
	{
		h_pos = new glm::vec3[numParticles];
		h_vel = new glm::vec3[numParticles];

		// initial position
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

		cudaMalloc((void**)&d_bss, sizeof(sph_sim_settings));
		h_bss = new sph_sim_settings[1];
		h_bss[0] = bss;
		cudaMemcpy(d_bss, h_bss, sizeof(sph_sim_settings), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_rho, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_f_viscosity, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_f_pressure, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_v_guess, numParticles * sizeof(glm::vec3));

		cudaMemset(d_rho        , 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_f_viscosity, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_f_pressure , 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_v_guess    , 0.0, numParticles * sizeof(glm::vec3));
		//cudaMemset(d_num_A, 0, numParticles * sizeof(int));
		//cudaMemset(d_num_B, 0, numParticles * sizeof(int));
		//cudaMemset(d_num_C, 0, numParticles * sizeof(int));

		cudaMalloc((void**)&d_min, sizeof(glm::vec3));
		cudaMalloc((void**)&d_max, sizeof(glm::vec3));
		cudaMemcpy(d_min, &h_min, sizeof(glm::vec3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &h_max, sizeof(glm::vec3), cudaMemcpyHostToDevice);

		d_func.numParticles = numParticles;
		d_func.rho = d_rho;
		d_func.f_viscosity = d_f_viscosity;
		d_func.f_pressure = d_f_pressure;
		d_func.v_guess = d_v_guess;
		d_func.d_bss = d_bss;

		v_func.numParticles = numParticles;
		v_func.rho = d_rho;
		v_func.f_viscosity = d_f_viscosity;
		v_func.f_pressure = d_f_pressure;
		v_func.v_guess = d_v_guess;
		v_func.d_bss = d_bss;

		p_func.numParticles = numParticles;
		p_func.rho = d_rho;
		p_func.f_viscosity = d_f_viscosity;
		p_func.f_pressure = d_f_pressure;
		p_func.v_guess = d_v_guess;
		p_func.d_bss = d_bss;


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

		LOG_EVENT("sph particle system initialized with {} particles", numParticles);
	}

	~SphParticleSys() {
		delete[] h_pos;
		delete[] h_vel;

		cudaFree(d_rho);
		cudaFree(d_f_viscosity);
		cudaFree(d_f_pressure);
		cudaFree(d_v_guess);

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

		d_func.resetVariables();

		int blocksize = 128;
		int numBlocks = numParticles / blocksize;

		Timer grid_timer;

		glm::vec3 offset(50.0, 50.0, 50.0);
		m_grid.computeGrid(d_pos, offset);
		cudaDeviceSynchronize();
		
		d_func.vel = d_vel;
		v_func.vel = d_vel;
		p_func.vel = d_vel;

		m_grid.apply_f_frnn<sph_density_functor>  (d_func, d_pos, h_bss->rad);
		m_grid.apply_f_frnn<sph_viscosity_functor>(v_func, d_pos, h_bss->rad);
		m_grid.apply_f_frnn<sph_pressure_functor> (p_func, d_pos, h_bss->rad);

		cudaDeviceSynchronize();
		LOG_TIMING("Grid update: {}", grid_timer.swap_time());

		move_sph_w_walls << <numBlocks + 1, blocksize >> > (numParticles, d_pos, d_vel, d_min, d_max, dt, d_bss);
		cudaDeviceSynchronize();
		LOG_TIMING("Integration: {}", grid_timer.swap_time());

		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_vel_cuda, 0));

		LOG_EVENT("sph particle system updated");
	}

};