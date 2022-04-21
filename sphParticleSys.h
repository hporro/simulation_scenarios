#pragma once

#include "particleSys.h"
#include "helper.cuh"
#include "grid.h"
#include <algorithm>

struct sph_sim_settings {
	float rad;
	float viscosity;
	float rho0;
	float k; // pressure coeficient
};

__device__ float smoothing_kernel(glm::vec3 r, float h) {
	float q = glm::length(r) / h;
	if (q<0 || q>1) return 0;
	else if (q <= 0.5) return 6*(q*q*q - q*q) + 1;
	else return 2*(1-q)*(1-q)*(1-q);
}

__device__ glm::vec3 D_smoothing_kernel(glm::vec3 r, float h) {
	float q = glm::length(r) / h;
	if (q < 0 || q>1) return glm::vec3(0);
	else if (q <= 0.5) return (6 * r / (h*h)) * (3 * q - 2);
	else return -6 * (1 - q) * (1 - q) * r / (h * glm::length(r));
}

struct sph_density_functor {
	void resetVariables() {
		cudaMemset(rho, 0.0, numParticles * sizeof(float));
		cudaMemset(f_viscosity, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(f_pressure, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(v_guess, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(f_external, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(pressure, 0.0, numParticles * sizeof(glm::vec3));
	}
	__device__ void operator()(const int& i, const int& j, glm::vec3 dist_vec, double dist) {
		rho[i] += smoothing_kernel(dist_vec, d_bss->rad);
		//printf("i: %d rho[i]: %f\n", i, rho[i]);
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	float* rho;
	glm::vec3 *f_viscosity, *f_pressure, *v_guess, *f_external, *pressure;
	sph_sim_settings* d_bss;
};
struct sph_viscosity_functor {
	__device__ void operator()(const int& i, const int& j, glm::vec3 dist_vec, double dist) {
		glm::vec3 lap_V_i = -(1 / rho[j]) * (vel[i] - vel[j]) * 2 * (glm::length(D_smoothing_kernel(dist_vec, d_bss->rad))) / glm::length(dist_vec);
		f_viscosity[i] += d_bss->viscosity * lap_V_i;
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	float *rho;
	glm::vec3 *f_viscosity, *f_pressure, *v_guess, *f_external, *pressure;
	sph_sim_settings* d_bss;
};
struct sph_pressure_functor {
	__device__ void operator()(const int& i, const int& j, glm::vec3 dist_vec, double dist) {
		float p_i = d_bss->k * (rho[i] - d_bss->rho0); // state equation
		float p_j = d_bss->k * (rho[j] - d_bss->rho0); // state equation
		glm::vec3 grad_P = ((p_i)/(rho[i]*rho[i])+(p_j)/(rho[j]*rho[j])) * D_smoothing_kernel(dist_vec,d_bss->rad);
		f_pressure[i] += -grad_P;
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	float *rho;
	glm::vec3 *f_viscosity, *f_pressure, *v_guess, *f_external, *pressure;
	sph_sim_settings* d_bss;
};

__global__ void move_sph_w_walls(int numParticles, glm::vec3* pos, glm::vec3* vel, glm::vec3* min, glm::vec3* max, float dt, sph_sim_settings* bss, glm::vec3* f_viscosity, glm::vec3 *f_external, glm::vec3 *f_pressure) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		
		glm::vec3 v_guess = vel[i] + dt * (f_viscosity[i] + f_external[i]);
		vel[i] = v_guess + dt * (f_pressure[i]);

		if (pos[i].x + dt * vel[i].x > max[0].x)vel[i].x=0;
		if (pos[i].y + dt * vel[i].y > max[0].y)vel[i].y=0;
		if (pos[i].z + dt * vel[i].z > max[0].z)vel[i].z=0;
		if (pos[i].x + dt * vel[i].x < min[0].x)vel[i].x=0;
		if (pos[i].y + dt * vel[i].y < min[0].y)vel[i].y=0;
		if (pos[i].z + dt * vel[i].z < min[0].z)vel[i].z=0;

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
	float* d_rho;
	glm::vec3 *d_f_viscosity, *d_f_pressure, *d_v_guess, *d_f_external, *d_pressure;

	struct cudaGraphicsResource* vbo_pos_cuda, * vbo_vel_cuda;

	SphParticleSys(int numParticles, glm::vec3 min, glm::vec3 max, sph_sim_settings bss) : ParticleSys(numParticles), h_min(min), h_max(max),
		m_grid({ glm::ivec3(20),glm::vec3(5.0),numParticles})
	{
		h_pos = new glm::vec3[numParticles];
		h_vel = new glm::vec3[numParticles];

		// initial position
		std::random_device dev;
		std::mt19937 rng(dev());
		std::uniform_real_distribution<> distx(h_min.x, h_max.x);
		std::uniform_real_distribution<> disty(h_min.x, h_max.x);
		std::uniform_real_distribution<> distz(h_min.x, h_max.x);

		for (int i = 0; i < numParticles; i++) {
			h_pos[i] = glm::vec3(distx(rng), disty(rng), distz(rng));
			h_vel[i] = glm::vec3(0.0);
		}

		cudaMalloc((void**)&d_bss, sizeof(sph_sim_settings));
		h_bss = new sph_sim_settings[1];
		h_bss[0] = bss;
		cudaMemcpy(d_bss, h_bss, sizeof(sph_sim_settings), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_rho, numParticles * sizeof(float));
		cudaMalloc((void**)&d_f_viscosity, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_f_pressure, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_v_guess, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_f_external, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_pressure, numParticles * sizeof(glm::vec3));

		cudaMemset(d_rho        , 0.0, numParticles * sizeof(float));
		cudaMemset(d_f_viscosity, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_f_pressure , 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_v_guess    , 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_f_external , 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_pressure   , 0.0, numParticles * sizeof(glm::vec3));

		cudaMalloc((void**)&d_min, sizeof(glm::vec3));
		cudaMalloc((void**)&d_max, sizeof(glm::vec3));
		cudaMemcpy(d_min, &h_min, sizeof(glm::vec3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &h_max, sizeof(glm::vec3), cudaMemcpyHostToDevice);

		d_func.numParticles = numParticles;
		d_func.rho = d_rho;
		d_func.f_viscosity = d_f_viscosity;
		d_func.f_pressure = d_f_pressure;
		d_func.v_guess = d_v_guess;
		d_func.f_external = d_f_external;
		d_func.pressure = d_pressure;
		d_func.d_bss = d_bss;

		v_func.numParticles = numParticles;
		v_func.rho = d_rho;
		v_func.f_viscosity = d_f_viscosity;
		v_func.f_pressure = d_f_pressure;
		v_func.v_guess = d_v_guess;
		v_func.f_external = d_f_external;
		v_func.pressure = d_pressure;
		v_func.d_bss = d_bss;

		p_func.numParticles = numParticles;
		p_func.rho = d_rho;
		p_func.f_viscosity = d_f_viscosity;
		p_func.f_pressure = d_f_pressure;
		p_func.v_guess = d_v_guess;
		p_func.f_external = d_f_external;
		p_func.pressure = d_pressure;
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

		m_grid.update(d_pos);
		cudaDeviceSynchronize();
		
		d_func.vel = d_vel;
		v_func.vel = d_vel;
		p_func.vel = d_vel;

		m_grid.apply_f_frnn<sph_density_functor>  (d_func, d_pos, h_bss->rad);
		m_grid.apply_f_frnn<sph_viscosity_functor>(v_func, d_pos, h_bss->rad);
		m_grid.apply_f_frnn<sph_pressure_functor> (p_func, d_pos, h_bss->rad);

		cudaDeviceSynchronize();
		LOG_TIMING("Grid update: {}", grid_timer.swap_time());

		move_sph_w_walls << <numBlocks + 1, blocksize >> > (numParticles, d_pos, d_vel, d_min, d_max, dt, d_bss, d_f_viscosity, d_f_external, d_f_pressure);
		cudaDeviceSynchronize();
		LOG_TIMING("Integration: {}", grid_timer.swap_time());

		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_vel_cuda, 0));

		LOG_EVENT("sph particle system updated");
	}

};