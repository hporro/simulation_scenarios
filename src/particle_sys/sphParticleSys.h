#pragma once

#include "particleSys.h"
#include "../math/batea_math.cuh"
#include "../data_structures/gridCount.h"
#include <algorithm>

struct sph_sim_settings {
	glm::vec3 ExtForce = glm::vec3(0.0f, 1200. * -9.8, 0.0f);
	float RestDensity = 1000.;
	float GasConst = 2000.;
	float KernelRadius = 2.0;
	float PartMass = 65.;
	float Viscosity = 250.;
	float ColRestitution = 0.20f;
	float Poly6 = 315.f / (65.f * H_PI * powf(KernelRadius, 9.f));
	float SpikyGrad = -45.f / (H_PI * powf(KernelRadius, 6.f));
	float ViscLap = 45.f / (H_PI * powf(KernelRadius, 6.f));
	float dt = 0.001;
};

struct sph_density_functor {
	void resetVariables() {
		cudaMemset(rho, 0.0, numParticles * sizeof(float));
		cudaMemset(f_viscosity, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(f_pressure, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(v_guess, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(f_external, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(pressure, 0.0, numParticles * sizeof(float));
	}
	__device__ void operator()(const int& i, const int& j, const glm::vec3 dist_vec, const double dist) {
		if (dist >= d_bss->KernelRadius)return;
		rho[i] += d_bss->PartMass * d_bss->Poly6 * pow(d_bss->KernelRadius * d_bss->KernelRadius - dist * dist, 3.);
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	float* rho;
	glm::vec3* f_viscosity, * f_pressure, * v_guess, * f_external;
	float* pressure;
	sph_sim_settings* d_bss;
};
struct sph_forces_functor {
	__device__ void operator()(const int& i, const int& j, const glm::vec3 dist_vec, const float dist) {
		if (i == j)return;
		if (dist > d_bss->KernelRadius || dist <= 0)return;

		glm::vec3 dnorm = dist_vec / dist;
		// pressureForce
		f_pressure[i] += -dnorm * d_bss->PartMass * (pressure[i] + pressure[j]) / (2.f * rho[i]) * d_bss->SpikyGrad * powf(d_bss->KernelRadius - dist, 2.f);
		// viscosityForce
		f_viscosity[i] += (vel[j] - vel[i]) * d_bss->Viscosity * d_bss->PartMass / rho[i] * d_bss->ViscLap * (d_bss->KernelRadius - dist);
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec3* vel;
	float* rho;
	glm::vec3* f_viscosity, * f_pressure, * v_guess, * f_external;
	float* pressure;
	sph_sim_settings* d_bss;
};

__global__ void move_sph_w_walls(int numParticles, glm::vec3* pos, glm::vec3* vel, glm::vec3* min, glm::vec3* max, sph_sim_settings* bss, glm::vec3* f_viscosity, glm::vec3* f_external, glm::vec3* f_pressure, float* rho) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {

		glm::vec3 acel = (f_viscosity[i] + f_pressure[i] + bss->ExtForce) / rho[i];
		if (acel != acel) {
			acel = bss->ExtForce;
		}
		vel[i] += acel * bss->dt;
		pos[i] += vel[i] * bss->dt;


		// Boundary conditions
		glm::vec3 b(50.0);
		//float d = sdBox(pos[i], b) + bss->h;
		//glm::vec3 n = normal_bx(pos[i],b);
		float d = sdCircle(pos[i], 50) + bss->KernelRadius;
		glm::vec3 n = normal_circle(pos[i], 50);
		if (d > 0.)
		{
			pos[i] += n * -d;
			vel[i] = reflect(vel[i], n);
			d = glm::dot(vel[i], n);
			vel[i] -= n * d * bss->ColRestitution;
			pos[i] += vel[i] * bss->dt;
		}
	}
}

__global__ void calc_pressure(int numParticles, float* pressure, float* rho, sph_sim_settings* d_bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		pressure[i] = d_bss->GasConst * (rho[i] - d_bss->RestDensity);;
	}
}

struct SphParticleSys : public ParticleSys {

	// simulation specific behaviour
	sph_sim_settings* d_bss;
	sph_sim_settings* h_bss;
	sph_density_functor   d_func;
	sph_forces_functor  p_func;

	// parsys variables
	glm::vec3* d_vel, * h_vel;
	glm::vec3* d_pos, * h_pos;
	glm::vec3  h_min, h_max;
	glm::vec3* d_min, * d_max;
	GridCount* m_grid;

	// simulation specific variables
	float* d_rho;
	glm::vec3* d_f_viscosity, * d_f_pressure, * d_v_guess, * d_f_external;
	float* d_pressure;

	struct cudaGraphicsResource* vbo_pos_cuda, * vbo_vel_cuda;

	SphParticleSys(int numParticles, glm::vec3 min, glm::vec3 max, sph_sim_settings bss) : ParticleSys(numParticles), h_min(min), h_max(max)
	{
		h_pos = new glm::vec3[numParticles];
		h_vel = new glm::vec3[numParticles];

		// initial position
		std::random_device dev;
		std::mt19937 rng(dev());
		std::uniform_real_distribution<> dista(0.0, 0.01);

		cudaMalloc((void**)&d_bss, sizeof(sph_sim_settings));
		h_bss = new sph_sim_settings[1];
		0[h_bss] = bss;
		cudaMemcpy(d_bss, h_bss, sizeof(sph_sim_settings), cudaMemcpyHostToDevice);

		m_grid = new GridCount(numParticles, min, glm::vec3(h_bss->KernelRadius), glm::ivec3(ceil(100.0 / h_bss->KernelRadius)));

		int numx = 22;
		int numy = 22;
		int numz = 22;

		for (int z = 0; z < numz; z++)
			for (int y = 0; y < numy; y++)
				for (int x = 0; x < numx; x++) {
					int i = x + numx * (y + numy * z);
					if (i >= numParticles)break;
					h_pos[i] = glm::vec3(-15 + h_bss->KernelRadius * (x + 0.5), -13 + h_bss->KernelRadius * (y + 0.5), -10 + h_bss->KernelRadius * (z + 0.5));
					//h_pos[i] = glm::vec3(distx(rng), disty(rng), distz(rng));
					h_vel[i] = glm::vec3(0.0f, 0.0f, 0.0f);
				}

		cudaMalloc((void**)&d_rho, numParticles * sizeof(float));
		cudaMalloc((void**)&d_f_viscosity, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_f_pressure, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_v_guess, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_f_external, numParticles * sizeof(glm::vec3));
		cudaMalloc((void**)&d_pressure, numParticles * sizeof(float));

		cudaMemset(d_rho, 0.0, numParticles * sizeof(float));
		cudaMemset(d_f_viscosity, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_f_pressure, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_v_guess, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_f_external, 0.0, numParticles * sizeof(glm::vec3));
		cudaMemset(d_pressure, 0.0, numParticles * sizeof(float));

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
		delete m_grid;

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

		m_grid->update(d_pos, d_vel);
		cudaDeviceSynchronize();

		LOG_TIMING("Grid update: {}", grid_timer.swap_time());

		LOG_EVENT("Mean num of particles: {}", m_grid->mean_num_particle_in_cell());
		LOG_TIMING("Calc mean num of particles: {}", grid_timer.swap_time());

		d_func.vel = d_vel;
		p_func.vel = d_vel;
		gpuErrchk(cudaGetLastError());

		m_grid->apply_f_frnn<sph_density_functor>(d_func, d_pos, h_bss->KernelRadius);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
		calc_pressure << <numBlocks + 1, blocksize >> > (numParticles, d_pressure, d_rho, d_bss);
		cudaDeviceSynchronize();
		m_grid->apply_f_frnn<sph_forces_functor>(p_func, d_pos, h_bss->KernelRadius);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		LOG_TIMING("Grid query: {}", grid_timer.swap_time());

		move_sph_w_walls << < numBlocks + 1, blocksize >> > (numParticles, d_pos, d_vel, d_min, d_max, d_bss, d_f_viscosity, d_f_external, d_f_pressure, d_rho);
		cudaDeviceSynchronize();
		LOG_TIMING("Integration: {}", grid_timer.swap_time());

		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_vel_cuda, 0));

		LOG_EVENT("sph particle system updated");
	}

};