#pragma once

#include "particleSys.h"
#include "helper.cuh"
#include "grid2.h"
#include <algorithm>

constexpr float H_PI = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208;

struct sph_sim_settings {
	float k = 1000.0;
	float viscosity = 250.0;
	float density = 1000;
	float gravity = -9.8;
	glm::vec4 box_size = glm::vec4(100.0);
	float spacing = (float)(0.5*100.0/pow(100000,1.0/3.0));
	float dt = 0.001;
	float h = (float)(2.0 * spacing);
	float mass = (float)(pow(spacing, 3) * density);
	float ColRestitution = 0.2;
};

__device__ float kernelM4(float x, sph_sim_settings* bss) {
	const float q = x / bss->h;
	if (q >= 1.f)
		return 0.f;

	const float factor = 2.546479089470325472 / pow(bss->h,3);
	if (q < .5f)
		return factor * (1.f - 6.f * q * q * (1 - q));
	float a = 1.f - q;
	return factor * 2.f * a * a * a;
}

__device__ float kernelM4_d(const float x, sph_sim_settings* bss) {
	const float q = x / bss->h;

	if (q >= 1.f)
		return 0.f;

	const float factor = 2.546479089470325472 / pow(bss->h,5) ;
	if (q < .5f)
		return factor * (-12.f + 18.f * q);

	return factor * (-6.f * (1.f - q) * (1.f - q) / q);
}

__device__ float kernelVisc_dd(const float x, sph_sim_settings* bss) {
	const float q = x / bss->h;
	if (q <= 1.f)
		return -45.f / (H_PI * pow(bss->h,5)) * (1 - q);

	return 0.f;
}

struct sph_density_functor {
	void resetVariables() {
		cudaMemset(rho, 0.0, numParticles * sizeof(float));
		cudaMemset(f_viscosity, 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(f_pressure, 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(v_guess, 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(f_external, 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(pressure, 0.0, numParticles * sizeof(float));
	}
	__device__ void operator()(const int& i, const int& j, glm::vec4 dist_vec, double dist) {
		rho[i] += kernelM4(dist, d_bss);
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec4* vel;
	float* rho;
	glm::vec4* f_viscosity, * f_pressure, * v_guess, * f_external;
	float *pressure;
	sph_sim_settings* d_bss;
};
struct sph_forces_functor {
	__device__ void operator()(const int& i, const int& j, glm::vec4 dist_vec, double dist) {
		if (i == j || dist < 0.0000001) return;
		if (rho[i] == 0.0) return;
		f_viscosity[i]+= (vel[i] - vel[j]) * (d_bss->k / rho[i] * kernelVisc_dd(dist, d_bss));
		f_pressure[i] -= (dist_vec)/dist * (0.5f * (pressure[i] + pressure[j]) / rho[i] * kernelM4_d(dist, d_bss));
	}
	int numParticles;

	// this class dont own this stuff
	glm::vec4* vel;
	float *rho;
	glm::vec4 *f_viscosity, *f_pressure, *v_guess, *f_external;
	float *pressure;
	sph_sim_settings* d_bss;
};

__device__ float sdCircle(glm::vec4 p, float r)
{
	return glm::length(p) - r;
}

__device__ glm::vec4 normal_circle(glm::vec4 p, float r) // for function f(p)
{
	float eps = 0.001; // or some other value
	glm::vec4 dx(eps, 0, 0, 0);
	glm::vec4 dy(0, eps, 0, 0);
	glm::vec4 dz(0, 0, eps, 0);
	return glm::normalize(glm::vec4(sdCircle(p + dx, r) - sdCircle(p - dx, r),
									sdCircle(p + dy, r) - sdCircle(p - dy, r),
									sdCircle(p + dz, r) - sdCircle(p - dz, r), 
									0.0));
}


template<class T>
__device__ T h_max(T a, T b) {
	return a > b ? a : b;
}

template<class T>
__device__ T h_min(T a, T b) {
	return a < b ? a : b;
}

__device__ glm::vec4 v_max(glm::vec4 a, glm::vec4 b) {
	return glm::vec4(h_max(a.x, b.x), h_max(a.y, b.y), h_max(a.z, b.z), h_max(a.w, b.w));
}

__device__ float v_max(glm::vec4 a) {
	return h_max(h_max(a.x, a.y), h_max(a.z, a.w));
}

__device__ float sdBox(glm::vec4 q, glm::vec4 b)
{
	glm::vec4 d = glm::abs(q) - b;
	return glm::length(v_max(d, glm::vec4(0.0))) + h_min(v_max(d), 0.0f);
}

__device__ glm::vec4 normal_bx(glm::vec4 p, glm::vec4 b) // for function f(p)
{
	float eps = 0.001; // or some other value
	glm::vec4 dx(eps, 0, 0, 0.0);
	glm::vec4 dy(0, eps, 0, 0.0);
	glm::vec4 dz(0, 0, eps, 0.0);
	return glm::normalize(glm::vec4(sdBox(p + dx, b) - sdBox(p - dx, b),
									sdBox(p + dy, b) - sdBox(p - dy, b),
									sdBox(p + dz, b) - sdBox(p - dz, b),
									0.0));
}

__global__ void move_sph_w_walls(int numParticles, glm::vec4* pos, glm::vec4* vel, glm::vec4* min, glm::vec4* max, sph_sim_settings* bss, glm::vec4* f_viscosity, glm::vec4 *f_external, glm::vec4 *f_pressure, float* rho) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		
		//if (vel[i] != vel[i])printf("BEFORE i: %d vel: %f\n", i, glm::length(vel[i]));
		glm::vec4 acel = (f_viscosity[i] + f_pressure[i])*(bss->mass/rho[i]) + glm::vec4(0.0f,bss->gravity,0.0f,0.0f);
		if (acel != acel) {
			acel = glm::vec4(0.0f, bss->gravity, 0.0f, 0.0f);
		}
		//printf("acel: %f bss->dt: %f\n", glm::length(acel), bss->dt);
		vel[i] += acel * bss->dt;
		//printf("vel: %f\n", glm::length(vel[i]));
		//if (vel[i] != vel[i])printf("rho[i]: %f\n", i, rho[i]);
		pos[i] += vel[i] * bss->dt;


		// Boundary conditions
		glm::vec4 b(50.0);
		float d = sdBox(pos[i], b) + bss->h;
		glm::vec4 n = normal_bx(pos[i],b);
		//float d = sdCircle(pos[i], 50) + bss->h;
		//glm::vec4 n = normal_circle(pos[i], 50);
		if (d > 0.)
		{
			pos[i] += n * -d;
			vel[i] = reflect(vel[i], n);
			d = glm::dot(vel[i], n);
			vel[i] -= n * d * bss->ColRestitution;
			//pos[i] += vel[i] * bss->dt;
		}

		//glm::vec4 v_guess = vel[i] + dt * (f_viscosity[i] + f_external[i]);
		//vel[i] = v_guess + dt * (f_pressure[i] + glm::vec4(0.0f,0.01f,0.0f));
		//
		//if (pos[i].x + dt * vel[i].x > max[0].x)vel[i].x=0;
		//if (pos[i].y + dt * vel[i].y > max[0].y)vel[i].y=0;
		//if (pos[i].z + dt * vel[i].z > max[0].z)vel[i].z=0;
		//if (pos[i].x + dt * vel[i].x < min[0].x)vel[i].x=0;
		//if (pos[i].y + dt * vel[i].y < min[0].y)vel[i].y=0;
		//if (pos[i].z + dt * vel[i].z < min[0].z)vel[i].z=0;

		//pos[i] += dt * vel[i];
	}
}

__global__ void calc_pressure(int numParticles, float* pressure, float* rho, sph_sim_settings* d_bss) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numParticles) {
		rho[i] *= d_bss->mass;
		pressure[i] = h_max(0.f, d_bss->k * (rho[i] - d_bss->density));
	}
}

struct SphParticleSys : public ParticleSys {

	// simulation specific behaviour
	sph_sim_settings* d_bss;
	sph_sim_settings* h_bss;
	sph_density_functor   d_func;
	sph_forces_functor  p_func;

	// parsys variables
	glm::vec4* d_vel, * h_vel;
	glm::vec4* d_pos, * h_pos;
	glm::vec4  h_min, h_max;
	glm::vec4* d_min, * d_max;
	GridCount* m_grid;

	// simulation specific variables
	float* d_rho;
	glm::vec4 *d_f_viscosity, *d_f_pressure, *d_v_guess, *d_f_external; 
	float *d_pressure;

	struct cudaGraphicsResource* vbo_pos_cuda, * vbo_vel_cuda;

	SphParticleSys(int numParticles, glm::vec4 min, glm::vec4 max, sph_sim_settings bss) : ParticleSys(numParticles), h_min(min), h_max(max)
	{
		h_pos = new glm::vec4[numParticles];
		h_vel = new glm::vec4[numParticles];

		// initial position
		std::random_device dev;
		std::mt19937 rng(dev());
		std::uniform_real_distribution<> dista(0.0, 0.01);

		cudaMalloc((void**)&d_bss, sizeof(sph_sim_settings));
		h_bss = new sph_sim_settings[1];
		h_bss[0] = bss;
		cudaMemcpy(d_bss, h_bss, sizeof(sph_sim_settings), cudaMemcpyHostToDevice);

		printf("h size: %f\n", h_bss->h);
		m_grid = new GridCount(numParticles, min, glm::vec4(h_bss->h), glm::ivec4(ceil(100.0 / h_bss->h)));
			
		int numx = 22;
		int numy = 22;
		int numz = 22;
		
		for (int z = 0; z < numz; z++)
			for (int y = 0; y < numy; y++)
				for (int x = 0; x < numx; x++){
					int i = x + numx * (y + numy * z);
					if (i >= numParticles)break;
					h_pos[i] = glm::vec4(-15 + h_bss->spacing * (x + 0.5), -13 + h_bss->spacing * (y + 0.5), -10 + h_bss->spacing * (z + 0.5), 0.0);
					//h_pos[i] = glm::vec4(distx(rng), disty(rng), distz(rng));
					h_vel[i] = glm::vec4(0.0f);
				}

		cudaMalloc((void**)&d_rho, numParticles * sizeof(float));
		cudaMalloc((void**)&d_f_viscosity, numParticles * sizeof(glm::vec4));
		cudaMalloc((void**)&d_f_pressure, numParticles *  sizeof(glm::vec4));
		cudaMalloc((void**)&d_v_guess, numParticles *     sizeof(glm::vec4));
		cudaMalloc((void**)&d_f_external, numParticles *  sizeof(glm::vec4));
		cudaMalloc((void**)&d_pressure, numParticles *    sizeof(float));

		cudaMemset(d_rho        , 0.0, numParticles * sizeof(float));
		cudaMemset(d_f_viscosity, 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(d_f_pressure , 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(d_v_guess    , 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(d_f_external , 0.0, numParticles * sizeof(glm::vec4));
		cudaMemset(d_pressure   , 0.0, numParticles * sizeof(float));

		cudaMalloc((void**)&d_min, sizeof(glm::vec4));
		cudaMalloc((void**)&d_max, sizeof(glm::vec4));
		cudaMemcpy(d_min, &h_min,  sizeof(glm::vec4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &h_max,  sizeof(glm::vec4), cudaMemcpyHostToDevice);

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
		glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::vec4), &h_pos[0], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_vel);
		glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::vec4), &h_vel[0], GL_DYNAMIC_DRAW);
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


		m_grid->apply_f_frnn<sph_density_functor>  (d_func, d_pos, h_bss->h);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
		calc_pressure<<<numBlocks + 1, blocksize >>>(numParticles, d_pressure, d_rho, d_bss);
		cudaDeviceSynchronize();
		m_grid->apply_f_frnn<sph_forces_functor> (p_func, d_pos, h_bss->h);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());

		LOG_TIMING("Grid query: {}", grid_timer.swap_time());

		move_sph_w_walls <<< numBlocks + 1, blocksize >>> (numParticles, d_pos, d_vel, d_min, d_max, d_bss, d_f_viscosity, d_f_external, d_f_pressure, d_rho);
		cudaDeviceSynchronize();
		LOG_TIMING("Integration: {}", grid_timer.swap_time());

		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_pos_cuda, 0));
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_vel_cuda, 0));

		LOG_EVENT("sph particle system updated");
	}

};