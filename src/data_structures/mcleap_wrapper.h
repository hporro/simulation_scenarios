#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>

#include <MCleap.h>
#include "../logging/Logging.h"
#include "../gpu/gpuErrCheck.h"
#include <frnn_kernels.cuh>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>


#include <algorithm>
#include <iostream>

template<int max_neighbors, int max_attracted>
struct triangulation2d {
	triangulation2d(int numP);
	~triangulation2d();
	void update(glm::dvec2* pos, glm::dvec2* vel);
	void update(glm::dvec2* pos);
	void build(glm::dvec2* pos);
	template<class Functor>
	void apply_f_frnn(Functor& f, glm::dvec2* pos, const double rad);
	template<class Functor>
	void apply_f_frnn_given_prev_info(Functor& f, glm::dvec2* pos, const double rad);
	void calc_attracted(glm::dvec2* pos, const double rad);
	float mean_num_particle_in_cell();
	
	// data
	int numP;
	int blocksize = 64;
	MCleap::mcleap_mesh *m, *m_copy;

	int* d_attracted, *d_neighbors;
	MCLEAP_REAL* d_diff;
};

template<int max_neighbors, int max_attracted>
triangulation2d<max_neighbors, max_attracted>::triangulation2d(int numP) : numP(numP) {
	cudaMalloc(&d_diff, sizeof(MCLEAP_REAL) * 2 * numP);
	cudaMalloc(&d_attracted, sizeof(int) * numP * max_attracted);
	cudaMalloc(&d_neighbors, sizeof(int) * numP * max_neighbors);
}

template<int max_neighbors, int max_attracted>
triangulation2d<max_neighbors, max_attracted>::~triangulation2d() {
	cudaFree(d_diff);
	cudaFree(d_attracted);
	cudaFree(d_neighbors);
}

template<int max_neighbors, int max_attracted>
void triangulation2d<max_neighbors, max_attracted>::build(glm::dvec2* pos) {
	m = MCleap::build_triangulation_from_buffer(numP,(MCLEAP_REAL*)pos);
	m_copy = MCleap::init_empty_mesh(m->num_vertices, m->num_edges, m->num_triangles);
	cudaDeviceSynchronize();
}

template<int max_neighbors, int max_attracted>
void triangulation2d<max_neighbors, max_attracted>::update(glm::dvec2* pos, glm::dvec2* vel) {
	update(pos);
}

__global__ void calc_diff(int numP, glm::dvec2* a, glm::dvec2* b, MCLEAP_REAL* diff) {
	const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < numP) {
		diff[idx*2+0] = a[idx].x - b[idx].x;
		diff[idx*2+1] = a[idx].y - b[idx].y;
	}
}

template<int max_neighbors, int max_attracted>
void triangulation2d<max_neighbors, max_attracted>::update(glm::dvec2* pos) {
	calc_diff <<<numP / blocksize + 1, blocksize >>> (numP, m->d_vbo_v, pos, d_diff);
	MCleap::move_vertices(m, m_copy, d_diff);
}

template<int max_neighbors, int max_attracted>
void triangulation2d<max_neighbors, max_attracted>::calc_attracted(glm::dvec2* pos, const double rad) {
	cudaDeviceSynchronize();
	Timer tim;
	calc_neighbors_kernel<max_neighbors> << <numP / blocksize + 1, blocksize >> > (pos, m->d_mesh->he, m->d_mesh->v_to_he, d_neighbors, m->num_vertices);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] calc neighbors: {}", tim.swap_time());
	calcAttracted_kernel<max_neighbors, max_attracted> << <numP / blocksize + 1, blocksize >> > (pos, d_neighbors, d_attracted, m->num_vertices, rad);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] calc attracted: {}", tim.swap_time());
}

template<int max_neighbors, int max_attracted>
template<class Functor>
void triangulation2d<max_neighbors, max_attracted>::apply_f_frnn(Functor& f, glm::dvec2* pos, const double rad) {

	cudaDeviceSynchronize();
	Timer tim;
	calc_neighbors_kernel<max_neighbors> << <numP / blocksize + 1, blocksize >> > (pos, m->d_mesh->he, m->d_mesh->v_to_he, d_neighbors, m->num_vertices);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] calc neighbors: {}", tim.swap_time());
	calcAttracted_kernel<max_neighbors, max_attracted> << <numP / blocksize + 1, blocksize >> > (pos, d_neighbors, d_attracted, m->num_vertices, rad);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] calc attracted: {}", tim.swap_time());
	frnn_given_attracted<Functor, max_attracted> << <numP / blocksize + 1, blocksize >> > (f, pos, d_attracted, m->num_vertices);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] apply functor: {}", tim.swap_time());
}

template<int max_neighbors, int max_attracted>
template<class Functor>
void triangulation2d<max_neighbors, max_attracted>::apply_f_frnn_given_prev_info(Functor& f, glm::dvec2* pos, const double rad) {
	cudaDeviceSynchronize();
	Timer tim;
	calc_neighbors_kernel<max_neighbors> << <numP / blocksize + 1, blocksize >> > (pos, m->d_mesh->he, m->d_mesh->v_to_he, d_neighbors, m->num_vertices);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] calc neighbors: {}", tim.swap_time());
	recalcAttracted_kernel<max_neighbors, max_attracted> << <numP / blocksize + 1, blocksize >> > (pos, d_neighbors, d_attracted, m->num_vertices, rad);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] calc reattracted: {}", tim.swap_time());
	frnn_given_attracted<Functor, max_attracted> << <numP / blocksize + 1, blocksize >> > (f, pos, d_attracted, m->num_vertices);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("[triangulation 2d] apply functor: {}", tim.swap_time());
}


template<int max_neighbors, int max_attracted>
float triangulation2d<max_neighbors, max_attracted>::mean_num_particle_in_cell() {
	return 0.0;
}