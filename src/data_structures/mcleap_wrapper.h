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

template<int queueSize, int visitedSizePerVertex>
struct triangulation2d {
	triangulation2d(int numP);
	~triangulation2d();
	void update(MCleap::MCLEAP_VEC* pos, MCleap::MCLEAP_VEC* vel);
	void update(MCleap::MCLEAP_VEC* pos);
	void build(MCleap::MCLEAP_VEC* pos);
	template<class Functor>
	void apply_f_frnn(Functor& f, MCleap::MCLEAP_VEC* pos, const double rad);
	float mean_num_particle_in_cell();

	// data
	int numP;
	int blocksize = 64;
	MCleap::mcleap_mesh *m, *m_copy;

	int* d_visited;
	MCleap::MCLEAP_REAL* d_diff;
};

template<int queueSize, int visitedSizePerVertex>
triangulation2d<queueSize, visitedSizePerVertex>::triangulation2d(int numP) : numP(numP) {
	cudaMalloc(&d_diff, sizeof(MCleap::MCLEAP_REAL) * 2 * numP);
	cudaMalloc(&d_visited, sizeof(int) * numP * visitedSizePerVertex);
}

template<int queueSize, int visitedSizePerVertex>
triangulation2d<queueSize, visitedSizePerVertex>::~triangulation2d() {
	cudaFree(d_diff);
	cudaFree(d_visited);
}

template<int queueSize, int visitedSizePerVertex>
void triangulation2d<queueSize, visitedSizePerVertex>::build(MCleap::MCLEAP_VEC* pos) {
	m = MCleap::build_triangulation_from_buffer(numP,(MCLEAP_REAL*)pos);
	m_copy = MCleap::init_empty_mesh(m->num_vertices, m->num_edges, m->num_triangles);
}

template<int queueSize, int visitedSizePerVertex>
void triangulation2d<queueSize, visitedSizePerVertex>::update(MCleap::MCLEAP_VEC* pos, MCleap::MCLEAP_VEC* vel) {
	update(pos);
}

__global__ void calc_diff(int numP, MCleap::MCLEAP_VEC* a, MCleap::MCLEAP_VEC* b, MCleap::MCLEAP_REAL* diff) {
	const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < numP) {
		diff[idx*2+0] = a[idx].x - b[idx].x;
		diff[idx*2+1] = a[idx].y - b[idx].y;
	}
}

template<int queueSize, int visitedSizePerVertex>
void triangulation2d<queueSize, visitedSizePerVertex>::update(MCleap::MCLEAP_VEC* pos) {
	calc_diff <<<numP / blocksize + 1, blocksize >>> (numP, (m->d_vbo_v), (MCleap::MCLEAP_VEC*)pos, d_diff);
	MCleap::move_vertices(m, m_copy, d_diff);
}

template<int queueSize, int visitedSizePerVertex>
template<class Functor>
void triangulation2d<queueSize, visitedSizePerVertex>::apply_f_frnn(Functor& f, MCleap::MCLEAP_VEC* pos, const double rad) {
	calcFRNN_frontier<Functor, queueSize, visitedSizePerVertex> <<<numP / blocksize + 1, blocksize>>> (f, m->d_vbo_v, m->d_t, m->d_mesh->he, m->d_mesh->v_to_he, d_visited, m->num_vertices, rad);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

template<int queueSize, int visitedSizePerVertex>
float triangulation2d<queueSize, visitedSizePerVertex>::mean_num_particle_in_cell() {
	return 0.0;
}