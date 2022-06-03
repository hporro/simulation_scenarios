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

#include <algorithm>
#include <iostream>

template<int queueSize, int visitedSizePerVertex>
struct triangulation2d {
	triangulation2d(int numP);
	~triangulation2d();
	void update(glm::vec2* pos, glm::vec2* vel);
	void update(glm::vec2* pos);
	void build(glm::vec2* pos);
	template<class Functor>
	void apply_f_frnn(Functor& f, glm::vec2* pos, const float rad);
	float mean_num_particle_in_cell();

	// data
	int numP;
	int blocksize = 64;
	MCleap::mcleap_mesh *m;

	int* d_visited;
	glm::vec2* d_diff;
};

template<int queueSize, int visitedSizePerVertex>
triangulation2d<queueSize, visitedSizePerVertex>::triangulation2d(int numP) : numP(numP) {
	cudaMalloc(&d_diff, sizeof(glm::vec2) * numP);
	cudaMalloc(&d_visited, sizeof(int) * numP * visitedSizePerVertex);
}

template<int queueSize, int visitedSizePerVertex>
triangulation2d<queueSize, visitedSizePerVertex>::~triangulation2d() {
	cudaFree(d_diff);
	cudaFree(d_visited);
}

template<int queueSize, int visitedSizePerVertex>
void triangulation2d<queueSize, visitedSizePerVertex>::build(glm::vec2* pos) {
	m = MCleap::build_triangulation_from_buffer(numP,pos);
}

template<int queueSize, int visitedSizePerVertex>
void triangulation2d<queueSize, visitedSizePerVertex>::update(glm::vec2* pos, glm::vec2* vel) {
	update(pos);
}

__global__ void calc_diff(int numP, glm::vec2* a, glm::vec2* b, glm::vec2* diff) {
	const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < numP) {
		diff[idx] = a[idx] - b[idx];
	}
}

template<int queueSize, int visitedSizePerVertex>
void triangulation2d<queueSize, visitedSizePerVertex>::update(glm::vec2* pos) {
	calc_diff <<<numP / blocksize + 1, blocksize >>> (numP, m->d_vbo_v, glm::value_ptr(pos), d_diff);
	MCleap::move_vertices(m, d_diff);
}

template<int queueSize, int visitedSizePerVertex>
template<class Functor>
void triangulation2d<queueSize, visitedSizePerVertex>::apply_f_frnn(Functor& f, glm::vec2* pos, const float rad) {
	calcFRNN_frontier<Functor, queueSize, visitedSizePerVertex> <<<numP / blocksize + 1, blocksize>>> (f, m->d_vbo_v, m->d_t, m->d_mesh->he, m->d_mesh->v_to_he, d_visited, m->num_vertices, rad);
}

template<int queueSize, int visitedSizePerVertex>
float triangulation2d<queueSize, visitedSizePerVertex>::mean_num_particle_in_cell() {
	return 0.0;
}