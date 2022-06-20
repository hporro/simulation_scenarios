#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>

#include "../math/batea_math.cuh"
#include "../logging/Logging.h"
#include "../gpu/gpuErrCheck.h"

#include <algorithm>
#include <iostream>

struct GridCount2d_data {
	glm::vec2 min;
	glm::vec2 cell_size;
	glm::ivec2 num_cells;
	int tot_num_cells;
};

struct GridCount2d {
	GridCount2d(int numP, glm::vec2 min, glm::vec2 cell_size, glm::ivec2 num_cells);
	~GridCount2d();
	void update(glm::vec2* pos, glm::vec2* vel);
	void update(glm::vec2* pos);
	template<class Functor>
	void apply_f_frnn(Functor& f, glm::vec2* pos, const float rad);
	float mean_num_particle_in_cell();

	// data
	int numP;
	int blocksize = 64;
	GridCount2d_data* d_gcdata, * h_gcdata;
	glm::vec2* pos_sorted, * vel_sorted;
	int* d_hash;
	int* d_count_keys, * d_cumulative_count_keys;
	int* d_ind;

	int* d_has_particles;
	int* d_num_particles;

	// private function
	void sort_hashed(glm::vec2* pos, glm::vec2* vel);
	void sort_hashed(glm::vec2* pos);
};

GridCount2d::GridCount2d(int numP, glm::vec2 min, glm::vec2 cell_size, glm::ivec2 num_cells) : numP(numP) {
	h_gcdata = new GridCount2d_data[1];
	h_gcdata[0].min = min;
	h_gcdata[0].cell_size = cell_size;
	h_gcdata[0].num_cells = num_cells;
	h_gcdata[0].tot_num_cells = num_cells.x * num_cells.y;
	cudaMalloc(&this->d_gcdata, sizeof(GridCount2d_data));
	cudaMemcpy(d_gcdata, &h_gcdata[0], sizeof(GridCount2d_data), cudaMemcpyHostToDevice);

	cudaMalloc(&this->pos_sorted, sizeof(glm::vec2) * numP);
	cudaMalloc(&this->vel_sorted, sizeof(glm::vec2) * numP);
	cudaMalloc(&this->d_hash, sizeof(int) * numP);
	cudaMalloc(&this->d_count_keys, sizeof(int) * (h_gcdata->tot_num_cells + 1));
	cudaMalloc(&this->d_cumulative_count_keys, sizeof(int) * (h_gcdata->tot_num_cells + 1));
	cudaMalloc(&this->d_ind, sizeof(int) * numP);

	cudaMalloc(&this->d_has_particles, sizeof(int) * (h_gcdata->tot_num_cells + 1));
	cudaMalloc(&this->d_num_particles, sizeof(int) * (h_gcdata->tot_num_cells + 1));

	gpuErrchk(cudaGetLastError());
}

GridCount2d::~GridCount2d() {
	gpuErrchk(cudaGetLastError());
	delete[] h_gcdata;
	cudaFree(d_gcdata);
	cudaFree(pos_sorted);
	cudaFree(vel_sorted);
	cudaFree(d_hash);
	cudaFree(d_cumulative_count_keys);
	cudaFree(d_count_keys);
	cudaFree(d_ind);
	cudaFree(d_has_particles);
	cudaFree(d_num_particles);
	gpuErrchk(cudaGetLastError());
}

__device__ unsigned int calcHash(unsigned int x, unsigned int y, GridCount2d_data* __restrict__ gcdata) {
	return (y * gcdata->num_cells.x + x) % gcdata->tot_num_cells;
}

__device__ glm::ivec2 calc_hashxy(glm::vec2 p, GridCount2d_data* __restrict__ gcdata) {
	p -= gcdata->min;
	int x = (p.x / gcdata->cell_size.x);
	int y = (p.y / gcdata->cell_size.y);
	x %= gcdata->num_cells.x;
	y %= gcdata->num_cells.y;
	return glm::ivec2(x, y);
}

__device__ unsigned int calcHash(glm::vec2 p, GridCount2d_data* __restrict__ gcdata) {
	glm::ivec2 x = calc_hashxy(p,gcdata);
	return calcHash(x.x,x.y,gcdata);
}

__global__ void calc_hash_kernel(int numP, glm::vec2* __restrict__ pos, int* hash, GridCount2d_data* __restrict__ gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) {
		hash[i] = calcHash(pos[i], gcdata);
	}
}

void GridCount2d::update(glm::vec2* pos, glm::vec2* vel) {
	calc_hash_kernel << <numP / blocksize + 1, blocksize >> > (numP, pos, d_hash, d_gcdata);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	sort_hashed(pos, vel);
	gpuErrchk(cudaGetLastError());
}

void GridCount2d::update(glm::vec2* pos) {
	calc_hash_kernel << <numP / blocksize + 1, blocksize >> > (numP, pos, d_hash, d_gcdata);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	sort_hashed(pos);
	gpuErrchk(cudaGetLastError());
}

__global__ void fill_count_array_kernel(int numP, int* count_keys, int* __restrict__ hash, GridCount2d_data* __restrict__ gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) {
		atomicAdd(&count_keys[hash[i]], 1);
	}
}

__global__ void compute_new_indices_kernel(int numP, int* ind, int* __restrict__ hash, int* cumulative_count_keys, GridCount2d_data* __restrict__ gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) {
		ind[i] = atomicAdd(&cumulative_count_keys[hash[i]], -1) - 1;
	}
}

__global__ void reorder_arrays_kernel(int numP, glm::vec2* __restrict__ pos, glm::vec2* __restrict__ vel, glm::vec2* sorted_pos, glm::vec2* sorted_vel, int* __restrict__ ind) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) {
		sorted_pos[ind[i]] = pos[i];
		sorted_vel[ind[i]] = vel[i];
	}
}

__global__ void reorder_arrays_kernel_s_vel(int numP, glm::vec2* __restrict__ pos, glm::vec2* sorted_pos, int* __restrict__ ind) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) {
		sorted_pos[ind[i]] = pos[i];
	}
}

void GridCount2d::sort_hashed(glm::vec2* pos, glm::vec2* vel) {
	Timer grid_timer;

	cudaMemset(d_count_keys, 0, h_gcdata->tot_num_cells * sizeof(int));
	cudaMemset(d_cumulative_count_keys, 0, h_gcdata->tot_num_cells * sizeof(int));
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();
	LOG_TIMING("Initialize buffers in sort hashed: {}", grid_timer.swap_time());

	fill_count_array_kernel << <numP / blocksize + 1, blocksize >> > (numP, d_count_keys, d_hash, d_gcdata);
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();

	LOG_TIMING("fill_count_array_kernel: {}", grid_timer.swap_time());

	thrust::device_ptr<int> acum_array = thrust::device_pointer_cast(d_cumulative_count_keys);
	thrust::device_ptr<int> count_array = thrust::device_pointer_cast(d_count_keys);
	thrust::inclusive_scan(count_array, count_array + h_gcdata->tot_num_cells, acum_array);
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();
	LOG_TIMING("inclusive_scan: {}", grid_timer.swap_time());

	compute_new_indices_kernel << <numP / blocksize + 1, blocksize >> > (numP, d_ind, d_hash, d_cumulative_count_keys, d_gcdata);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("compute_new_indices_kernel: {}", grid_timer.swap_time());

	reorder_arrays_kernel << <numP / blocksize + 1, blocksize >> > (numP, pos, vel, pos_sorted, vel_sorted, d_ind);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("reorder_arrays_kernel: {}", grid_timer.swap_time());

	cudaMemcpy(pos, pos_sorted, numP * sizeof(glm::vec2), cudaMemcpyDeviceToDevice);
	cudaMemcpy(vel, vel_sorted, numP * sizeof(glm::vec2), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("reorder arrays: {}", grid_timer.swap_time());
}

template<class T>
__global__ void init_arr(int n, T* arr, T val) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < n) arr[i] = val;
}

void GridCount2d::sort_hashed(glm::vec2* pos) {
	Timer grid_timer;

	cudaMemset(d_count_keys, 0, (h_gcdata->tot_num_cells + 1) * sizeof(int));
	cudaMemset(d_cumulative_count_keys, 0, (h_gcdata->tot_num_cells + 1) * sizeof(int));
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();
	LOG_TIMING("Initialize buffers in sort hashed: {}", grid_timer.swap_time());

	fill_count_array_kernel << <numP / blocksize + 1, blocksize >> > (numP, d_count_keys, d_hash, d_gcdata);
	cudaDeviceSynchronize();

	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();
	LOG_TIMING("fill_count_array_kernel: {}", grid_timer.swap_time());

	thrust::device_ptr<int> acum_array = thrust::device_pointer_cast(d_cumulative_count_keys);
	thrust::device_ptr<int> count_array = thrust::device_pointer_cast(d_count_keys);
	thrust::inclusive_scan(count_array, count_array + h_gcdata->tot_num_cells, acum_array);
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();
	LOG_TIMING("inclusive_scan: {}", grid_timer.swap_time());

	compute_new_indices_kernel << <numP / blocksize + 1, blocksize >> > (numP, d_ind, d_hash, d_cumulative_count_keys, d_gcdata);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("compute_new_indices_kernel: {}", grid_timer.swap_time());

	reorder_arrays_kernel_s_vel << <numP / blocksize + 1, blocksize >> > (numP, pos, pos_sorted, d_ind);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("reorder_arrays_kernel: {}", grid_timer.swap_time());

	cudaMemcpy(pos, pos_sorted, numP * sizeof(glm::vec2), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("reorder arrays: {}", grid_timer.swap_time());
}

template<class Functor>
__global__ void apply_f_frnn_gc_kernel(Functor f, int numP, glm::vec2* __restrict__ pos, int* __restrict__ count_keys, int* __restrict__ cumulative_count_keys, float rad2, GridCount2d_data* gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) {
		int const num_x = gcdata->num_cells.x;
		int const num_y = gcdata->num_cells.y;
		const glm::vec2 pos_i = pos[i];
		glm::ivec2 hi = calc_hashxy(pos_i, gcdata);

		for (int ddx = -1; ddx <= 1; ddx++) {
			for (int ddy = -1; ddy <= 1; ddy++) {
				glm::ivec2 hh = hi + glm::ivec2(ddx, ddy);
				if (hh.x > gcdata->num_cells.x || hh.x < 0)continue;
				if (hh.y > gcdata->num_cells.y || hh.y < 0)continue;
				int h = calcHash(hh.x, hh.y, gcdata);

				for (int j = cumulative_count_keys[h]; j < cumulative_count_keys[h] + count_keys[h]; j++) {
					const glm::vec2 sub_vector = pos[j] - pos_i;

					float r2 = glm::dot(sub_vector, sub_vector);
					if (r2 <= rad2) f(i, j, sub_vector, sqrt(r2));
				}
			}
		}
	}
}

template<class Functor>
void GridCount2d::apply_f_frnn(Functor& f, glm::vec2* pos, const float rad) {
	apply_f_frnn_gc_kernel<Functor> << <numP / blocksize + 1, blocksize >> > (f, numP, pos, d_count_keys, d_cumulative_count_keys, rad * rad, d_gcdata);
	gpuErrchk(cudaGetLastError());
}

__global__ void mean_num_particle_in_cell_kernel(int* __restrict__ count_keys, int* has_particles, int* num_particles, GridCount2d_data* __restrict__ gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < gcdata->tot_num_cells) {
		has_particles[i] = (count_keys[i] > 0);
		num_particles[i] = count_keys[i];
	}
}

float GridCount2d::mean_num_particle_in_cell() {
	cudaMemset(d_has_particles, 0, h_gcdata->tot_num_cells * sizeof(int));
	cudaMemset(d_num_particles, 0, h_gcdata->tot_num_cells * sizeof(int));
	cudaDeviceSynchronize();
	mean_num_particle_in_cell_kernel << <numP / blocksize + 1, blocksize >> > (d_count_keys, d_has_particles, d_num_particles, d_gcdata);
	cudaDeviceSynchronize();
	thrust::device_ptr<int> has_particles_array = thrust::device_pointer_cast(d_has_particles);
	thrust::device_ptr<int> num_particles_array = thrust::device_pointer_cast(d_num_particles);
	thrust::device_ptr<int> max_val = thrust::max_element(num_particles_array, num_particles_array + h_gcdata->tot_num_cells);
	thrust::inclusive_scan(has_particles_array, has_particles_array + h_gcdata->tot_num_cells, has_particles_array);
	thrust::inclusive_scan(num_particles_array, num_particles_array + h_gcdata->tot_num_cells, num_particles_array);
	cudaDeviceSynchronize();
	int* res = new int[2];
	cudaMemcpy(res, &d_has_particles[h_gcdata->tot_num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(res + 1, &d_num_particles[h_gcdata->tot_num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	float answer = (float)res[1] / (float)res[0];
	delete[] res;
	return answer;
}