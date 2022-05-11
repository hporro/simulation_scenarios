#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>

struct GridCount_data {
	glm::vec3 min;
	glm::vec3 cell_size;
	glm::ivec3 num_cells;
	int tot_num_cells;
};

struct GridCount {
	GridCount(int numP, glm::vec3 min, glm::vec3 cell_size, glm::ivec3 num_cells);
	~GridCount();
	void update(glm::vec3* pos, glm::vec3* vel);
	template<class Functor>
	void apply_f_frnn(Functor f, glm::vec3* pos, const float rad);
	//float mean_num_particle_in_cell();
	
	// data
	int numP;
	int blocksize = 32;
	GridCount_data *d_gcdata, *h_gcdata;
	glm::vec3 *pos_sorted, *vel_sorted;
	int *d_hash;
	int *d_count_keys, *d_cumulative_count_keys;
	int *d_ind;

	// private function
	void sort_hashed(glm::vec3* pos, glm::vec3* vel);
};

GridCount::GridCount(int numP, glm::vec3 min, glm::vec3 cell_size, glm::ivec3 num_cells) : numP(numP) {
	h_gcdata = new GridCount_data[1];
	h_gcdata[0].min = min;
	h_gcdata[0].cell_size = cell_size;
	h_gcdata[0].num_cells = num_cells;
	h_gcdata[0].tot_num_cells = num_cells.x*num_cells.y*num_cells.z;
	cudaMalloc(&this->d_gcdata, sizeof(GridCount_data));
	cudaMemcpy(d_gcdata, &h_gcdata[0], sizeof(GridCount_data), cudaMemcpyHostToDevice);

	cudaMalloc(&this->pos_sorted,              sizeof(glm::vec3)*numP);
	cudaMalloc(&this->vel_sorted,              sizeof(glm::vec3)*numP);
	cudaMalloc(&this->d_hash,                  sizeof(int)*numP);
	cudaMalloc(&this->d_count_keys,            sizeof(int)*h_gcdata->tot_num_cells+1);
	cudaMalloc(&this->d_cumulative_count_keys, sizeof(int)*h_gcdata->tot_num_cells+1);
	cudaMalloc(&this->d_ind,                   sizeof(int)*numP);
	gpuErrchk(cudaGetLastError());
}

GridCount::~GridCount() {
	gpuErrchk(cudaGetLastError());
	delete[] h_gcdata;
	cudaFree(d_gcdata);
	cudaFree(pos_sorted);
	cudaFree(vel_sorted);
	cudaFree(d_hash);
	cudaFree(d_cumulative_count_keys);
	cudaFree(d_count_keys);
	cudaFree(d_ind);
	gpuErrchk(cudaGetLastError());
}

__device__ int calcHash(glm::vec3 p, GridCount_data* gcdata) {
	p -= gcdata->min;
	int num_x = gcdata->num_cells.x;
	int num_y = gcdata->num_cells.y;
	int num_z = gcdata->num_cells.z;
	glm::ivec3 q = glm::ivec3(floor(p.x / num_x), floor(p.y / num_y), floor(p.z / num_z));
	return (q.x+num_x)%num_x + ((q.y+num_y)%num_y + (q.z+num_z)%num_z * num_y) * num_x;
}

__global__ void calc_hash_kernel(int numP, glm::vec3* pos, int* hash, GridCount_data* gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) hash[i] = calcHash(pos[i], gcdata); 
}

void GridCount::update(glm::vec3* pos, glm::vec3* vel) {
	calc_hash_kernel<<<numP/blocksize + 1, blocksize>>>(numP, pos, d_hash, d_gcdata);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	sort_hashed(pos,vel);
	gpuErrchk(cudaGetLastError());

}

__global__ void fill_count_array_kernel(int numP, int* count_keys, int* hash, GridCount_data* gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) atomicAdd(&count_keys[hash[i]], 1);
}

__global__ void compute_new_indices_kernel(int numP, int* ind, int* hash, int * cumulative_count_keys, GridCount_data* gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if(i<numP){
		//printf("i: %d ind[i]: %d cumulative_count_keys[hash[i]]: %d hash[i]: %d\n", i, ind[i], cumulative_count_keys[hash[i]], hash[i]);
		ind[i] = atomicAdd(&cumulative_count_keys[hash[i]],-1) - 1;
		//printf("i: %d ind[i]: %d cumulative_count_keys[hash[i]]: %d hash[i]: %d\n", i, ind[i], cumulative_count_keys[hash[i]], hash[i]);
	}
}

__global__ void reorder_arrays_kernel(int numP, glm::vec3* pos, glm::vec3* vel, glm::vec3* sorted_pos, glm::vec3* sorted_vel, int *ind) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	if (i < numP) {
		sorted_pos[ind[i]] = pos[i];
		sorted_vel[ind[i]] = vel[i];
	}
}

void GridCount::sort_hashed(glm::vec3* pos, glm::vec3* vel) {
	cudaMemset(d_count_keys, 0, h_gcdata->tot_num_cells * sizeof(int));
	cudaMemset(d_cumulative_count_keys, 0, h_gcdata->tot_num_cells * sizeof(int));
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();

	fill_count_array_kernel<<<numP/blocksize+1,blocksize>>>(numP, d_count_keys, d_hash, d_gcdata);
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();

	thrust::device_ptr<int> acum_array = thrust::device_pointer_cast(d_cumulative_count_keys);
	thrust::device_ptr<int> count_array = thrust::device_pointer_cast(d_count_keys);
	thrust::inclusive_scan(count_array, count_array + h_gcdata->tot_num_cells, acum_array);
	gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();

	compute_new_indices_kernel<<<numP/blocksize + 1, blocksize>>>(numP, d_ind, d_hash, d_cumulative_count_keys, d_gcdata);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	reorder_arrays_kernel<<<numP/blocksize + 1, blocksize>>>(numP,pos,vel,pos_sorted,vel_sorted,d_ind);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	cudaMemcpy(pos, pos_sorted, numP * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(vel, vel_sorted, numP * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

template<class Functor>
__global__ void apply_f_frnn_gc_kernel(Functor f, int numP, glm::vec3* __restrict__ pos, int* __restrict__ count_keys, int* __restrict__ cumulative_count_keys, float rad2, GridCount_data* gcdata) {
	const int i = threadIdx.x + (blockDim.x * blockIdx.x);
	int num_x = gcdata->num_cells.x;
	int num_y = gcdata->num_cells.y;
	int num_z = gcdata->num_cells.z;
	if (i < numP) {
		glm::vec3 pos_i = pos[i];
		int hi = calcHash(pos_i, gcdata);
		for (int ddx = -1; ddx <= 1; ddx++) {
			for (int ddy = -1; ddy <= 1; ddy++) {
				for (int ddz = -1; ddz <= 1; ddz++) {
					int h = hi + ddx + (ddz * num_y + ddy) * num_x;
					h += (h > gcdata->tot_num_cells ? -gcdata->tot_num_cells : 0) + (h < 0 ? gcdata->tot_num_cells : 0); // border case
					//if (h > gcdata->tot_num_cells || h < 0)printf("i: %d h: %d tot: %d\n", i, h, gcdata->tot_num_cells);
					for (int j = cumulative_count_keys[h]; j < cumulative_count_keys[h] + count_keys[h]; j++) {
						//printf("i: %d j: %d\n", i, j);
						glm::vec3 sub_vector = pos[j] - pos_i;
						float r2 = glm::dot(sub_vector, sub_vector);
						if (r2 < rad2) {
							f(i, j, sub_vector, sqrt(rad2));
						}
					}
				}
			}
		}
	}
}

template<class Functor>
void GridCount::apply_f_frnn(Functor f, glm::vec3* pos, const float rad) {
	apply_f_frnn_gc_kernel<Functor><<<numP/blocksize+ 1, blocksize >>>(f, numP, pos, d_count_keys, d_cumulative_count_keys, rad*rad, d_gcdata);
	gpuErrchk(cudaGetLastError());
}