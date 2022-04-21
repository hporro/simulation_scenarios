#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <algorithm>

struct grid_t {
	glm::ivec3 cellPerAxis;
	glm::vec3 cellSize; // gridSize / cellPerAxis;
	int dnbr;
};

struct Grid {
	Grid(grid_t grid_settings);
	~Grid();
	void update(glm::vec3* pos);
	template<typename Functor>
	void apply_f_frnn(Functor f, glm::vec3* pos, const float rad);
	float mean_num_particle_in_cell();

	int* start, * stop, * hash, * partId;
	glm::vec3* pos_sorted;
	thrust::device_ptr<int> dev_start;
	thrust::device_ptr<int> dev_stop;
	thrust::device_ptr<int> dev_hash;
	thrust::device_ptr<int> dev_partId;
	grid_t *h_gridp, *d_gridp;
};

// Compute cells coordinate for a particle with x,y,z position
inline __device__ glm::ivec3 ggetGridCellPos(const glm::vec3 pos, grid_t* dgrid) {
	return glm::floor(pos / dgrid->cellSize);
}

//  Compute cells coordinate for a particle as int
inline __device__ int ggetGridCellHash(glm::ivec3 cell, grid_t* dgrid) {
	cell.x = cell.x & (dgrid->cellPerAxis.x - 1);
	cell.y = cell.y & (dgrid->cellPerAxis.y - 1);
	cell.z = cell.z & (dgrid->cellPerAxis.z - 1);
	return cell.x + (cell.y + cell.z * dgrid->cellPerAxis.y) * dgrid->cellPerAxis.x;
}

__global__ void gcalcHash(int* out_hash, int* out_index, const glm::vec3* __restrict__ pos, grid_t* dgrid) {
	const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < dgrid->dnbr) {
		glm::vec3 p = pos[idx];
		glm::ivec3 index_cell = ggetGridCellPos(p, dgrid);
		int cell_hash = ggetGridCellHash(index_cell, dgrid);

		out_hash[idx] = cell_hash;
		out_index[idx] = idx;
	}
}

__global__ void gfindCellBounds(int* start, int* stop, const int* __restrict__ hash, const int* __restrict__ index, const glm::vec3* __restrict__ pos, glm::vec3* pos_sorted, grid_t* dgrid) {
	const int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dgrid->dnbr)
	{
		atomicMin(&start[hash[idx]], idx);
		atomicMax(&stop[hash[idx]], idx);

		pos_sorted[idx] = pos[index[idx]];
	}

}

Grid::Grid(grid_t grid_settings) {
	h_gridp = new grid_t[1];
	h_gridp[0] = grid_settings;
	const int nbCell = h_gridp->cellPerAxis.x * h_gridp->cellPerAxis.y * h_gridp->cellPerAxis.z;
	cudaMalloc(&this->start, sizeof(int) * nbCell);
	cudaMalloc(&this->stop, sizeof(int) * nbCell);
	cudaMalloc(&this->hash, sizeof(int) * h_gridp->dnbr);
	cudaMalloc(&this->partId, sizeof(int) * h_gridp->dnbr);
	cudaMalloc(&this->pos_sorted, sizeof(glm::vec3) * h_gridp->dnbr);
	cudaMalloc(&this->d_gridp, sizeof(grid_t));
	cudaMemcpy(d_gridp, h_gridp, sizeof(grid_t),cudaMemcpyHostToDevice);
	//*d_gridp = *h_gridp;
	cudaDeviceSynchronize();
	this->dev_start = thrust::device_pointer_cast(this->start);
	this->dev_stop = thrust::device_pointer_cast(this->start);
	this->dev_hash = thrust::device_pointer_cast(this->hash);
}

Grid::~Grid() {
	cudaFree(&this->start);
	cudaFree(&this->stop);
	cudaFree(&this->hash);
	cudaFree(&this->partId);
	cudaFree(&this->pos_sorted);
	cudaFree(&this->d_gridp);
	delete h_gridp;
}

template<typename T>
void swap(T** r, T** s) {
	T* pSwap = *r;
	*r = *s;
	*s = pSwap;
}


void Grid::update(glm::vec3* pos) {
	const int nbCell = h_gridp->cellPerAxis.x * h_gridp->cellPerAxis.y* h_gridp->cellPerAxis.z;

	Timer tim;

	// find particle position in grid
	gcalcHash <<<h_gridp->dnbr /1024+1,1024>>> (hash, partId, pos, d_gridp);
	cudaDeviceSynchronize();
	// sort for reorder by part_id
	dev_partId = thrust::device_pointer_cast(partId);
	dev_hash = thrust::device_pointer_cast(hash);

	cudaDeviceSynchronize();
	thrust::sort_by_key(thrust::device, dev_hash, dev_hash + h_gridp->dnbr, dev_partId);
	cudaDeviceSynchronize();
	//print_int_vector << <h_gridp->dnbr / 1024 + 1, 1024 >> > (h_gridp->dnbr, hash, "hash");

	// reset grid start/stop
	thrust::fill(thrust::device, dev_stop, dev_stop + nbCell, -1);
	cudaDeviceSynchronize();
	thrust::fill(thrust::device, dev_start, dev_start + nbCell, 100000000);
	cudaDeviceSynchronize();
	
	gpuErrchk(cudaGetLastError());

	tim.swap_time();
	gfindCellBounds <<<h_gridp->dnbr / 1024 + 1, 1024>>> (start, stop, hash, partId, pos, pos_sorted, d_gridp);
	cudaDeviceSynchronize();
	LOG_TIMING("findCellBounds: {}", tim.swap_time());
	gpuErrchk(cudaGetLastError());

	/*
	// reorder position
	thrust::device_ptr<glm::vec3> dev_pos = thrust::device_pointer_cast(pos);
	cudaDeviceSynchronize();
	thrust::device_ptr<glm::vec3> dev_pos_sorted = thrust::device_pointer_cast(this->pos_sorted);
	cudaDeviceSynchronize();
	thrust::copy(dev_pos_sorted,dev_pos_sorted+h_gridp->dnbr,dev_pos);
	cudaDeviceSynchronize();
	*/
	// find wich particle in cells
	//swap(&pos, &this->pos_sorted);
}

template<class Functor>
__global__ void apply_f_frnn_kernel(Functor f, const glm::vec3* __restrict__ pos, const int* __restrict__ start, const int* __restrict__ stop, const float rad, grid_t *dgrid, int* __restrict__ ind) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dgrid->dnbr) {
		const glm::ivec3 pl = ggetGridCellPos(pos[i], dgrid);
		const int nbCell = dgrid->cellPerAxis.x * dgrid->cellPerAxis.y * dgrid->cellPerAxis.z;

		for (int a = -1; a <= 1; a++)
			for (int b = -1; b <= 1; b++)
				for (int c = -1; c <= 1; c++) {
					int current = ggetGridCellHash(pl + glm::ivec3(a, b, c), dgrid);
					for (int j = start[current]; j <= stop[current]; j++) {
						if (i == j)continue;
						glm::vec3 dist_vec = pos[i] - pos[j];
						float dist = glm::length(dist_vec);
						if(dist<rad) f(ind[i], ind[j], dist_vec, dist);
					}
				}

	}
}

template<class Functor>
void Grid::apply_f_frnn(Functor f, glm::vec3* pos, float rad) {
	LOG_EVENT("applying frnn");
	apply_f_frnn_kernel<Functor> <<<(h_gridp->dnbr / 1024) + 1, 1024 >>> (f, pos_sorted, start, stop, rad, d_gridp, partId);
	gpuErrchk(cudaGetLastError());
	LOG_EVENT("Frnn applied");
}


__global__ void num_cells_kernel(int nbCell, const int* __restrict__ start, const int* __restrict__ stop, int* num_cells) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nbCell) {
		if(stop[i]>start[i])
		num_cells[i] = stop[i] - start[i];
	}
}


float Grid::mean_num_particle_in_cell() {
	const int nbCell = h_gridp->cellPerAxis.x * h_gridp->cellPerAxis.y * h_gridp->cellPerAxis.z;
	int *d_mean;
	cudaMalloc((void**)&d_mean, nbCell*sizeof(int));
	cudaMemset(d_mean, 0, nbCell*sizeof(int));

	num_cells_kernel <<<nbCell / 32 + 1, 32 >>> (nbCell, start, stop, d_mean);
	thrust::device_ptr<int> dev_mean = thrust::device_pointer_cast(d_mean);
	thrust::inclusive_scan(dev_mean, dev_mean + nbCell, dev_mean);
	cudaDeviceSynchronize();

	int* h_sum = new int[nbCell];
	cudaMemcpy(h_sum, d_mean, nbCell*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_mean);

	float ret = (float)h_sum[nbCell - 1] / (float)nbCell;
	delete[] h_sum;
	return ret;
}