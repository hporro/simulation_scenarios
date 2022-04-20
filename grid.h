#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

struct grid_t {
	glm::ivec3 cellPerAxis;
	glm::vec3 cellSize; // gridSize / cellPerAxis;
	int dnbr;
};

class Grid {
public:
	Grid(grid_t grid_settings);
	~Grid();
	void computeGrid(glm::vec3* pos, glm::vec3 offset);
	template<typename Functor>
	void apply_f_frnn(Functor f, glm::vec3* pos, const float rad);

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
	//cell.x = cell.x & (dgrid->cellPerAxis.x - 1);
	//cell.y = cell.y & (dgrid->cellPerAxis.y - 1);
	//cell.z = cell.z & (dgrid->cellPerAxis.z - 1);
	return cell.x + cell.y * dgrid->cellPerAxis.x + cell.z * dgrid->cellPerAxis.x * dgrid->cellPerAxis.y;
}

__global__ void gcalcHash(int* out_hash, int* out_index, const glm::vec3* __restrict__ pos, grid_t* dgrid, glm::vec3 offset) {
	const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < dgrid->dnbr) {
		glm::vec3 p = pos[idx] + offset;
		glm::ivec3 index_cell = ggetGridCellPos(p, dgrid);
		int cell_hash = ggetGridCellHash(index_cell, dgrid);

		out_hash[idx] = cell_hash;
		out_index[idx] = idx;
	}
}

__global__ void gfindCellBounds(int* start, int* stop, const int* __restrict__ hash, const int* __restrict__ index, const glm::vec3* __restrict__ pos, glm::vec3* pos_sorted, grid_t* dgrid, glm::vec3 offset) {
	int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dgrid->dnbr)
	{
		atomicMin(&start[hash[idx]], idx);
		atomicMax(&stop[hash[idx]], idx);

		pos_sorted[idx] = pos[index[idx]] + offset;
	}

}

__global__ void print_int_vector(int n, int* vec, char* msg) {
	const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < n) {
		printf("i: %d, vec: %d\n", idx, vec[idx]);
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


void Grid::computeGrid(glm::vec3* pos, glm::vec3 offset) {
	const int nbCell = h_gridp->cellPerAxis.x * h_gridp->cellPerAxis.y* h_gridp->cellPerAxis.z;
	// find particle position in grid
	gcalcHash <<<h_gridp->dnbr /1024+1,1024>>> (hash, partId, pos, d_gridp, offset);
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
	//print_int_vector << <1, nbCell >> > (nbCell, start, "start");
	//print_int_vector << <1, nbCell >> > (nbCell, stop, "stop");
	
	gpuErrchk(cudaGetLastError());

	gfindCellBounds <<<h_gridp->dnbr / 1024 + 1, 1024>>> (start, stop, hash, partId, pos, pos_sorted, d_gridp, offset);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	//print_int_vector << <h_gridp->dnbr / 1024 + 1, 1024 >>> (h_gridp->dnbr, hash,"hash");
	//print_int_vector << <1, nbCell >> > (nbCell, start, "start");
	//print_int_vector << <1, nbCell >> > (nbCell, stop, "stop");


	/*
	// reorder position
	thrust::device_ptr<glm::vec3> dev_pos = thrust::device_pointer_cast(pos);
	cudaDeviceSynchronize();
	thrust::device_ptr<glm::vec3> dev_pos_sorted = thrust::device_pointer_cast(this->pos_sorted);
	cudaDeviceSynchronize();
	thrust::copy(dev_pos_sorted,dev_pos_sorted+this->nbr,dev_pos);
	cudaDeviceSynchronize();
	// find wich particle in cells
	*/
	//swap(&pos, &this->pos_sorted);
}

template<class Functor>
__global__ void apply_f_frnn_kernel(Functor f, const glm::vec3* __restrict__ pos, const int* __restrict__ start, const int* __restrict__ stop, const float rad, grid_t *dgrid, int* __restrict__ ind) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < dgrid->dnbr) {
		int count = 0;
		const int nbCell = dgrid->cellPerAxis.x * dgrid->cellPerAxis.y * dgrid->cellPerAxis.z;
		const glm::ivec3 pl = ggetGridCellPos(pos[idx], dgrid);

		for (int a = -1; a <= 1; a++) {
			for (int b = -1; b <= 1; b++) {
				for (int c = -1; c <= 1; c++) {
					glm::ivec3 neighboring_cell = pl + glm::ivec3(a, b, c);
					int current = ggetGridCellHash(neighboring_cell, dgrid);
					//printf("i: %d Current: %d actual: %d\n", idx, current, ggetGridCellHash(pl, dgrid));
					if (current < 0) continue;
					if (current >= nbCell) continue;
					//printf("i: %d current: %d\n", idx, current);
					//printf("i: %d start: %d\n", idx, start[current]);
					//printf("i: %d end: %d\n", idx, stop[current]);
					for (int i = start[current]; i <= stop[current]; i++) {
						if (i == idx) continue;
						glm::dvec3 tmp = pos[idx] - pos[i];
						const float sqrDist = glm::dot(tmp, tmp);
						if (sqrDist < rad * rad) {
							f(ind[idx], ind[i], tmp, sqrDist);
						}
					}
				}
			}
		}
	}
}

template<class Functor>
void Grid::apply_f_frnn(Functor f, glm::vec3* pos, float rad) {
	LOG_EVENT("applying frnn");
	apply_f_frnn_kernel<Functor> <<<h_gridp->dnbr / 124 + 1, 124 >>> (f, pos_sorted, start, stop, rad, d_gridp, partId);
	gpuErrchk(cudaGetLastError());
	LOG_EVENT("Frnn applied");
}