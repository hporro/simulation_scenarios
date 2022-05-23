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
#include <cooperative_groups.h>

#include <algorithm>

struct grid_t {
	int numP;
	glm::vec3 min;
	glm::vec3 cell_size;
	glm::ivec3 num_cells;
	int tot_num_cells;
};

struct Grid {
	Grid(int numP, glm::vec3 min, glm::vec3 cell_size, glm::ivec3 num_cells);
	~Grid();
	void update(glm::vec3* pos, glm::vec3* vel);
	void update(glm::vec3* pos);
	template<typename Functor>
	void apply_f_frnn(Functor f, glm::vec3* pos, const float rad);
	float mean_num_particle_in_cell();

	int* start, * stop, * hash, * partId;
	glm::vec3* pos_sorted, *vel_sorted;
	thrust::device_ptr<int> dev_start;
	thrust::device_ptr<int> dev_stop;
	thrust::device_ptr<int> dev_hash;
	thrust::device_ptr<int> dev_partId;
	grid_t *h_gridp, *d_gridp;
};

__device__ int calcHash(glm::vec3 p, grid_t* __restrict__ gcdata) {
	p -= gcdata->min;
	//printf("p: %f %f %f\n", p.x, p.y, p.z);
	int x = p.x / gcdata->cell_size.x;
	int y = p.y / gcdata->cell_size.y;
	int z = p.z / gcdata->cell_size.z;
	//printf("p: %f %f %f\n", p.x, p.y, p.z);
	return (z * gcdata->num_cells.y + y) * gcdata->num_cells.x + x;
}

__global__ void gcalcHash(int* out_hash, int* out_index, const glm::vec3* __restrict__ pos, grid_t* dgrid) {
	const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < dgrid->numP) {
		glm::vec3 p = pos[idx];
		
		int cell_hash = calcHash(p, dgrid);

		out_hash[idx] = cell_hash;
		out_index[idx] = idx;
	}
}

template<class T>
void print_d_vec(int n, T* d_vec) {
	T* h_vec = new T[n];
	cudaMemcpy(h_vec, d_vec, n * sizeof(T), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (int i = 0; i < n; i++) {
		printf("%d ", h_vec[i]);
	}
	std::cout << std::endl;

	delete[] h_vec;
}

__global__ void gfindCellBounds(int* start, int* stop, const int* __restrict__ hash_arr, const int* __restrict__ ind, const glm::vec3* __restrict__ pos, const glm::vec3* __restrict__ vel, glm::vec3* pos_sorted, glm::vec3* vel_sorted, grid_t* dgrid) {
	const int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dgrid->numP)
	{
		atomicMin(&start[hash_arr[idx]], idx);
		atomicMax(&stop[hash_arr[idx]], idx);

		pos_sorted[idx] = pos[ind[idx]];
		vel_sorted[idx] = vel[ind[idx]];
	}
}

__global__ void gfindCellBounds_s_vel(int* start, int* stop, const int* __restrict__ hash_arr, const int* __restrict__ ind, const glm::vec3* __restrict__ pos, glm::vec3* pos_sorted, grid_t* dgrid) {
	const int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dgrid->numP)
	{
		atomicMin(&start[hash_arr[idx]], idx);
		atomicMax(&stop[hash_arr[idx]], idx);

		pos_sorted[idx] = pos[ind[idx]];
	}
}

Grid::Grid(int numP, glm::vec3 min, glm::vec3 cell_size, glm::ivec3 num_cells) {
	gpuErrchk(cudaGetLastError());
	h_gridp = new grid_t[1];
	h_gridp[0].min = min;
	h_gridp[0].cell_size = cell_size;
	h_gridp[0].num_cells = num_cells;
	h_gridp[0].tot_num_cells = num_cells.x * num_cells.y * num_cells.z;
	h_gridp[0].numP = numP;

	cudaMalloc(&this->d_gridp, sizeof(grid_t));
	gpuErrchk(cudaGetLastError());
	cudaMemcpy(d_gridp, &h_gridp[0], sizeof(grid_t), cudaMemcpyHostToDevice);
	gpuErrchk(cudaGetLastError());

	cudaMalloc(&this->start, sizeof(int) * h_gridp->tot_num_cells);
	cudaMalloc(&this->stop, sizeof(int) * h_gridp->tot_num_cells);
	cudaMalloc(&this->hash, sizeof(int) * h_gridp->numP);
	cudaMalloc(&this->partId, sizeof(int) * h_gridp->numP);
	cudaMalloc(&this->pos_sorted, sizeof(glm::vec3) * numP);
	cudaMalloc(&this->vel_sorted, sizeof(glm::vec3) * numP);
	gpuErrchk(cudaGetLastError());
	//*d_gridp = *h_gridp;
	cudaDeviceSynchronize();
	this->dev_start = thrust::device_pointer_cast(this->start);
	this->dev_stop = thrust::device_pointer_cast(this->stop);
	this->dev_hash = thrust::device_pointer_cast(this->hash);
	gpuErrchk(cudaGetLastError());
}

Grid::~Grid() {
	cudaFree(this->start);
	cudaFree(this->stop);
	cudaFree(this->hash);
	cudaFree(this->partId);
	cudaFree(this->pos_sorted);
	cudaFree(this->vel_sorted);
	cudaFree(this->d_gridp);
	delete h_gridp;
	gpuErrchk(cudaGetLastError());
}

template<typename T>
void swap(T** r, T** s) {
	T* pSwap = *r;
	*r = *s;
	*s = pSwap;
}

void Grid::update(glm::vec3* pos, glm::vec3* vel) {
	const int nbCell = h_gridp->tot_num_cells;

	Timer tim;

	// find particle position in grid
	gcalcHash <<<h_gridp->numP/1024+1,1024>>> (hash, partId, pos, d_gridp);
	cudaDeviceSynchronize();
	LOG_TIMING("Calc hash: {}", tim.swap_time());
	// sort for reorder by part_id
	dev_partId = thrust::device_pointer_cast(partId);
	dev_hash = thrust::device_pointer_cast(hash);
	cudaDeviceSynchronize();

	tim.swap_time();
	thrust::sort_by_key(thrust::device, dev_hash, dev_hash + h_gridp->numP, dev_partId);
	cudaDeviceSynchronize();
	LOG_TIMING("Sort by key: {}", tim.swap_time());
	//print_int_vector << <h_gridp->dnbr / 1024 + 1, 1024 >> > (h_gridp->dnbr, hash, "hash");

	// reset grid start/stop
	thrust::fill(thrust::device, dev_stop, dev_stop + nbCell, -1);
	cudaDeviceSynchronize();
	thrust::fill(thrust::device, dev_start, dev_start + nbCell, 100000000);
	cudaDeviceSynchronize();
	
	gpuErrchk(cudaGetLastError());

	tim.swap_time();
	gfindCellBounds <<<h_gridp->numP / 1024 + 1, 1024>>> (start, stop, hash, partId, pos, vel, pos_sorted, vel_sorted, d_gridp);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
	LOG_TIMING("findCellBounds: {}", tim.swap_time());

	cudaMemcpy(pos, pos_sorted, h_gridp->numP * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(vel, vel_sorted, h_gridp->numP * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	LOG_TIMING("Copy arrays: {}", tim.swap_time());

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

void Grid::update(glm::vec3* pos) {
	const int nbCell = h_gridp->tot_num_cells;

	Timer tim;

	// find particle position in grid
	gcalcHash <<<h_gridp->numP /1024+1,1024>>> (hash, partId, pos, d_gridp);
	cudaDeviceSynchronize();
	LOG_TIMING("Calc hash: {}", tim.swap_time());
	//printf("Hash:");
	//print_d_vec(h_gridp->numP, hash);

	// sort for reorder by part_id
	dev_partId = thrust::device_pointer_cast(partId);
	dev_hash = thrust::device_pointer_cast(hash);

	cudaDeviceSynchronize();
	thrust::sort_by_key(thrust::device, dev_hash, dev_hash + h_gridp->numP, dev_partId);
	cudaDeviceSynchronize();
	LOG_TIMING("Sort by key: {}", tim.swap_time());
	//printf("sorted by key:");
	//print_d_vec(h_gridp->numP, partId);
	//print_d_vec(h_gridp->numP, hash);
	
	// reset grid start/stop
	thrust::fill(thrust::device, dev_stop, dev_stop + nbCell, -1);
	cudaDeviceSynchronize();
	thrust::fill(thrust::device, dev_start, dev_start + nbCell, 100000000);
	cudaDeviceSynchronize();
	
	gpuErrchk(cudaGetLastError());

	tim.swap_time();
	gfindCellBounds_s_vel <<<h_gridp->numP / 1024 + 1, 1024>>> (start, stop, hash, partId, pos, pos_sorted, d_gridp);
	cudaDeviceSynchronize();
	//printf("Find bounds:");
	//print_d_vec(h_gridp->tot_num_cells, start);
	//print_d_vec(h_gridp->tot_num_cells, stop);
	LOG_TIMING("findCellBounds: {}", tim.swap_time());
	gpuErrchk(cudaGetLastError());

	cudaMemcpy(pos, pos_sorted, h_gridp->numP * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	LOG_TIMING("Copy arrays: {}", tim.swap_time());

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
__global__ void apply_f_frnn_kernel(Functor f, const glm::vec3* __restrict__ pos, const int* __restrict__ start, const int* __restrict__ stop, const double rad2, grid_t *dgrid, int* __restrict__ ind) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dgrid->numP) {
		int const num_x = dgrid->num_cells.x;
		int const num_y = dgrid->num_cells.y;
		int const num_z = dgrid->num_cells.z;

		const glm::vec3 pos_i = pos[i];
		const int hi = calcHash(pos_i, dgrid);

		for (int a = -1; a <= 1; a++)
			for (int b = -1; b <= 1; b++)
				for (int c = -1; c <= 1; c++) {
					int current = hi + a + (c * num_y + b) * num_x;
					//current += (current > dgrid->tot_num_cells ? -dgrid->tot_num_cells : 0) + (current < 0 ? dgrid->tot_num_cells : 0); // border case. The particles in the border also check for neighbors in the oposite borders
					//if(i==0)printf("i: %d current: %d start[current]: %d stop[current]: %d\n", i, current, start[current], stop[current]);
					if (current > dgrid->tot_num_cells || current < 0)continue;
					for (int j = start[current]; j <= stop[current]; j++) {
						//printf("i: %d current: %d j: %d start[current]: %d\n", i, current, j , start[current]);
						const glm::vec3 sub_vector = pos[j] - pos_i;
						double r2 = glm::dot(sub_vector, sub_vector);
						//printf("FRNN hi: %d h: %d i: %d j: %d r2: %f r: %f pos[i]: %f %f %f pos[j]: %f %f %f dist_vec: %f %f %f\n", hi, h, i, j, r2, sqrt(r2), pos_i.x, pos_i.y, pos_i.z, pos[j].x, pos[j].y, pos[j].z, sub_vector.x, sub_vector.y, sub_vector.z);
						if (r2 <= rad2) f(i, j, sub_vector, sqrt(r2));
					}
				}
	}
}

template<class Functor>
void Grid::apply_f_frnn(Functor f, glm::vec3* pos, float rad) {
	LOG_EVENT("applying frnn");
	apply_f_frnn_kernel<Functor> <<<(h_gridp->numP / 64) + 1, 64 >>> (f, pos, start, stop, rad*rad, d_gridp, partId);
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
	const int nbCell = h_gridp->tot_num_cells;
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