#include <iostream>

#include <tinytest.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "glm/vec3.hpp"
#include "glm/glm.hpp"

#include "../include/gpuErrCheck.h"
#include "../include/Logging.h"
#include "../include/timing_helpers.h"
#include "../grid2.h"

struct SaveNeighborsFunctor {
	SaveNeighborsFunctor(float rad, int numP, int max_neighbors) : m_numP(numP) {
		cudaMalloc((void**)&m_rad,		   sizeof(float));
		cudaMalloc((void**)&m_max_neighbors, sizeof(int));
		cudaMalloc((void**)&m_num_neighbors, numP*sizeof(int));
		cudaMalloc((void**)&m_neighbors,     numP*sizeof(int));
		
		cudaMemcpy(m_rad, &rad, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(m_max_neighbors, &max_neighbors, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

	}
	~SaveNeighborsFunctor() {
		cudaFree(m_rad);
		cudaFree(m_max_neighbors);
		cudaFree(m_num_neighbors);
		cudaFree(m_neighbors);
	}
	void resetFunctor() {
		cudaMemset(m_num_neighbors, 0, m_numP * sizeof(int));
		cudaMemset(m_neighbors,     0, m_numP * sizeof(int));
	}
	__device__ void operator()(const int& i, const int& j, const glm::vec3 dist_vec, const double dist) {
		printf("i: %d j: %d\n", i, j);
		if(i!=j)if (dist <= *m_rad)m_neighbors[i * (*m_max_neighbors) + atomicAdd(&m_num_neighbors[i],1)] = j;
	}
	int m_numP;
	float* m_rad;
	int* m_max_neighbors;
	int* m_num_neighbors;
	int* m_neighbors;
};

void test_neighbors() {
	int numP = 1000;
	glm::vec3 min(0.0);
	glm::vec3 cell_size(1.0);
	glm::ivec3 num_cells(10);
	GridCount gc(numP, min, cell_size, num_cells);

	SaveNeighborsFunctor snfunctor(1.0, numP, 20);
	snfunctor.resetFunctor();

	glm::vec3* pos = new glm::vec3[numP];
	for (int x = 0; x < 10; x++) {
		for (int y = 0; y < 10; y++) {
			for (int z = 0; z < 10; z++) {
				int i = x + 10 * (z * 10 + y);
				pos[i] = glm::vec3((x%10)*0.5, (y%10)*0.5, (z%10)*0.5);
				printf("x: %d y: %d z: %d pos[i]: %f %f %f\n", x, y, z, pos[i].x, pos[i].y, pos[i].z);
			}
		}
	}
	glm::vec3* d_pos;
	cudaMalloc((void**)&d_pos, numP*sizeof(glm::vec3));
	cudaMemcpy(d_pos, &pos, numP*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	gc.apply_f_frnn<SaveNeighborsFunctor>(snfunctor, d_pos, 2.0);

}


int main() {
	init_logging();
	RUN(test_neighbors);
	return TEST_REPORT();
}
