#include <iostream>

#include <tinytest.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <random>

#include "glm/vec2.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "../src/gpu/gpuErrCheck.h"
#include "../src/logging/Logging.h"
#include "../src/timing/timing_helpers.h"
#include "../src/data_structures/mcleap_wrapper.h"

struct SaveNeighborsFunctor {
	SaveNeighborsFunctor(float rad, int numP, int max_neighbors) : m_numP(numP), h_m_max_neighbors(max_neighbors) {
		cudaMalloc((void**)&m_rad, sizeof(float));
		cudaMalloc((void**)&m_max_neighbors, sizeof(int));
		cudaMalloc((void**)&m_num_neighbors, numP * sizeof(int));
		cudaMalloc((void**)&m_neighbors, h_m_max_neighbors * numP * sizeof(int));

		cudaMemcpy(m_rad, &rad, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(m_max_neighbors, &max_neighbors, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	~SaveNeighborsFunctor() {
		//cudaFree(m_rad);
		//cudaFree(m_max_neighbors);
		//cudaFree(m_num_neighbors);
		//cudaFree(m_neighbors);
	}
	void resetFunctor() {
		cudaMemset(m_num_neighbors, 0, m_numP * sizeof(int));
		cudaMemset(m_neighbors, 0, m_numP * h_m_max_neighbors * sizeof(int));
		cudaDeviceSynchronize();
	}
	__device__ void operator()(const int& i, const int& j, const glm::dvec2 dist_vec, const double dist) {
		if (i != j)if (dist <= *m_rad && dist > 0) {
			if(i==4)printf("i: %d j: %d dist: %f\n", i, j, dist);
			int ind = i * (*m_max_neighbors) + m_num_neighbors[i];
			if (ind < m_numP * (*m_max_neighbors))m_neighbors[ind] = j;
			m_num_neighbors[i]++;
		}
	}
	int m_numP, h_m_max_neighbors;
	float* m_rad;
	int* m_max_neighbors;
	int* m_num_neighbors;
	int* m_neighbors;
};

void lattice_test() {
	int numP = 10000;
	int max_neighs = 30;
	float rad = 1.0;

	triangulation2d<100, 100> gc(numP);

	SaveNeighborsFunctor* snfunctor = new SaveNeighborsFunctor(rad, numP, max_neighs);
	snfunctor->resetFunctor();

	glm::dvec2* pos = new glm::dvec2[numP];
	for (int x = 0; x < 100; x++) {
		for (int y = 0; y < 100; y++) {
			int i = x + 100 * +y;
			if (i > numP)continue;
			pos[i] = glm::dvec2(x, y);
		}
	}
	pos[0].x = -1000.0;
	pos[0].y = -1000.0;
	pos[1].x = 1000.0;
	pos[1].y = -1000.0;
	pos[2].x = 1000.0;
	pos[2].y = 1000.0;
	pos[3].x = -1000.0;
	pos[3].y = 1000.0;

	gc.build(pos);

	glm::dvec2* d_pos;
	cudaMalloc((void**)&d_pos, numP * sizeof(glm::dvec2));
	cudaMemcpy(d_pos, pos, numP * sizeof(glm::dvec2), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	//gc.update((MCleap::MCLEAP_VEC*)d_pos);
	cudaDeviceSynchronize();
	gc.apply_f_frnn<SaveNeighborsFunctor>(*snfunctor, d_pos, rad);
	cudaDeviceSynchronize();

	int* h_num_neighbors = new int[numP];
	int* h_neighbors = new int[numP * max_neighs];

	cudaMemcpy(h_num_neighbors, snfunctor->m_num_neighbors, numP * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_neighbors, snfunctor->m_neighbors, max_neighs * numP * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pos, gc.m->d_vbo_v, numP * sizeof(glm::dvec2), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();



	int* res_num_neighs = new int[numP];

	for (int i = 0; i < numP; i++) {
		res_num_neighs[i] = 0;
		for (int j = 0; j < numP; j++) {
			if (i == j)continue;
			glm::dvec2 dist_vec = pos[i] - pos[j];
			float dist = glm::dot(dist_vec, dist_vec);
			//if (i == 3)printf("i: %d j: %d dist: %f\n", i, j, sqrt(dist));
			if (dist <= rad * rad) {
				res_num_neighs[i]++;
			}
		}
	}

	for (int i = 0; i < numP; i++) {
		//printf("i: %d num_true: %d num_calc: %d\n", i, res_num_neighs[i], h_num_neighbors[i]);
		ASSERT_EQUALS(res_num_neighs[i], h_num_neighbors[i]);
	}

	delete[] res_num_neighs;

	delete[] pos;

	delete[] h_num_neighbors;
	delete[] h_neighbors;
	delete snfunctor;
	cudaFree(d_pos);
}

void packed_lattice_test() {
	int numP = 400;
	int max_neighs = 40;
	float rad = 1.0;
	triangulation2d<100, 100> gc(numP);

	SaveNeighborsFunctor* snfunctor = new SaveNeighborsFunctor(rad, numP, max_neighs);
	snfunctor->resetFunctor();

	glm::dvec2* pos = new glm::dvec2[numP];
	for (int x = 0; x < 20; x++) {
		for (int y = 0; y < 20; y++) {
			int i = x + 20 * y;
			if (i > numP)continue;
			pos[i] = glm::dvec2(x * 0.5, y * 0.5);
		}
	}
	pos[0].x = -1000.0;
	pos[0].y = -1000.0;
	pos[1].x = 1000.0;
	pos[1].y = -1000.0;
	pos[2].x = 1000.0;
	pos[2].y = 1000.0;
	pos[3].x = -1000.0;
	pos[3].y = 1000.0;

	gc.build(pos);

	glm::dvec2* d_pos;
	cudaMalloc((void**)&d_pos, numP * sizeof(glm::dvec2));
	cudaMemcpy(d_pos, pos, numP * sizeof(glm::dvec2), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	//gc.update(d_pos);
	cudaDeviceSynchronize();
	gc.apply_f_frnn<SaveNeighborsFunctor>(*snfunctor, d_pos, rad);
	cudaDeviceSynchronize();

	int* h_num_neighbors = new int[numP];
	int* h_neighbors = new int[numP * max_neighs];

	cudaMemcpy(h_num_neighbors, snfunctor->m_num_neighbors, numP * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_neighbors, snfunctor->m_neighbors, max_neighs * numP * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pos, d_pos, numP * sizeof(glm::dvec2), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int* res_num_neighs = new int[numP];

	for (int i = 0; i < numP; i++) {
		res_num_neighs[i] = 0;
		for (int j = 0; j < numP; j++) {
			if (i == j)continue;
			glm::dvec2 dist_vec = pos[i] - pos[j];
			float dist = glm::dot(dist_vec, dist_vec);
			if (dist <= rad * rad) {
				if(i==4)printf("i: %d j: %d pos[i]: %f %f pos[j]: %f %f dist: %f\n", i, j, pos[i].x, pos[i].y, pos[j].x, pos[j].y, sqrt(dist));
				res_num_neighs[i]++;
			}
		}

		printf("i: %d num_true: %d num_calc: %d\n", i, res_num_neighs[i], h_num_neighbors[i]);
		ASSERT_EQUALS(res_num_neighs[i], h_num_neighbors[i]);
	}

	delete[] res_num_neighs;
	delete[] pos;

	delete[] h_num_neighbors;
	delete[] h_neighbors;
	delete snfunctor;
	cudaFree(d_pos);
}
//
//void random_points_test() {
//	int numP = 10000;
//	int max_neighs = 40;
//	float rad = 1.0;
//	glm::dvec2 min(-50.0);
//	glm::dvec2 cell_size(1.0);
//	glm::ivec2 num_cells(100);
//	GridCount2d gc(numP, min, cell_size, num_cells);
//
//	SaveNeighborsFunctor* snfunctor = new SaveNeighborsFunctor(rad, numP, max_neighs);
//	snfunctor->resetFunctor();
//
//	std::random_device dev;
//	std::mt19937 rng{ dev() };
//	rng.seed(10);
//	std::uniform_real_distribution<> dista(-50.0, 50.0);
//
//	glm::dvec2* pos = new glm::dvec2[numP];
//	for (int i = 0; i < numP; i++) {
//		pos[i] = glm::dvec2(dista(rng), dista(rng));
//	}
//
//	glm::dvec2* d_pos;
//	cudaMalloc((void**)&d_pos, numP * sizeof(glm::dvec2));
//	cudaMemcpy(d_pos, pos, numP * sizeof(glm::dvec2), cudaMemcpyHostToDevice);
//	cudaDeviceSynchronize();
//
//	gc.update(d_pos);
//	cudaDeviceSynchronize();
//	gc.apply_f_frnn<SaveNeighborsFunctor>(*snfunctor, d_pos, rad);
//	cudaDeviceSynchronize();
//
//	int* h_num_neighbors = new int[numP];
//	int* h_neighbors = new int[numP * max_neighs];
//
//	cudaMemcpy(h_num_neighbors, snfunctor->m_num_neighbors, numP * sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_neighbors, snfunctor->m_neighbors, max_neighs * numP * sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(pos, d_pos, numP * sizeof(glm::dvec2), cudaMemcpyDeviceToHost);
//	cudaDeviceSynchronize();
//
//	int* res_num_neighs = new int[numP];
//
//	for (int i = 0; i < numP; i++) {
//		if (i % 1000)printProgress((double)i * (1.0 / (double)numP));
//
//
//		res_num_neighs[i] = 0;
//		for (int j = 0; j < numP; j++) {
//			if (i == j)continue;
//			glm::dvec2 dist_vec = pos[i] - pos[j];
//			float dist = glm::dot(dist_vec, dist_vec);
//			const double EPS = 0.0001;
//			if ((dist - EPS) <= (rad * rad) || (sqrt(dist - EPS) <= rad)) {
//				//printf("i: %d j: %d hi: %d hj: %d pos[i]: %f %f %f pos[j]: %f %f %f dist: %f\n", i, j, calcHash(pos[i], gc.h_gcdata), calcHash(pos[j], gc.h_gcdata), pos[i].x, pos[i].y, pos[i].z, pos[j].x, pos[j].y, pos[j].z, sqrt(dist));
//				res_num_neighs[i]++;
//			}
//		}
//
//		//printf("i: %d num_neighs: %d h_num_neighs: %d pos[i]: % f %f %f\n", i, res_num_neighs[i], h_num_neighbors[i], pos[i].x, pos[i].y, pos[i].z);
//		for (int j = 0; j < h_num_neighbors[i]; j++) {
//			double leng = glm::length(pos[h_neighbors[i * max_neighs + j]] - pos[i]);
//			//printf("j: %d pos: %f %f %f dist: %f\n", h_neighbors[i * max_neighs + j], pos[h_neighbors[i * max_neighs + j]].x, pos[h_neighbors[i * max_neighs + j]].y, pos[h_neighbors[i * max_neighs + j]].z, leng);
//		}
//
//		if (abs(res_num_neighs[i] - h_num_neighbors[i]) > 1)printf("i: %d num_neighs: %d h_num_neighs: %d pos[i]: % f %f\n", i, res_num_neighs[i], h_num_neighbors[i], pos[i].x, pos[i].y);
//		ASSERT_TRUE(abs(res_num_neighs[i] - h_num_neighbors[i]) <= 1);
//	}
//
//	delete[] res_num_neighs;
//	delete[] pos;
//
//	delete[] h_num_neighbors;
//	delete[] h_neighbors;
//	delete snfunctor;
//	cudaFree(d_pos);
//}
//
//void more_packed_lattice_test() {
//	int numP = 64000;
//	int max_neighs = 10;
//	float rad = 0.25;
//	glm::dvec2 min(0.0);
//	glm::dvec2 cell_size(1.0);
//	glm::ivec2 num_cells(10);
//	GridCount2d gc(numP, min, cell_size, num_cells);
//
//	SaveNeighborsFunctor* snfunctor = new SaveNeighborsFunctor(rad, numP, max_neighs);
//	snfunctor->resetFunctor();
//	cudaDeviceSynchronize();
//	gpuErrchk(cudaGetLastError());
//
//	glm::dvec2* pos = new glm::dvec2[numP];
//	for (int x = 0; x < 40; x++) {
//		for (int y = 0; y < 40; y++) {
//			for (int z = 0; z < 40; z++) {
//				int i = x + 40 * y;
//				if (i > numP)continue;
//				pos[i] = glm::dvec2(x * 0.25, y * 0.25);
//			}
//		}
//	}
//
//	glm::dvec2* d_pos;
//	cudaMalloc((void**)&d_pos, numP * sizeof(glm::dvec2));
//	cudaMemcpy(d_pos, pos, numP * sizeof(glm::dvec2), cudaMemcpyHostToDevice);
//	cudaDeviceSynchronize();
//	gpuErrchk(cudaGetLastError());
//
//	gc.update(d_pos);
//	cudaDeviceSynchronize();
//	gc.apply_f_frnn<SaveNeighborsFunctor>(*snfunctor, d_pos, rad);
//	cudaDeviceSynchronize();
//	gpuErrchk(cudaGetLastError());
//
//	int* h_num_neighbors = new int[numP];
//	int* h_neighbors = new int[numP * max_neighs];
//
//	gpuErrchk(cudaMemcpy(h_num_neighbors, snfunctor->m_num_neighbors, numP * sizeof(int), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(h_neighbors, snfunctor->m_neighbors, max_neighs * numP * sizeof(int), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(pos, d_pos, numP * sizeof(glm::dvec2), cudaMemcpyDeviceToHost));
//	cudaDeviceSynchronize();
//	gpuErrchk(cudaGetLastError());
//
//	int* res_num_neighs = new int[numP];
//
//	for (int i = 0; i < numP; i++) {
//		printProgress((double)i / (double)numP);
//		res_num_neighs[i] = 0;
//		for (int j = 0; j < numP; j++) {
//			if (i == j)continue;
//			glm::dvec2 dist_vec = pos[i] - pos[j];
//			float dist = glm::dot(dist_vec, dist_vec);
//			if (dist <= rad * rad || sqrt(dist) <= rad) {
//				//printf("i: %d j: %d hi: %d hj: %d pos[i]: %f %f %f pos[j]: %f %f %f dist: %f\n", i, j, calcHash(pos[i], gc.h_gcdata), calcHash(pos[j], gc.h_gcdata), pos[i].x, pos[i].y, pos[i].z, pos[j].x, pos[j].y, pos[j].z, sqrt(dist));
//				res_num_neighs[i]++;
//			}
//		}
//
//		//printf("i: %d num_neighs: %d h_num_neighs: %d pos[i]: % f %f %f\n", i, res_num_neighs[i], h_num_neighbors[i], pos[i].x, pos[i].y, pos[i].z);
//		//for (int j = 0; j < h_num_neighbors[i]; j++) {
//		//	double leng = glm::length(pos[h_neighbors[i * max_neighs + j]] - pos[i]);
//		//	printf("j: %d pos: %f %f %f dist: %f\n", h_neighbors[i * max_neighs + j], pos[h_neighbors[i * max_neighs + j]].x, pos[h_neighbors[i * max_neighs + j]].y, pos[h_neighbors[i * max_neighs + j]].z, leng);
//		//}
//
//		ASSERT_EQUALS(res_num_neighs[i], h_num_neighbors[i]);
//	}
//
//	delete[] res_num_neighs;
//	delete[] pos;
//
//	delete[] h_num_neighbors;
//	delete[] h_neighbors;
//	delete snfunctor;
//	cudaFree(d_pos);
//}

int main() {
	init_logging();
	RUN(lattice_test);
	printf("\n");
	RUN(packed_lattice_test);
//	RUN(more_packed_lattice_test);
//	RUN(random_points_test);
	return TEST_REPORT();
}
