#include <cuda.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>
#include <random>
#include <MCleap.h>

int main(int argc, char** argv)
{

	constexpr int numParticles = 10000;
	double* h_pos;
	h_pos = new double[2*numParticles];

	std::random_device dev;
	std::mt19937 rng{ dev() };;
	rng.seed(10);
	std::uniform_real_distribution<> distx(-100.0, 100.0);

	for (int i = 0; i < 2*numParticles; i++) {
		h_pos[i] = distx(rng);
	}

	MCleap::build_triangulation_from_buffer(numParticles, (MCleap::MCLEAP_REAL*)h_pos);
}
