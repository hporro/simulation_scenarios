#include <cuda.h>
#include <glm/glm.hpp>
#include <stdio.h>


glm::vec3 operator*(glm::vec3 v, float a) {
	return glm::vec3(v.x * a, v.y * a, v.z * a);
}

int main(int argc, char** argv)
{
	glm::vec3 a(10.0, 20.0, 2.0);
	a = glm::normalize(a) * 5.0;
	printf("a: %f, %f, %f\n", a.x, a.y, a.z);
}
