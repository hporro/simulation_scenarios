#pragma once

__device__ float clamp(float val, float min, float max) {
	if (val < min)return min;
	if (val > max)return max;
	return val;
}

__device__ glm::vec3 operator*(glm::vec3 v, float a) {
	return glm::vec3(v.x * a, v.y * a, v.z * a);
}
__device__ glm::vec3 operator*(float a, glm::vec3 v) {
	return glm::vec3(v.x * a, v.y * a, v.z * a);
}
__device__ glm::vec3 operator/(glm::vec3 v, float a) {
	return glm::vec3(v.x / a, v.y / a, v.z / a);
}