#pragma once

__device__ float clamp(float val, float min, float max) {
	if (val < min)return min;
	if (val > max)return max;
	return val;
}

__device__ glm::vec4 operator*(glm::vec4 v, float a) {
	return glm::vec4(v.x * a, v.y * a, v.z * a, v.w * a);
}
__device__ glm::vec4 operator*(float a, glm::vec4 v) {
	return glm::vec4(v.x * a, v.y * a, v.z * a, v.w * a);
}
__device__ glm::vec4 operator/(glm::vec4 v, float a) {
	return glm::vec4(v.x / a, v.y / a, v.z / a, v.w / a);
}