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


__device__ float sdCircle(glm::vec3 p, float r)
{
	return glm::length(p) - r;
}

__device__ glm::vec3 normal_circle(glm::vec3 p, float r) // for function f(p)
{
	float eps = 0.001; // or some other value
	glm::vec3 dx(eps, 0, 0);
	glm::vec3 dy(0, eps, 0);
	glm::vec3 dz(0, 0, eps);
	return glm::normalize(glm::vec3(sdCircle(p + dx, r) - sdCircle(p - dx, r),
		sdCircle(p + dy, r) - sdCircle(p - dy, r),
		sdCircle(p + dz, r) - sdCircle(p - dz, r)));
}


template<class T>
__device__ T h_max(T a, T b) {
	return a > b ? a : b;
}

template<class T>
__device__ T h_min(T a, T b) {
	return a < b ? a : b;
}

__device__ glm::vec3 v_max(glm::vec3 a, glm::vec3 b) {
	return glm::vec3(h_max(a.x, b.x), h_max(a.y, b.y), h_max(a.z, b.z));
}

__device__ float v_max(glm::vec3 a) {
	return h_max(h_max(a.x, a.y), a.z);
}

__device__ float sdBox(glm::vec3 q, glm::vec3 b)
{
	glm::vec3 d = glm::abs(q) - b;
	return glm::length(v_max(d, glm::vec3(0.0))) + h_min(v_max(d), 0.0f);
}

__device__ glm::vec3 normal_bx(glm::vec3 p, glm::vec3 b) // for function f(p)
{
	float eps = 0.001; // or some other value
	glm::vec3 dx(eps, 0, 0);
	glm::vec3 dy(0, eps, 0);
	glm::vec3 dz(0, 0, eps);
	return glm::normalize(glm::vec3(sdBox(p + dx, b) - sdBox(p - dx, b),
		sdBox(p + dy, b) - sdBox(p - dy, b),
		sdBox(p + dz, b) - sdBox(p - dz, b)));
}