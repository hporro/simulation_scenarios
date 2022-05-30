#include "timing_helpers.h"

Timer::Timer() { begin = std::chrono::high_resolution_clock::now(); }
float Timer::get_time() { end = std::chrono::high_resolution_clock::now(); return std::chrono::duration<float, std::milli>(end - begin).count(); }
float Timer::swap_time() { end = std::chrono::high_resolution_clock::now(); auto diff = std::chrono::duration<float, std::milli>(end - begin); begin = end; return diff.count(); }

void printProgress(double percentage) {
	int val = (int)(percentage * 100);
	int lpad = (int)(percentage * PBWIDTH);
	int rpad = PBWIDTH - lpad;
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}
