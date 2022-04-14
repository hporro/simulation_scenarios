#pragma once

#include <chrono>
#include <fstream>
#include <iostream>

struct Timer {
	Timer() { begin = std::chrono::high_resolution_clock::now(); }
	float get_time() { end = std::chrono::high_resolution_clock::now(); return std::chrono::duration<float, std::milli>(end - begin).count(); }
	float swap_time() { end = std::chrono::high_resolution_clock::now(); auto diff = std::chrono::duration<float, std::milli>(end - begin); begin = end; return diff.count(); }
	std::chrono::high_resolution_clock::time_point begin, end;
};

template<int i>
struct StaticCounter {
	static long long int count;
	static long long int getCount() {
		return count;
	}
	static void addOne() {
		count++;
	}
};

#define CREATE_STATIC_COUNTER(i) long long int StaticCounter<i>::count = 0;

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage) {
	int val = (int)(percentage * 100);
	int lpad = (int)(percentage * PBWIDTH);
	int rpad = PBWIDTH - lpad;
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}
