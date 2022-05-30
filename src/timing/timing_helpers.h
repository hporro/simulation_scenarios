#pragma once

#include <chrono>
#include <fstream>
#include <iostream>

struct Timer {
	Timer();
	float get_time();
	float swap_time();
	std::chrono::high_resolution_clock::time_point begin, end;
};

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage);
