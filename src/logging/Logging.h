#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

void init_logging();

#define LOG_EVENT(...) spdlog::get("events")->info(__VA_ARGS__); spdlog::get("events")->flush()
#define LOG_TIMING(...) spdlog::get("timings")->info(__VA_ARGS__)
