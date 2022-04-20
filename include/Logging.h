#pragma once

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

void init_logging() {
    std::shared_ptr<spdlog::logger> time_logger = spdlog::basic_logger_mt("timings", "logs/timings.txt", true);
    time_logger->set_pattern("[%H:%M:%S %z] [thread %t] %v");
    std::shared_ptr<spdlog::logger> event_logger = spdlog::basic_logger_mt("events", "logs/events.txt", true);
    event_logger->set_pattern("[%H:%M:%S %z] [thread %t] %v");
};

#define LOG_EVENT(...) spdlog::get("events")->info(__VA_ARGS__); spdlog::get("events")->flush()
#define LOG_TIMING(...) spdlog::get("timings")->info(__VA_ARGS__)
