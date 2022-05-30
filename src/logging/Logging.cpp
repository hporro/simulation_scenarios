#include "Logging.h"

void init_logging() {
    std::shared_ptr<spdlog::logger> time_logger = spdlog::basic_logger_mt("timings", "logs/timings.txt", true);
    time_logger->set_pattern("[%H:%M:%S %z] [thread %t] %v");
    std::shared_ptr<spdlog::logger> event_logger = spdlog::basic_logger_mt("events", "logs/events.txt", true);
    event_logger->set_pattern("[%H:%M:%S %z] [thread %t] %v");
};