add_executable(test_grid_count test_grid_count.cu)
target_include_directories(test_grid_count PRIVATE ${CMAKE_SOURCE_DIR}/lib/tinytest ${CMAKE_SOURCE_DIR}/lib/glm ${CMAKE_SOURCE_DIR}/lib/spdlog/include)
target_link_libraries(test_grid_count PRIVATE BATEA_TIMING BATEA_LOGGING)

add_executable(test_grid test_grid.cu)
target_include_directories(test_grid PRIVATE ${CMAKE_SOURCE_DIR}/lib/tinytest ${CMAKE_SOURCE_DIR}/lib/glm ${CMAKE_SOURCE_DIR}/lib/spdlog/include)
target_link_libraries(test_grid PRIVATE BATEA_TIMING BATEA_LOGGING)