
add_executable(test_grid2 test_grid2.cu)
target_include_directories(test_grid2 PRIVATE ${CMAKE_SOURCE_DIR}/lib/tinytest ${CMAKE_SOURCE_DIR}/lib/glm ${CMAKE_SOURCE_DIR}/lib/spdlog/include)