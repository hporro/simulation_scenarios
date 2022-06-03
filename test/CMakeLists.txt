function(make_batea_test target_name)
add_executable(${target_name} ${ARGN})
target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/lib/tinytest ${CMAKE_SOURCE_DIR}/lib/glm ${CMAKE_SOURCE_DIR}/lib/spdlog/include)
target_link_libraries(${target_name} PRIVATE BATEA_TIMING BATEA_LOGGING)
endfunction()

make_batea_test(test_grid_count test_grid_count.cu)
make_batea_test(test_grid_count2d test_grid_count2d.cu)
make_batea_test(test_grid test_grid.cu)
