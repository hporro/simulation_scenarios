cmake_minimum_required(VERSION 3.11)

project(simulation_scenarios 
		LANGUAGES CUDA CXX C)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_definitions(-DGLEW_STATIC)
add_subdirectory(lib/glfw EXCLUDE_FROM_ALL)
add_subdirectory(lib/glew EXCLUDE_FROM_ALL)
add_subdirectory(lib/glm EXCLUDE_FROM_ALL)

set (IMGUI_SRCs
	lib/imgui/imgui.cpp
	lib/imgui/imgui_draw.cpp
	lib/imgui/imgui_tables.cpp
	lib/imgui/imgui_widgets.cpp
	lib/imgui/backends/imgui_impl_glfw.cpp 
	lib/imgui/backends/imgui_impl_opengl3.cpp
)

file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})

add_executable(main main.cu ${IMGUI_SRCs})
target_link_libraries(main
	PRIVATE glfw
	PRIVATE libglew_static
	PRIVATE glm
)

target_compile_options(main PRIVATE -Wall)
target_include_directories(main PRIVATE lib/spdlog/include lib/imgui include)

add_executable(playground playground.cu)
target_link_libraries(playground
	PRIVATE glm
)

add_subdirectory(test)