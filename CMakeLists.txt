cmake_minimum_required(VERSION 3.20)

project(BATEA
		LANGUAGES CUDA CXX C)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CUDA_STANDARD 14)
SET(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_definitions(-DGLEW_STATIC)
add_subdirectory(lib/glfw EXCLUDE_FROM_ALL)
add_subdirectory(lib/glew EXCLUDE_FROM_ALL)
add_subdirectory(lib/glm EXCLUDE_FROM_ALL)
add_subdirectory(lib/mcleap)

# --------------- BATEA LOGGING ---------------
add_library(BATEA_LOGGING src/logging/Logging.cpp)
target_include_directories(BATEA_LOGGING PUBLIC lib/spdlog/include)

# --------------- BATEA TIMING ---------------
add_library(BATEA_TIMING src/timing/timing_helpers.cpp)

# --------------- BATEA MATH ---------------
add_library(BATEA_MATH INTERFACE src/math/batea_math.cuh)
target_include_directories(BATEA_MATH INTERFACE lib/glm)

# --------- BATEA DATA STRUCTURES ----------
## TODO

# -------------- BATEA RENDER --------------
set (IMGUI_SRCs
	lib/imgui/imgui.cpp
	lib/imgui/imgui_draw.cpp
	lib/imgui/imgui_tables.cpp
	lib/imgui/imgui_widgets.cpp
	lib/imgui/backends/imgui_impl_glfw.cpp 
	lib/imgui/backends/imgui_impl_opengl3.cpp
)

add_library(BATEA_RENDER src/rendering/Application.cpp ${IMGUI_SRCs})
target_link_libraries(BATEA_RENDER 
	PUBLIC glfw
	PUBLIC libglew_static
	PRIVATE BATEA_LOGGING
	PRIVATE BATEA_TIMING
)
target_include_directories(BATEA_RENDER PUBLIC lib/imgui)

file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})


# ------------- SIMULATIONS ----------------

function(make_simulation_target target_name)
add_executable(${target_name} ${ARGN})
target_link_libraries(${target_name}
	PRIVATE BATEA_RENDER
	PRIVATE BATEA_MATH
	PUBLIC MCleap
)
target_compile_options(${target_name} PRIVATE -Wall)
target_include_directories(${target_name} PRIVATE lib/spdlog/include lib/imgui include lib/mcleap/include)
endfunction()

make_simulation_target(SPH simulations/SPH.cu)
make_simulation_target(boids simulations/boids.cu)
make_simulation_target(boids2d simulations/boids2d.cu)

# ------------- PLAYGROUND ----------------

add_executable(playground playground.cu)
target_link_libraries(playground
	PRIVATE glm MCleap
)
target_include_directories(playground PRIVATE lib/glm lib/mcleap/include)

# ------------- TESTS ----------------
add_subdirectory(test)