cmake_minimum_required(VERSION 3.20)

project(BATEA
		LANGUAGES CUDA CXX C)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CUDA_STANDARD 14)
SET(CMAKE_CUDA_ARCHITECTURES 80)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_definitions(-DGLEW_STATIC)
add_subdirectory(lib/glfw EXCLUDE_FROM_ALL)
add_subdirectory(lib/glew EXCLUDE_FROM_ALL)
add_subdirectory(lib/glm EXCLUDE_FROM_ALL)

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

add_executable(SPH simulations/SPH.cu ${IMGUI_SRCs})
target_link_libraries(SPH
	PRIVATE BATEA_RENDER
	PRIVATE BATEA_MATH
)

target_compile_options(SPH PRIVATE -Wall)
target_include_directories(SPH PRIVATE lib/spdlog/include lib/imgui include)

add_executable(boids simulations/boids.cu ${IMGUI_SRCs})
target_link_libraries(boids
	PRIVATE BATEA_RENDER
	PRIVATE BATEA_MATH
)

target_compile_options(boids PRIVATE -Wall)
target_include_directories(boids PRIVATE lib/spdlog/include lib/imgui include)

# ------------- PLAYGROUND ----------------

add_executable(playground playground.cu)
target_link_libraries(playground
	PRIVATE glm
)

# ------------- TESTS ----------------
add_subdirectory(test)