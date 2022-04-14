#pragma once

#include "cuda_gl_interop.h"

#define GLCHECKERR() glCheckError(__FILE__, __LINE__)
void glCheckError(const char* file, unsigned int line) {
    GLenum errorCode = glGetError();

    while (errorCode != GL_NO_ERROR) {
        std::string fileString(file);
        std::string error = "unknown error";

        // clang-format off
        switch (errorCode) {
        case GL_INVALID_ENUM:      error = "GL_INVALID_ENUM"; break;
        case GL_INVALID_VALUE:     error = "GL_INVALID_VALUE"; break;
        case GL_INVALID_OPERATION: error = "GL_INVALID_OPERATION"; break;
        case GL_STACK_OVERFLOW:    error = "GL_STACK_OVERFLOW"; break;
        case GL_STACK_UNDERFLOW:   error = "GL_STACK_UNDERFLOW"; break;
        case GL_OUT_OF_MEMORY:     error = "GL_OUT_OF_MEMORY"; break;
        }
        // clang-format on

        std::cerr << "OpenglError : file=" << file << " line=" << line
            << " error:" << error << std::endl;
        errorCode = glGetError();
    }
}

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk(ans) (ans)

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}