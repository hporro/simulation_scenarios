#pragma once

#define GLM_FORCE_RADIANS
#include <GL/glew.h>
#include <initializer_list>
#include <map>
#include <string>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../logging/Logging.h"

class Shader;
class ShaderProgram;

// Loads a shader from a file into OpenGL.
class Shader {
public:
	// Load Shader from a file
	Shader(const std::string& filename, GLenum type);

	// provide opengl shader identifiant.
	GLuint getHandle() const;

	~Shader();

private:
	// opengl program identifiant
	GLuint handle;

	friend class ShaderProgram;
};

// A shader program is a set of shader (for instance vertex shader + pixel
// shader) defining the rendering pipeline.
//
// This class provide an interface to define the OpenGL uniforms and attributes
// using GLM objects.
class ShaderProgram {
public:
	// constructor
	ShaderProgram(std::initializer_list<Shader> shaderList);

	// bind the program
	void use() const;
	void unuse() const;

	// provide the opengl identifiant
	GLuint getHandle() const;

	// clang-format off
	// provide attributes informations.
	GLint attribute(const std::string& name);
	void setAttribute(const std::string& name, GLint size, GLsizei stride, GLuint offset, GLboolean normalize, GLenum type);
	void setAttribute(const std::string& name, GLint size, GLsizei stride, GLuint offset, GLboolean normalize);
	void setAttribute(const std::string& name, GLint size, GLsizei stride, GLuint offset, GLenum type);
	void setAttribute(const std::string& name, GLint size, GLsizei stride, GLuint offset);
	// clang-format on

	// provide uniform location
	GLint uniform(const std::string& name);
	GLint operator[](const std::string& name);

	// affect uniform
	void setUniform(const std::string& name, float x, float y, float z);
	void setUniform(const std::string& name, float val);
	void setUniform(const std::string& name, double val);
	void setUniform(const std::string& name, int val);
	void setUniform(const std::string& name, glm::mat4 val);

	~ShaderProgram();

private:
	ShaderProgram();

	std::map<std::string, GLint> uniforms;
	std::map<std::string, GLint> attributes;

	// opengl id
	GLuint handle;

	void link();
};

// file reading
void getFileContents(const char* filename, std::vector<char>& buffer) {
	std::ifstream file(filename, std::ios_base::binary);
	if (file) {
		file.seekg(0, std::ios_base::end);
		std::streamsize size = file.tellg();
		if (size > 0) {
			file.seekg(0, std::ios_base::beg);
			buffer.resize(static_cast<size_t>(size));
			file.read(&buffer[0], size);
		}
		buffer.push_back('\0');
	}
	else {
		LOG_EVENT("The file {} doesn't exists", filename);
	}
}

Shader::Shader(const std::string& filename, GLenum type) {
	// file loading
	std::vector<char> fileContent;
	getFileContents(filename.c_str(), fileContent);

	// creation
	handle = glCreateShader(type);
	if (handle == 0) {
		LOG_EVENT("Couldn't create the new shader");
	}
	// code source assignation
	const char* shaderText(&fileContent[0]);
	glShaderSource(handle, 1, (const GLchar**)&shaderText, NULL);

	// compilation
	glCompileShader(handle);

	// compilation check
	GLint compile_status;
	glGetShaderiv(handle, GL_COMPILE_STATUS, &compile_status);
	if (compile_status != GL_TRUE) {
		GLsizei logsize = 0;
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logsize);

		char* log = new char[logsize + 1];
		glGetShaderInfoLog(handle, logsize, &logsize, log);

		LOG_EVENT("Compilation error");
	}
	else {
		LOG_EVENT("Shader compiled successfully");
	}
}

GLuint Shader::getHandle() const {
	return handle;
}

Shader::~Shader() {}

ShaderProgram::ShaderProgram() {
	handle = glCreateProgram();
	if (!handle) { LOG_EVENT("Couldn't create the new shader program"); }
	else { LOG_EVENT("Shader program created successfully"); }
}

ShaderProgram::ShaderProgram(std::initializer_list<Shader> shaderList)
	: ShaderProgram() {
	for (auto& s : shaderList)
		glAttachShader(handle, s.getHandle());

	LOG_EVENT("Shaders attached to the program");
	link();
}

void ShaderProgram::link() {
	glLinkProgram(handle);
	GLint result;
	glGetProgramiv(handle, GL_LINK_STATUS, &result);
	if (result != GL_TRUE) {

		GLsizei logsize = 0;
		glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &logsize);

		char* log = new char[logsize];
		glGetProgramInfoLog(handle, logsize, &logsize, log);

		LOG_EVENT("Linkage error: {}", log);
	}
	else {
		LOG_EVENT("Program linked successfully");
	}
}

GLint ShaderProgram::uniform(const std::string& name) {
	auto it = uniforms.find(name);
	if (it == uniforms.end()) {
		// uniform that is not referenced
		GLint r = glGetUniformLocation(handle, name.c_str());
		if (r == GL_INVALID_OPERATION || r < 0) {
			LOG_EVENT("Uniform {} doesn't exist in the program", name);
		}
		// add it anyways
		uniforms[name] = r;

		return r;
	}
	else
		return it->second;
}

GLint ShaderProgram::attribute(const std::string& name) {
	GLint attrib = glGetAttribLocation(handle, name.c_str());
	if (attrib == GL_INVALID_OPERATION || attrib < 0) {
		LOG_EVENT("Attribute {} doesn't exist in program", name);
	}
	return attrib;
}

void ShaderProgram::setAttribute(const std::string& name,
	GLint size,
	GLsizei stride,
	GLuint offset,
	GLboolean normalize,
	GLenum type) {
	GLint loc = attribute(name);
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc, size, type, normalize, stride,
		reinterpret_cast<void*>(offset));
}

void ShaderProgram::setAttribute(const std::string& name,
	GLint size,
	GLsizei stride,
	GLuint offset,
	GLboolean normalize) {
	setAttribute(name, size, stride, offset, normalize, GL_FLOAT);
}

void ShaderProgram::setAttribute(const std::string& name,
	GLint size,
	GLsizei stride,
	GLuint offset,
	GLenum type) {
	setAttribute(name, size, stride, offset, false, type);
}

void ShaderProgram::setAttribute(const std::string& name,
	GLint size,
	GLsizei stride,
	GLuint offset) {
	setAttribute(name, size, stride, offset, false, GL_FLOAT);
}

void ShaderProgram::setUniform(const std::string& name,
	float x,
	float y,
	float z) {
	glUniform3f(uniform(name), x, y, z);
}

void ShaderProgram::setUniform(const std::string& name, float val) {
	glUniform1f(uniform(name), val);
}

void ShaderProgram::setUniform(const std::string& name, double val) {
	glUniform1f(uniform(name), (float)val);
}

void ShaderProgram::setUniform(const std::string& name, int val) {
	glUniform1i(uniform(name), val);
}

void ShaderProgram::setUniform(const std::string& name, glm::mat4 val) {
	glUniformMatrix4fv(uniform(name), 1, GL_FALSE, &val[0][0]);
}

ShaderProgram::~ShaderProgram() {
	glDeleteProgram(handle);
}

void ShaderProgram::use() const {
	glUseProgram(handle);
}
void ShaderProgram::unuse() const {
	glUseProgram(0);
}

GLuint ShaderProgram::getHandle() const {
	return handle;
}
