#version 420 core

in layout (location = 0) vec3 aPos;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;

out vec4 gl_Position;

void main() {
	gl_Position = u_projection * u_view * u_model * vec4(aPos, 1.0f);
}
