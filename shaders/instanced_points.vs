#version 420 core

in layout (location = 0) vec3 aPos;
in layout (location = 1) vec3 aQuadPos;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;
uniform float u_radius;

out vec4 gl_Position;
out vec2 quadPos;
out vec4 viewPos;

void main() {
	quadPos = (aQuadPos/0.05).xy;
	viewPos = (u_view * (u_model * vec4(aPos, 1.0f)) + vec4(aQuadPos * u_radius, 0.0));
	gl_Position = u_projection * viewPos;
}
