#version 420 core

out vec4 FragColor;
in vec2 quadPos;
in vec4 viewPos;
uniform float u_radius;

void main() {
    vec3 normal;
    vec2 dq = quadPos.xy;
    if(length(dq)>1) {
        discard;
    } else {
        normal = vec3(dq.x/sqrt(-dq.x*dq.x-dq.y*dq.y+1),dq.y/sqrt(-dq.x*dq.x-dq.y*dq.y+1),1);
    }
    float diff = max(dot(normal, normalize(vec3(0.0,1.0,0.0))),0.0);
    vec4 diffColor = vec4(0.2,0.2,0.7,1.0);
    FragColor = diffColor * diff;
    //FragColor = vec4(normal,1.0);
} 

