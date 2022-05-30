#version 420 core

out vec4 FragColor;
in vec2 quadPos;
in vec4 viewPos;
uniform float u_radius;

void main() {
    vec3 normal;
    vec2 dq = quadPos.xy;
    if(length(dq)>=1) {
        discard;
    } else {
        normal = normalize(vec3(dq.x,dq.y,sqrt(1-dq.x*dq.x-dq.y*dq.y)));
    }
    float diff = max(dot(normal, normalize(vec3(0.0,1.0,0.0))),0.0);
    diff += max(dot(normal, vec3(0.3,0.0,0.3)),0.0);

    vec4 diffColor = vec4(0.3,0.2,0.7,1.0);
    FragColor = diffColor * diff;
    //FragColor = 0.5*(vec4(1.0) + vec4(normal,1.0));
} 

