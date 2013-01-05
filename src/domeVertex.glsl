#version 150

uniform mat4 projectionXf;
uniform mat4 viewXf;

in vec3 position;
in vec3 normal;
in vec2 uv;

out vec2 outUV;

void main ()
{
    gl_Position = projectionXf * viewXf * vec4(position, 1.0);
    outUV = uv;
}
