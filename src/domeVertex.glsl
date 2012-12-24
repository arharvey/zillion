#version 120

uniform mat4 projectionXf;
uniform mat4 viewXf;

attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;

varying vec2 outUV;

void main ()
{
    gl_Position = projectionXf * viewXf * vec4(position, 1.0);
    outUV = uv;
}
