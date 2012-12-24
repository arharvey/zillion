#version 120

uniform mat4 projectionXf;
uniform mat4 viewXf;
uniform mat3 normalXf;
uniform vec3 lightDirWorld;

attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;

varying float outIntensity;
varying vec2 outUV;

void main ()
{
    gl_Position = projectionXf * viewXf * vec4(position, 1.0);

    outIntensity = dot(normalize(normalXf * normal), -lightDirWorld);
    if(outIntensity < 0.0)
        outIntensity = 0.0;

    outUV = uv;
}
