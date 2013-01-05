#version 150

uniform mat4 projectionXf;
uniform mat4 viewXf;
uniform mat3 normalXf;
uniform vec3 lightDirWorld;

in vec3 position;
in vec3 normal;
in vec2 uv;

out float outIntensity;
out vec2 outUV;

void main ()
{
    gl_Position = projectionXf * viewXf * vec4(position, 1.0);

    outIntensity = dot(normalize(normalXf * normal), -lightDirWorld);
    if(outIntensity < 0.0)
        outIntensity = 0.0;

    outUV = uv;
}
