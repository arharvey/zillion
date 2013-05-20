#version 150

uniform mat4 projectionXf;
uniform mat4 modelViewXf;
uniform mat3 normalXf;
uniform vec3 lightDirWorld;
uniform vec3 color;

in vec3 normal;
in vec3 position;
in vec2 uv;

out float outIntensity;
out vec2 outUV;
out vec3 outSurfaceColor;

void main()
{
    gl_Position = projectionXf * modelViewXf * vec4(position, 1.0);

    outSurfaceColor = color;

    outIntensity = dot(normalize(normalXf * normal), -lightDirWorld);
    if(outIntensity < 0.0)
        outIntensity = 0.0;

    outUV = uv;
}
