#version 150

uniform float scale;

uniform mat4 projectionXf;
uniform mat4 modelViewXf;
uniform mat3 normalXf;

uniform vec3 lightDirWorld;

in vec3 position;
in vec3 normal;
in vec2 uv;
in vec3 center;
in vec3 color;

out float outIntensity;
out vec2 outUV;
out vec3 outSurfaceColor;

void main()
{
    gl_Position = projectionXf * modelViewXf *
                    vec4((scale * position) + center, 1.0);

    outSurfaceColor = color;

    outIntensity = dot(normalize(normalXf * normal), -lightDirWorld);
    if(outIntensity < 0.0)
        outIntensity = 0.0;

    outUV = uv;
}
