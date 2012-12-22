#version 120

uniform mat4 projectionXf;
uniform mat4 modelViewXf;
uniform mat3 normalXf;

uniform vec3 lightDirWorld;

attribute vec3 pt;

attribute vec3 position;
attribute vec3 normal;

varying float outIntensity;

const float scale = 0.04;

void main ()
{
    gl_Position = projectionXf * modelViewXf *
                    vec4((scale * position) + pt, 1.0);

    outIntensity = dot(normalize(normalXf * normal), -lightDirWorld);
    if(outIntensity < 0.0)
        outIntensity = 0.0;
}
