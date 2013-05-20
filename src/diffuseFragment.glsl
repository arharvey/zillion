#version 150

in vec3 outSurfaceColor;
in float outIntensity;

out vec4 outColor;

void main ()
{
    outColor = vec4((outIntensity+0.1) * outSurfaceColor, 1.0);
}
