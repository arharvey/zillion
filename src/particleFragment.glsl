#version 150

uniform vec3 surfaceColor0 = vec3(0.9, 0.9, 1.0);
uniform vec3 surfaceColor1 = vec3(0.7, 0.7, 1.0);

in float outIntensity;
in vec2 outUV;

out vec4 outColor;

void main ()
{
    vec3 color = outUV.t < 0.5 ? surfaceColor0 : surfaceColor1;

    outColor = vec4(outIntensity * color, 1.0);
}
