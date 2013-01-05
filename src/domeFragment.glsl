#version 150

uniform vec3 color0;
uniform vec3 color1;
uniform vec3 color2;

in vec2 outUV;

out vec4 outColor;

void main ()
{
    vec3 c = outUV.t < 0.5 ? mix(color0, color1, outUV.t/0.5) :
                                mix(color1, color2, (outUV.t-0.5)/0.5);

    outColor = vec4(c, 1.0);
}
