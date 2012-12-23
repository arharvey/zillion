#version 120

uniform vec3 surfaceColor;

varying float outIntensity;
varying vec2 outUV;

void main ()
{
    vec3 uvColor = vec3(outUV.x - int(outUV.x), outUV.y - int(outUV.y), 0);
    if(uvColor.x < 0.0)
        uvColor.x = 1.0 + uvColor.x;

    if(uvColor.y < 0.0)
        uvColor.y = 1.0 + uvColor.y;

    float value = uvColor.x > uvColor.y ? uvColor.x : uvColor.y;
    gl_FragColor = vec4(outIntensity*value*surfaceColor, 1.0);
}
