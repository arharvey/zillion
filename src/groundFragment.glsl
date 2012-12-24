#version 120

uniform vec3 surfaceColor;
uniform sampler2D tex;

varying float outIntensity;
varying vec2 outUV;

void main ()
{
    vec3 uvColor = vec3(outUV.x - int(outUV.x), outUV.y - int(outUV.y), 0);
    if(uvColor.x < 0.0)
        uvColor.x = 1.0 + uvColor.x;

    if(uvColor.y < 0.0)
        uvColor.y = 1.0 + uvColor.y;

    gl_FragColor = vec4(outIntensity * texture2D(tex, uvColor.st).rgb, 1);
}
