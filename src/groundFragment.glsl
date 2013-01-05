#version 150

uniform vec3 surfaceColor;
uniform sampler2D tex;

in float outIntensity;
in vec2 outUV;

out vec4 outColor;

void main ()
{
    vec3 uvColor = vec3(outUV.x - int(outUV.x), outUV.y - int(outUV.y), 0);
    if(uvColor.x < 0.0)
        uvColor.x = 1.0 + uvColor.x;

    if(uvColor.y < 0.0)
        uvColor.y = 1.0 + uvColor.y;

    outColor = vec4(outIntensity * texture(tex, uvColor.st).rgb, 1);
}
