#version 120

varying float outIntensity;

vec3 surfaceColor = vec3(0.8, 0.8, 1.0);

void main ()
{
    gl_FragColor = vec4(outIntensity*surfaceColor, 1.0);
}
