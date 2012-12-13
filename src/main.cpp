#include <iostream>

#include <GL/glew.h>

#include "SDL.h"

int
main(int argc, char* argv[])
{
	SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO);

	SDL_Surface* pSurface = SDL_SetVideoMode(
								800, 600, 32,
								SDL_HWSURFACE | SDL_DOUBLEBUF | SDL_OPENGL);
	SDL_WM_SetCaption("Zillion", 0);

	glewExperimental = GL_TRUE;
	glewInit();

	GLuint vertexBuffer;
	glGenBuffers(1, &vertexBuffer);

	std::cout << vertexBuffer << std::endl;

	SDL_Event windowEvent;
	while(true)
	{
		if(SDL_PollEvent(&windowEvent))
		{
			if(windowEvent.type == SDL_QUIT)
				break;

			if(windowEvent.type == SDL_KEYUP &&
				windowEvent.key.keysym.sym == SDLK_ESCAPE)
				break;
		}

		SDL_GL_SwapBuffers();
	}

	SDL_Quit();

	return 0;
}
