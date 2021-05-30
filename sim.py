import math
import time

import pygame
import particles
from objects import *


def main():
    initial_states = [(p.x, p.y, p.speed, p.angle) for p in universe.particles]
    print(initial_states)

    paused = False
    running = True
    selected_particle = None
    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = (True, False)[paused]
            elif event.type == pygame.MOUSEBUTTONDOWN:
                selected_particle = universe.find_particle(*pygame.mouse.get_pos())

            elif event.type == pygame.MOUSEBUTTONUP:
                selected_particle = None

        if not paused:
            universe.update()

        if selected_particle:
            selected_particle.mouse_move(*pygame.mouse.get_pos())

        screen.fill(universe.colour)

        for p in universe.particles:
            pygame.draw.circle(screen, p.colour, (int(p.x), int(p.y)), p.size, 0)

        for s in universe.springs:
            pygame.draw.aaline(
                screen,
                (0, 0, 0),
                (int(s.p1.x), int(s.p1.y)),
                (int(s.p2.x), int(s.p2.y)),
            )

        speeds = [abs(p.speed)for p in universe.particles]
        print(max(speeds))
        if max(speeds) < 0.1 and frame_count > 100:
            running = False

        pygame.display.flip()
        frame_count += 1
        time.sleep(1)

    # final_states = [(p.x, p.y, p.speed, p.angle) for p in universe.particles]
    # print(final_states)

    universe.backpropagate()


(width, height) = (400, 400)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Spring sim")

universe = particles.Environment(width, height)
universe.colour = (255, 255, 255)
universe.add_functions(["move", "bounce", "collide", "drag", "accelerate"])
universe.acceleration = (math.pi, 0.01)
universe.mass_of_air = 0.02

# add_box(universe, width=5, height=5)
# add_wedge(universe, width=10, height=10, spacing=20, start_x=200, start_y=200, slope=0.8)
add_wheel(universe, radius=20, center_x=150, center_y=200)
add_wheel(universe, radius=20, center_x=180, center_y=350)
add_wheel(universe, radius=20, center_x=250, center_y=200)
add_wheel(universe, radius=20, center_x=150, center_y=300)



if __name__ == "__main__":
    main()
