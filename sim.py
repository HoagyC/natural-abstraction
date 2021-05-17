import math
import pygame
import particles


def main():
    initial_states = [(p.x, p.y, p.speed, p.angle) for p in universe.particles]
    print(initial_states)

    paused = False
    running = True
    selected_particle = None

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

        if all([abs(p.speed) < 0.01 for p in universe.particles]):
            running = False

        pygame.display.flip()
    #
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

for _ in range(4):
    universe.add_particles(
        mass=100, size=10, speed=2, elasticity=1, colour=(20, 40, 200)
    )

universe.add_spring(0, 1, length=100, strength=0.5)
universe.add_spring(1, 2, length=100, strength=0.1)
universe.add_spring(0, 2, length=100, strength=0.05)

if __name__ == "__main__":
    main()
