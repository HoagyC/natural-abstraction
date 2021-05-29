import math
import pygame
import particles


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
    #
    # final_states = [(p.x, p.y, p.speed, p.angle) for p in universe.particles]
    # print(final_states)

    universe.backpropagate()


def add_box(env, width=5, height=5, start_x=10, start_y=10, spacing=15):
    for y in range(height):
        for x in range(width):
            x_pos = start_x + x * spacing
            y_pos = start_y + y * spacing
            env.add_particle(
                x=x_pos, y=y_pos, mass=5, size=2, speed=2, elasticity=0.8, colour=(20, 40, 200)
            )
            curr_id = len(env.particles) - 1
            if x > 0:
                env.add_spring(curr_id, curr_id - 1, length=spacing)
            if y > 0:
                env.add_spring(curr_id, curr_id - width, length=spacing)
            if y > 0 and x > 0:
                env.add_spring(curr_id, curr_id - width - 1, length=spacing*1.414)
            if y > 0 and x < width - 1:
                env.add_spring(curr_id, curr_id - width + 1, length=spacing*1.414)


def add_wheel(env, radius=30, spacing=4, center_x=200, center_y=300):
    circumference = radius * math.pi * 2
    total_n = int(math.ceil(circumference / spacing))

    initial_id = len(universe.particles)
    env.add_particle(x=center_x, y=center_y, mass=10, size=5, elasticity=0.8)

    for i in range(total_n):
        x_pos = center_x + math.sin(i * 2 * math.pi / total_n)
        y_pos = center_y + math.cos(i * 2 * math.pi / total_n)
        env.add_particle(x=x_pos, y=y_pos, mass=5, size=2, elasticity=0.8)

        curr_id = len(env.particles) - 1
        if i > 0:
            env.add_spring(curr_id, curr_id - 1, length=spacing)
        if i == total_n - 1:
            env.add_spring(curr_id, initial_id + 1, length=spacing)

        env.add_spring(curr_id, initial_id, length=radius * 2, strength=1)




(width, height) = (400, 400)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Spring sim")

universe = particles.Environment(width, height)
universe.colour = (255, 255, 255)
universe.add_functions(["move", "bounce", "collide", "drag", "accelerate"])
universe.acceleration = (math.pi, 0.01)
universe.mass_of_air = 0.02

add_box(universe, width=5, height=5)
add_wheel(universe, radius=20, center_x=50, center_y=200)



if __name__ == "__main__":
    main()
