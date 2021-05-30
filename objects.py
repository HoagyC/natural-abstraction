import math


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


def add_wedge(env, width=5, height=5, start_x=10, start_y=10, spacing=15, slope=0.5):
    for y in range(height):
        for x in range(width):
            x_pos = start_x + x * spacing
            y_pos = start_y + y * spacing * (1 - (slope * x / (width - 1)))
            env.add_particle(
                x=x_pos, y=y_pos, mass=5, size=2, speed=2, elasticity=0.8, colour=(20, 40, 200)
            )
            curr_id = len(env.particles) - 1

            if x > 0:
                env.add_spring(curr_id, curr_id - 1, length=spacing)
            if y > 0:
                y_gap = spacing * (1 - slope * x / (width - 1))
                y_gap_left = y_gap - spacing * (slope / (width - 1))
                y_gap_right = y_gap + spacing * (slope / (width - 1))
                left_diag = math.hypot(y_gap_left, spacing)
                right_diag = math.hypot(y_gap_right, spacing)
                env.add_spring(curr_id, curr_id - width, length=y_gap)
                if x > 0:
                    env.add_spring(curr_id, curr_id - width - 1, length=left_diag)
                if x < width - 1:
                    env.add_spring(curr_id, curr_id - width + 1, length=right_diag)


def add_wheel(env, radius=30, spacing=6, center_x=200, center_y=300):
    circumference = radius * math.pi * 2
    total_n = int(math.ceil(circumference / spacing))

    initial_id = len(env.particles)
    env.add_particle(x=center_x, y=center_y, mass=10, size=5, elasticity=0.8)

    for i in range(total_n):
        x_pos = center_x + math.sin(i * 2 * math.pi / total_n) * radius
        y_pos = center_y + math.cos(i * 2 * math.pi / total_n) * radius
        env.add_particle(x=x_pos, y=y_pos, mass=5, size=2, elasticity=0.8)

        curr_id = len(env.particles) - 1
        if i > 0:
            env.add_spring(curr_id, curr_id - 1, length=spacing)
        if i == total_n - 1:
            env.add_spring(curr_id, initial_id + 1, length=spacing)

        env.add_spring(curr_id, initial_id, length=radius, strength=1)
