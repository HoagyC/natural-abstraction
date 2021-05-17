import math
import random
import time

import numpy as np

import diff
from classes import *


def add_vectors(angle1, length1, angle2, length2):
    """Returns the sum of two vectors"""

    x = math.sin(angle1) * length1 + math.sin(angle2) * length2
    y = math.cos(angle1) * length1 + math.cos(angle2) * length2

    angle = 0.5 * math.pi - math.atan2(y, x)
    length = math.hypot(x, y)

    return angle, length


def combine(p1, p2):
    if math.hypot(p1.x - p2.x, p1.y - p2.y) < p1.size + p2.size:
        total_mass = p1.mass + p2.mass
        p1.x = (p1.x * p1.mass + p2.x * p2.mass) / total_mass
        p1.y = (p1.y * p1.mass + p2.y * p2.mass) / total_mass
        (p1.angle, p1.speed) = add_vectors(
            p1.angle,
            p1.speed * p1.mass / total_mass,
            p2.angle,
            p2.speed * p2.mass / total_mass,
        )
        p1.speed *= p1.elasticity * p2.elasticity
        p1.mass += p2.mass
        p1.collide_with = p2


def collide(p1, p2):
    """Tests whether two particles overlap
    If they do, make them bounce, i.e. update their angle, speed and position"""

    dx = p1.x - p2.x
    dy = p1.y - p2.y

    dist = math.hypot(dx, dy)
    if dist < p1.size + p2.size:
        angle = math.atan2(dy, dx) + 0.5 * math.pi
        total_mass = p1.mass + p2.mass

        (p1.angle, p1.speed) = add_vectors(
            p1.angle,
            p1.speed * (p1.mass - p2.mass) / total_mass,
            angle,
            2 * p2.speed * p2.mass / total_mass,
        )
        (p2.angle, p2.speed) = add_vectors(
            p2.angle,
            p2.speed * (p2.mass - p1.mass) / total_mass,
            angle + math.pi,
            2 * p1.speed * p1.mass / total_mass,
        )
        elasticity = p1.elasticity * p2.elasticity
        p1.speed *= elasticity
        p2.speed *= elasticity

        overlap = 0.5 * (p1.size + p2.size - dist + 1)
        p1.x += math.sin(angle) * overlap
        p1.y -= math.cos(angle) * overlap
        p2.x -= math.sin(angle) * overlap
        p2.y += math.cos(angle) * overlap


class Particle:
    """A circular object with a velocity, size and mass"""

    def __init__(self, x, y, size, ident, mass=1):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (0, 0, 255)
        self.thickness = 0
        self.speed = 0
        self.angle = 0
        self.mass = mass
        self.drag = 1
        self.elasticity = 0.9
        self.ident = ident

    def move(self):
        """Update position based on speed, angle"""

        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed

        # d(p.x)/d(p.speed) = math.sin(self.angle)
        # d(p.x)/d(p.angle) = math.cos(self.angle) * p.speed
        # d(p.x)/d(p.x) = p.x
        # d(p.y)/d(p.speed) = -math.cos(self.angle)
        # d(p.y)/d(p.angle) = math.sin(self.angle) * p.speed
        # d(p.y)/d(p.y) = p.y

        """
        with x, y, angle, speed we have:
        [[1, 0, math.cos(p.angle)*p.speed, math.sin(angle)],
         [0, 1, math.sin(p.angle)*p.speed, -math.cos(angle)],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
        """

    def experience_drag(self):
        self.speed *= self.drag

    def mouse_move(self, x, y):
        """Change angle and speed to move towards a given point"""

        dx = x - self.x
        dy = y - self.y
        self.angle = 0.5 * math.pi + math.atan2(dy, dx)
        self.speed = math.hypot(dx, dy) * 0.1

    def accelerate(self, vector):
        """Change angle and speed by a given vector"""
        (self.angle, self.speed) = add_vectors(self.angle, self.speed, *vector)

    def attract(self, other):
        """ " Change velocity based on gravitational attraction between two particle"""

        dx = self.x - other.x
        dy = self.y - other.y
        dist = math.hypot(dx, dy)

        if dist < self.size + other.size:
            return True

        theta = math.atan2(dy, dx)
        force = 0.1 * self.mass * other.mass / dist ** 2
        self.accelerate((theta - 0.5 * math.pi, force / self.mass))
        other.accelerate((theta + 0.5 * math.pi, force / other.mass))


class Spring:
    def __init__(self, p1, p2, length=50, strength=0.5):
        self.p1 = p1
        self.p2 = p2
        self.length = length
        self.strength = strength

    def update(self):
        dx = self.p1.x - self.p2.x
        dy = self.p1.y - self.p2.y
        dist = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        force = (self.length - dist) * self.strength

        self.p1.accelerate((theta + 0.5 * math.pi, force / self.p1.mass))
        self.p2.accelerate((theta - 0.5 * math.pi, force / self.p2.mass))


def make_matrix(fs: FrameState, es: EnvState, n_particles: int, diff_matrices):
    """
    List of functions of which to take derivative
    'move': (1, lambda p: p.move()),
    'drag': (1, lambda p: p.experienceDrag()),
    'bounce': (1, lambda p: self.bounce(p)),
    'accelerate': (1, lambda p: p.accelerate(self.acceleration)),
    'collide': (2, lambda p1, p2: collide(p1, p2)),
    'combine': (2, lambda p1, p2: combine(p1, p2)),
    'attract': (2, lambda p1, p2: p1.attract(p2))

    function code to be differentiated

    for i, particle in enumerate(self.particles, 1):
        for f in self.particle_functions1:
            f(particle)
        for particle2 in self.particles[i:]:
            for f in self.particle_functions2:
                f(particle, particle2)
    """

    def d_bounce(ps: ParticleState, pp: ParticleParams, ep: EnvParams):
        d_matrix = np.zeros((4, 4))

        if ps.x > ep.width - pp.size or ps.x < pp.size:
            d_matrix += np.array(
                [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, ep.e]]
            )

        if ps.y > ep.height - pp.size or ps.y < pp.size:
            d_matrix += np.array(
                [[0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, ep.e]]
            )

        return d_matrix

    def d_move(ps: ParticleState, _: ParticleParams, __: EnvParams):
        d_matrix = np.zeros((4, 4))

        d_matrix += [
            [1, 0, math.cos(ps.angle) * ps.speed, math.sin(ps.angle)],
            [0, 1, math.sin(ps.angle) * ps.angle, -math.cos(ps.angle)],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        return d_matrix

    def d_drag(_: ParticleState, pp: ParticleParams, __: EnvParams):
        d_matrix = np.zeros((4, 4))

        d_matrix += np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, pp.drag]]
        )

        return d_matrix

    def d_spring(
        ps1: ParticleState,
        ps2: ParticleState,
        pp1: ParticleParams,
        pp2: ParticleParams,
        sp: SpringParams,
    ):

        M = diff_matrices['spring']

        in_vars = [
            ("a1", ps1.angle),
            ("s1", ps1.speed),
            ("x1", ps1.x),
            ("y1", ps1.y),
            ("a2", ps2.angle),
            ("s2", ps2.speed),
            ("x2", ps2.x),
            ("y2", ps2.y),
            ("mass1", pp1.mass),
            ("mass2", pp2.mass),
            ("sp_l", sp.length),
            ("sp_s", sp.strength),
        ]

        d_matrix = [[y.subs(in_vars).evalf() for y in x] for x in M]

        return np.array(d_matrix, dtype=float)

    def d_collide(
        ps1: ParticleState,
        ps2: ParticleState,
        pp1: ParticleParams,
        pp2: ParticleParams,
        _: EnvParams,
    ):
        dx = ps1.x - ps2.x
        dy = ps1.y - ps2.y

        dist = math.hypot(dx, dy)

        if dist < pp1.size + pp2.size:
            M = diff_matrices['collide']
            in_vars = [
                ("a1", ps1.angle),
                ("s1", ps1.speed),
                ("x1", ps1.x),
                ("y1", ps1.y),
                ("a2", ps2.angle),
                ("s2", ps2.speed),
                ("x2", ps2.x),
                ("y2", ps2.y),
                ("mass1", pp1.mass),
                ("mass2", pp2.mass),
                ("elasticity1", pp1.e),
                ("elasticity2", pp2.e),
                ("size1", pp1.size),
                ("size2", pp2.size),
            ]

            d_matrix = [[y.subs(in_vars).evalf() for y in x] for x in M]

        else:
            d_matrix = np.zeros((8, 8))

        return np.array(d_matrix, dtype=float)

    def d_accelerate(ps: ParticleState, pp: ParticleParams, ep: EnvParams):
        M = diff_matrices['accelerate']
        in_vars = [("x", ps.x),
                   ("y", ps.y),
                   ("a", ps.angle),
                   ("s", ps.speed),
                   ("a_a", ep.acc_angle),
                   ("a_s", ep.acc_speed)
                   ]

        d_matrix = [[y.subs(in_vars).evalf() for y in x] for x in M]

        return np.array(d_matrix, dtype=float)

    particle_derivatives1 = [d_bounce, d_move, d_drag, d_accelerate]
    particle_derivatives2 = [d_collide]

    n_vars = 4

    size = n_vars * n_particles
    matrix = np.zeros((size, size))
    # going down the matrix changes the variable being differentiated
    # going across changes the variable we are differentiating with respect to

    for i, ps1 in enumerate(fs.states):
        for f in particle_derivatives1:
            add_matrix = f(ps1, es.ps[i], es.env)
            matrix[i * n_vars:(i + 1) * n_vars, i * n_vars:(i + 1) * n_vars] += add_matrix

        for f in particle_derivatives2:
            for j, ps2 in enumerate(fs.states[i + 1:]):
                add_matrix = f(ps1, ps2, es.ps[i], es.ps[j], es.env)
                matrix[
                    i * n_vars:(i + 1) * n_vars, i * n_vars: (i + 1) * n_vars
                ] += add_matrix[:n_vars, :n_vars]
                matrix[
                    i * n_vars:(i + 1) * n_vars, j * n_vars: (j + 1) * n_vars
                ] += add_matrix[:n_vars, n_vars:]
                matrix[
                    j * n_vars:(j + 1) * n_vars, i * n_vars: (i + 1) * n_vars
                ] += add_matrix[n_vars:, :n_vars]
                matrix[
                    j * n_vars:(j + 1) * n_vars, j * n_vars: (j + 1) * n_vars
                ] += add_matrix[n_vars:, n_vars:]

    for spring in es.springs:
        i = spring.p1.ident
        j = spring.p2.ident
        add_matrix = d_spring(fs.states[i], fs.states[j], es.ps[i], es.ps[j], spring)
        matrix[
            i * n_vars:(i + 1) * n_vars, i * n_vars: (i + 1) * n_vars
        ] += add_matrix[:n_vars, :n_vars]
        matrix[
            i * n_vars:(i + 1) * n_vars, j * n_vars: (j + 1) * n_vars
        ] += add_matrix[:n_vars, n_vars:]
        matrix[
            j * n_vars:(j + 1) * n_vars, i * n_vars: (i + 1) * n_vars
        ] += add_matrix[n_vars:, :n_vars]
        matrix[
            j * n_vars:(j + 1) * n_vars, j * n_vars: (j + 1) * n_vars
        ] += add_matrix[n_vars:, n_vars:]

    return matrix


class Environment:
    """Defines the boundary of a simulation and its properties"""

    def __init__(self, width, height):
        self.frame_n = 0
        self.saved_states = []
        self.width = width
        self.height = height
        self.particles = []
        self.springs = []

        self.colour = (255, 255, 255)
        self.mass_of_air = 0.2
        self.elasticity = 0.75
        self.acceleration = (0, 0)

        self.particle_functions1 = []
        self.particle_functions2 = []
        self.function_dict = {
            "move": (1, lambda p: p.move()),
            "drag": (1, lambda p: p.experience_drag()),
            "bounce": (1, lambda p: self.bounce(p)),
            "accelerate": (1, lambda p: p.accelerate(self.acceleration)),
            "collide": (2, lambda p1, p2: collide(p1, p2)),
            "combine": (2, lambda p1, p2: combine(p1, p2)),
            "attract": (2, lambda p1, p2: p1.attract(p2)),
        }

    def save_frame(self):
        frame_state = FrameState(
            [ParticleState(p.x, p.y, p.speed, p.angle) for p in self.particles]
        )
        self.saved_states.append(frame_state)

    def save_env_state(self):
        env_params = EnvParams(self.width, self.height, self.elasticity, self.acceleration[0], self.acceleration[1])
        particles = []
        for p in self.particles:
            particles.append(ParticleParams(p.size, p.mass, p.elasticity, p.drag))

        springs = []
        for s in self.springs:
            springs.append(SpringParams(s.p1, s.p2, s.length, s.strength))
        env = EnvState(env_params, springs, particles)
        return env

    def add_functions(self, function_list):
        for func in function_list:
            (n, f) = self.function_dict.get(func, (-1, None))
            if n == 1:
                self.particle_functions1.append(f)
            elif n == 2:
                self.particle_functions2.append(f)
            else:
                print("No such function: %s" % f)

    def add_particles(self, n=1, **kwargs):
        """Add n particles with properties given by keyword arguments"""

        for i in range(n):
            size = kwargs.get("size", random.randint(10, 20))
            mass = kwargs.get("mass", random.randint(100, 10000))
            x = kwargs.get("x", random.uniform(size, self.width - size))
            y = kwargs.get("y", random.uniform(size, self.height - size))

            particle = Particle(x, y, size, len(self.particles), mass)
            particle.speed = kwargs.get("speed", random.random())
            particle.angle = kwargs.get("angle", random.uniform(0, math.pi * 2))
            particle.colour = kwargs.get("colour", (0, 0, 255))
            particle.elasticity = kwargs.get("elasticity", self.elasticity)
            particle.drag = (
                particle.mass / (particle.mass + self.mass_of_air)
            ) ** particle.size

            self.particles.append(particle)

    def add_spring(self, p1, p2, length=50, strength=0.5):
        """Add a spring between particles p1 and p2"""
        self.springs.append(
            Spring(self.particles[p1], self.particles[p2], length, strength)
        )

    def update(self):
        """Moves particles and tests for collisions with the walls and each other"""

        for i, particle in enumerate(self.particles, 1):
            for f in self.particle_functions1:
                f(particle)
            for particle2 in self.particles[i:]:
                for f in self.particle_functions2:
                    f(particle, particle2)

        for spring in self.springs:
            spring.update()

        self.save_frame()
        self.frame_n += 1

    def bounce(self, particle):
        """Tests whether a particle has hit the boundary of the environment"""

        if particle.x > self.width - particle.size:
            particle.x = 2 * (self.width - particle.size) - particle.x
            particle.angle = -particle.angle
            particle.speed *= self.elasticity

            # d(p.x)/d(p.x) = -1
            # d(p.angle)/d(p.angle) = -1
            # d(p.speed)/d(p.speed) = self.elasticity

        elif particle.x < particle.size:
            particle.x = 2 * particle.size - particle.x
            particle.angle = -particle.angle
            particle.speed *= self.elasticity

            # d(p.x)/d(p.x) = -1
            # d(p.angle)/d(p.angle) = -1
            # d(p.speed)/d(p.speed) = self.elasticity

        if particle.y > self.height - particle.size:
            particle.y = 2 * (self.height - particle.size) - particle.y
            particle.angle = math.pi - particle.angle
            particle.speed *= self.elasticity

            # d(p.y)/d(p.y) = -1
            # d(p.angle)/d(p.angle) = -1
            # d(p.speed)/d(p.speed) = self.elasticity

        elif particle.y < particle.size:
            particle.y = 2 * particle.size - particle.y
            particle.angle = math.pi - particle.angle
            particle.speed *= self.elasticity

            # d(p.y)/d(p.y) = -1
            # d(p.angle)/d(p.angle) = -1
            # d(p.speed)/d(p.speed) = self.elasticity

        """
        differentiating this function:
        inputs are p.x, p.y, p.angle, p.speed
        outputs are p.x, p.y, p.angle, p.speed
        
        constants are p.size, elasticity
        
        so at the y extreme we have:
        [0, 0, 0, 0,
         0, -1, 0, 0,
         0, 0, -1, 0,
         0, 0, 0, self.elasticity]
         
        at the x extreme we have
        [0, 0, 0, 0,
         0, -1, 0, 0,
         0, 0, -1, 0,
         0, 0, 0, self.elasticity]
         
        and otherwise
        [0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0]
        
        """

    def find_particle(self, x, y):
        """Returns any particle that occupies position x, y"""

        for particle in self.particles:
            if math.hypot(particle.x - x, particle.y - y) <= particle.size:
                return particle
        return None

    def backpropagate(self):
        initial_time = time.time()
        finals = self.saved_states[-1]
        env_state = self.save_env_state()
        diff_matrices = {'spring': diff.auto_diff_spring(),
                         'accelerate': diff.auto_diff_accelerate(),
                         'collide': diff.auto_diff_collide()}

        matrices = []
        for i, state in enumerate(self.saved_states[:5]):
            frame_matrix = make_matrix(state, env_state, len(self.particles), diff_matrices)
            print(frame_matrix, '\n', f"{i} / {len(self.saved_states) - 1}", (time.time() - initial_time) / (i + 1))
            matrices.append(frame_matrix)

        vars = np.array(self.saved_states[0].flatten())
        for m in matrices:
            vars = vars * m

        print(vars)