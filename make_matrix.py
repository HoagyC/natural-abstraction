import math
import time

import numpy as np
import sympy as sym

from classes import *


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
            ("x1", ps1.x),
            ("y1", ps1.y),
            ("a1", ps1.angle),
            ("s1", ps1.speed),
            ("a2", ps2.angle),
            ("x2", ps2.x),
            ("y2", ps2.y),
            ("s2", ps2.speed),
            ("mass1", pp1.mass),
            ("mass2", pp2.mass),
            ("sp_l", sp.length),
            ("sp_s", sp.strength),
        ]

        bare_ins = [x[1] for x in in_vars]
        # init_t_ = time.time()
        d_matrix = [[y(*bare_ins) for y in x] for x in M]
        # print('inner spring eval', time.time() - init_t_)

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
                ("x1", ps1.x),
                ("y1", ps1.y),
                ("a1", ps1.angle),
                ("s1", ps1.speed),
                ("x2", ps2.x),
                ("y2", ps2.y),
                ("a2", ps2.angle),
                ("s2", ps2.speed),
                ("mass1", pp1.mass),
                ("mass2", pp2.mass),
                ("elasticity1", pp1.e),
                ("elasticity2", pp2.e),
                ("size1", pp1.size),
                ("size2", pp2.size),
            ]

            bare_ins = [x[1] for x in in_vars]

            d_matrix = [[y(*bare_ins) for y in x] for x in M]

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

        bare_ins = [x[1] for x in in_vars]
        d_matrix = [[y(*bare_ins) for y in x] for x in M]
        return np.array(d_matrix, dtype=float)

    particle_derivatives1 = [d_bounce, d_move, d_drag, d_accelerate]
    particle_derivatives2 = [d_collide]

    n_vars = 4

    size = n_vars * n_particles
    matrix = np.zeros((size, size))
    # going down the matrix changes the variable being differentiated
    # going across changes the variable we are differentiating with respect t

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
