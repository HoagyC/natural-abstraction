import sympy as sym


def add_vectors(angle1, length1, angle2, length2):
    """Returns the sum of two vectors"""

    x = sym.sin(angle1) * length1 + sym.sin(angle2) * length2
    y = sym.cos(angle1) * length1 + sym.cos(angle2) * length2

    angle = 0.5 * sym.pi - sym.atan2(y, x)
    length = sym.atan2(y, x) + 0.5 * sym.pi

    return angle, length


def auto_diff_spring():
    x1 = sym.Symbol("x1")
    x2 = sym.Symbol("x2")
    y1 = sym.Symbol("y1")
    y2 = sym.Symbol("y2")
    a1 = sym.Symbol("a1")
    a2 = sym.Symbol("a2")
    s1 = sym.Symbol("s1")
    s2 = sym.Symbol("s2")

    x1n = sym.Symbol("x1n")
    x2n = sym.Symbol("x2n")
    y1n = sym.Symbol("y1n")
    y2n = sym.Symbol("y2n")
    # a1n = sym.Symbol("a1n")
    # a2n = sym.Symbol("a2n")
    # s1n = sym.Symbol("s1n")
    # s2n = sym.Symbol("s2n")

    sp_l = sym.Symbol("sp_l")
    sp_s = sym.Symbol("sp_s")

    mass1 = sym.Symbol("mass1")
    mass2 = sym.Symbol("mass2")
    # e1 = sym.Symbol("elasticity1")
    # e2 = sym.Symbol("elasticity2")
    # size1 = sym.Symbol("size1")
    # size2 = sym.Symbol("size2")

    dx = x1 - x2
    dy = y1 - y2
    dist = (dx ** 2 + dy ** 2) ** 0.5
    theta = sym.atan2(dy, dx)

    force = (sp_l - dist) * sp_s

    a1n, s1n = add_vectors(a1, s1, theta + 0.5 * sym.pi, force / mass1)
    a2n, s2n = add_vectors(a2, s2, theta - 0.5 * sym.pi, force / mass2)

    in_vars = [x1, y1, a1, s1, x2, y2, a2, s2]
    out_vars = [x1n, y1n, a1n, s1n, x2n, y2n, a2n, s2n]

    M = [[sym.diff(out_v, in_v) for in_v in in_vars] for out_v in out_vars]

    return M


def auto_diff_accelerate():
    x = sym.Symbol("x")
    y = sym.Symbol("y")
    a = sym.Symbol("a")
    s = sym.Symbol("s")
    a_a = sym.Symbol("a_a")
    s_s = sym.Symbol("a_s")

    x_n = sym.Symbol("x_n")
    y_n = sym.Symbol("y_n")
    # a_n = sym.Symbol("a_n")
    # s_n = sym.Symbol("s_n")

    a_n, s_n = add_vectors(a, s, a_a, s_s)

    in_vars = [x, y, a, s]
    out_vars = [x_n, y_n, a_n, s_n]

    M = [[sym.diff(out_v, in_v) for in_v in in_vars] for out_v in out_vars]

    return M


def auto_diff_collide():
    x1 = sym.Symbol("x1")
    x2 = sym.Symbol("x2")
    y1 = sym.Symbol("y1")
    y2 = sym.Symbol("y2")
    a1 = sym.Symbol("a1")
    a2 = sym.Symbol("a2")
    s1 = sym.Symbol("s1")
    s2 = sym.Symbol("s2")

    mass1 = sym.Symbol("mass1")
    mass2 = sym.Symbol("mass2")
    e1 = sym.Symbol("elasticity1")
    e2 = sym.Symbol("elasticity2")
    size1 = sym.Symbol("size1")
    size2 = sym.Symbol("size2")

    dx = x1 - x2
    dy = y1 - y2
    dist = (dx ** 2 + dy ** 2) ** 0.5
    angle = sym.atan2(dy, dx) + 0.5 * sym.pi

    total_mass = mass1 + mass2

    a1n, s1n = add_vectors(
        a1, s1 * (mass1 - mass2) / total_mass, angle, 2 * s2 * mass2 / total_mass
    )
    a2n, s2n = add_vectors(
        a2,
        s2 * (mass2 - mass1) / total_mass,
        angle + sym.pi,
        2 * s1 * mass1 / total_mass,
    )
    elasticity = e1 * e2
    s1n *= elasticity
    s2n *= elasticity

    overlap = 0.5 * (size1 + size2 - dist + 1)

    x1n = x1 + sym.sin(angle) * overlap
    y1n = y1 - sym.cos(angle) * overlap
    x2n = x2 - sym.sin(angle) * overlap
    y2n = y2 + sym.cos(angle) * overlap

    in_vars = [x1, y1, a1, s1, x2, y2, a2, s2]
    out_vars = [x1n, y1n, a1n, s1n, x2n, y2n, a2n, s2n]

    M = [[sym.diff(out_v, in_v) for in_v in in_vars] for out_v in out_vars]

    return M


if __name__ == "__main__":
    # auto_diff_collide()
    print(auto_diff_accelerate())
