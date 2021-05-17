from typing import List, NamedTuple


class ParticleState(NamedTuple):
    x: float
    y: float
    angle: float
    speed: float


class SpringParams(NamedTuple):
    p1: int
    p2: int
    length: float
    strength: float


class FrameState(NamedTuple):
    states: List[ParticleState] = []

    def flatten(self):
        flat_list = []
        for p in self.states:
            flat_list += [p.x, p.y, p.angle, p.speed]
        return flat_list


class ParticleParams(NamedTuple):
    size: float
    mass: float
    e: float
    drag: float


class EnvParams(NamedTuple):
    width: int
    height: int
    e: float
    acc_angle: float
    acc_speed: float


class EnvState(NamedTuple):
    env: EnvParams
    springs: List[SpringParams]
    ps: List[ParticleParams] = []
