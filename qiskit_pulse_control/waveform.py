from typing import Callable

import numpy as np
import sympy


def gaussian(
    duration: float | sympy.Symbol,
    amplitude: float | sympy.Symbol,
    sigma: float | sympy.Symbol,
    angle: float | sympy.Symbol = 0.0
) -> Callable[[float | sympy.Symbol], float | sympy.Expr]:
    if isinstance(angle, sympy.Symbol):
        phase = sympy.exp(sympy.I * angle)
    else:
        phase = np.exp(1.j * angle)

    def generator(t: float | sympy.Symbol) -> float | sympy.Expr:
        if any(isinstance(v, sympy.Symbol) for v in (t, duration, sigma)):
            return sympy.exp(-(t - duration / 2)**2 / 2 / sigma**2)
        return np.exp(-(t - duration / 2)**2 / 2 / sigma**2)

    zero_level = generator(-1)

    def weighted_generator(t: float | sympy.Symbol) -> float | sympy.Expr:
        return amplitude * phase * (generator(t) - zero_level) / (1. -
                                                                  zero_level)

    return weighted_generator


def drag(
    duration: float | sympy.Symbol,
    amplitude: float | sympy.Symbol,
    sigma: float | sympy.Symbol,
    beta: float | sympy.Symbol,
    angle: float | sympy.Symbol = 0.0
) -> Callable[[float | sympy.Symbol], float | sympy.Expr]:
    gaussian_waveform = gaussian(duration, amplitude, sigma, angle)

    def generator(t):
        return gaussian_waveform(t) * (1. - 1.j * beta *
                                       (t - duration / 2) / sigma**2)

    return generator
