import numpy as np


def gaussian(duration, amplitude, sigma, angle=0.0):

    def generator(t):
        return np.exp(-(t - duration / 2)**2 / 2 / sigma**2)

    zero_level = generator(-1)

    def weighted_generator(t):
        return amplitude * np.exp(
            1.j * angle) * (generator(t) - zero_level) / (1. - zero_level)

    return weighted_generator


def drag(duration, amplitude, sigma, beta, angle=0.0):
    gaussian_waveform = gaussian(duration, amplitude, sigma, angle)

    def generator(t):
        return gaussian_waveform(t) * (1. - 1.j * beta *
                                       (t - duration / 2) / sigma**2)

    return generator
