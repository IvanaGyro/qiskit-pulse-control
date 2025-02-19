import numpy as np


def to_spherical(x, y, z):
    r = (x**2 + y**2 + z**2)**0.5
    theta = np.arctan2((x**2 + y**2)**0.5, z)
    phi = np.arctan2(y, x)
    return (r, theta, phi)


def to_spherical_degree(x, y, z):
    r, theta, phi = to_spherical(x, y, z)
    return (r, theta / np.pi * 180, phi / np.pi * 180)
