import qutip

import numpy as np
import quaternion
import matplotlib.pyplot as plt
import math

theta = math.pi / 6
rotate_pi_over_6 = np.quaternion(np.cos(theta / 2), np.sin(theta / 2), 0, 0)
# print(rotate_pi_over_6 * rotate_pi_over_6)

# This is the constant depending on the design of the physical device.
# The Rx(\theta) operation is equivalent to
# $e^{i \times OMEGA \times \frac{\theta}{2} \sigma_x}$
OMEGA = 0.1185  # value from ibm_sherbrooke's zeroth qubit
OMEGA = 0.1238  # value from ibm_sherbrooke's zeroth qubit

# Define the Pauli matrices
sigma_x = qutip.sigmax()
sigma_y = qutip.sigmay()

#                        duration=256,
#                        amplitude=0.2002363461992037,
#                        sigma=64,

# Time parameters
t_start = 0.0  # Start time
t_end = 256.0  # End time
# t_end = math.ceil(52.8 * 6)

frequency_shift = 2*1e-4
frequency_shift = 0

def to_spherical(x, y, z):
    r = (x**2 + y**2 + z**2)**0.5
    theta = np.arctan2((x**2 + y**2)**0.5, z)
    phi = np.arctan2(y, x)
    return (r, theta, phi)

def to_spherical_degree(x, y ,z):
    r, theta, phi = to_spherical(x, y, z)
    return (r, theta / math.pi * 180, phi / math.pi * 180)




def quaternion_shifted_h(t):
    sigma = 52.8
    amplitude = 0.2
    duration = t_end - t_start
    gaussian_waveform = gaussian(duration, amplitude, sigma)
    rotation_angle = gaussian_waveform(t) * OMEGA
    return np.quaternion(
        np.cos(rotation_angle / 2),
        np.sin(rotation_angle / 2) * np.cos(frequency_shift * t),
        np.sin(rotation_angle / 2) * np.sin(frequency_shift * t), 0)

def quaternion_not_shifted_h(t):
    sigma = 52.8
    amplitude = 0.2
    duration = t_end - t_start
    gaussian_waveform = gaussian(duration, amplitude, sigma)
    theta = gaussian_waveform(t) * OMEGA
    return np.quaternion(np.cos(theta / 2), np.sin(theta / 2), 0, 0) 


def quaternion_h(t):
    sigma = 64
    duration = t_end - t_start
    amplitude = 0.2002363461992037
    gaussian_waveform = gaussian(duration, amplitude, sigma)
    theta = gaussian_waveform(t) * OMEGA
    return np.quaternion(np.cos(theta / 2), np.sin(theta / 2), 0, 0)


# Define the Hamiltonian as a function of time
def H(t):
    sigma = 64
    duration = t_end - t_start
    amplitude = 0.2002363461992037
    # sigma = 32
    # duration = 32 * 6
    # amplitude = 1
    gaussian_waveform = gaussian(duration, amplitude, sigma)
    return gaussian_waveform(t) * OMEGA / 2 * sigma_x


def gaussian(duration, amplitude, sigma, angle=0.0):
 
    def generator(t):
        return np.exp(-(t - duration / 2)**2 / 2 / sigma**2)
    
    zero_level = generator(-1)

    def weighted_generator(t):
        return amplitude * np.exp(1.j * angle) * (generator(t) - zero_level) / (1. - zero_level)

    return weighted_generator



def drag(duration, amplitude, sigma, beta, angle=0.0):
    gaussian_waveform = gaussian(duration, amplitude, sigma, angle)
    def generator(t):
        return gaussian_waveform(t) * (1. - 1.j * beta * (t - duration / 2) / sigma**2)
    return generator

def quaternion_drag_x(t):
    duration=256
    amplitude=0.2002363461992037
    sigma=64
    beta=3.279359125685733
    angle=0.0
    drag_waveform = drag(duration, amplitude, sigma, beta, angle)
    complex_amplitude: complex = drag_waveform(t) * OMEGA
    rotation_angle = abs(complex_amplitude)
    return np.quaternion(
        np.cos(rotation_angle / 2),
        np.sin(rotation_angle / 2) * np.cos(frequency_shift * t) * complex_amplitude.real,
        np.sin(rotation_angle / 2) * np.sin(frequency_shift * t) * complex_amplitude.imag, 0)

# xs = list(range(256))
# ys = [drag(256, 0.2, 64, 3.28)(x).real for x in xs]
# plt.plot(xs, ys, color='blue')
# ys = [gaussian(256, 0.2, 64)(x) for x in xs]
# plt.plot(xs, ys, color='red')
# ys = [drag(256, 0.2, 64, 3.28)(x).imag for x in xs]
# plt.plot(xs, ys, color='blue')
# ys = [abs(drag(256, 0.2, 64, 3.28)(x)) for x in xs]
# plt.plot(xs, ys, color='orange')
# plt.show()
# exit()


num_steps = int(t_end - t_start)  # Number of steps for integration

# Create a list of time points
t_list = np.linspace(t_start, t_end - 1, num_steps)
print(t_list)

# Initialize the unitary operator as identity
U_total = qutip.Qobj(np.eye(2))

# Numerical integration to compute the total evolution operator
for i in range(num_steps - 1):
    # t_mid = (t_list[i] + t_list[i + 1]) / 2  # Midpoint for better accuracy
    # H_t_mid = H(t_mid)  # Hamiltonian at midpoint
    H_t_mid = H(t_list[i])
    dt = t_list[i + 1] - t_list[i]  # Time step
    U_dt = (-1j * H_t_mid * dt).expm()  # Unitary operator for this time step
    U_total = U_dt * U_total  # Update total unitary operator

# Display the total unitary operator after time t_end
print("Total Unitary Operator U(0 to {}):".format(t_end))
print(U_total)

# Initialize the unitary operator as identity
q_shifted_total = np.quaternion(1, 0, 0, 0)
q_not_shifted_total = np.quaternion(1, 0, 0, 0)

# Numerical integration to compute the total evolution operator
for i in range(num_steps - 1):
    t = t_list[i]
    q_shifted_total *= quaternion_shifted_h(t)
    q_not_shifted_total *= quaternion_not_shifted_h(t)

print('q_not_shifted_total')
print(q_not_shifted_total)
print(to_spherical_degree(q_not_shifted_total.x, q_not_shifted_total.y, q_not_shifted_total.z))
print()

print('q_shifted_total')
print(q_shifted_total)
print(to_spherical_degree(q_shifted_total.x, q_shifted_total.y, q_shifted_total.z))
print()


q_drag_total = np.quaternion(1, 0, 0, 0)
for i in range(256):
    q_drag_total *= quaternion_h(i)
print('q_drag_total')
print(q_drag_total)
print(to_spherical_degree(q_drag_total.x, q_drag_total.y, q_drag_total.z))
print()


    


# duration=256,
    #                        amplitude=0.2002363461992037,
    #                        sigma=64,
    #                        beta=3.279359125685733,
    #                        angle=0.0