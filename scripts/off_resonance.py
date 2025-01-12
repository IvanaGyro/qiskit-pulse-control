from matplotlib import animation
from scipy.integrate._ivp import ivp
import numpy as np
import matplotlib.pyplot as plt
import qiskit.pulse
import qiskit.pulse.library
import qiskit_dynamics
import qiskit_dynamics.pulse
from qiskit import quantum_info
import qutip


def plot_qubit_dynamics(sol: ivp.OdeResult, t_eval, X, Y, Z):
    '''Plot the tomography of the qubit dynamics'''
    fontsize = 16
    n_times = len(sol.y)
    x_data = np.zeros((n_times,))
    y_data = np.zeros((n_times,))
    z_data = np.zeros((n_times,))
    r_data = np.zeros((n_times,))

    for t_i, sol_t in enumerate(sol.y):
        x_data[t_i] = sol_t.expectation_value(X).real
        y_data[t_i] = sol_t.expectation_value(Y).real
        z_data[t_i] = sol_t.expectation_value(Z).real
        r_data[t_i] = (x_data[t_i]**2 + y_data[t_i]**2 + z_data[t_i]**2)**0.5

    _, ax = plt.subplots(figsize = (10, 6))
    plt.rcParams.update({'font.size': fontsize})
    plt.plot(t_eval, x_data, label = r'$\langle X \rangle$')
    plt.plot(t_eval, y_data, label = r'$\langle Y \rangle$')
    plt.plot(t_eval, z_data, label = r'$\langle Z \rangle$')
    plt.plot(t_eval, r_data, label = '$|r|$')
    plt.legend(fontsize = fontsize)
    ax.set_xlabel('$t$', fontsize = fontsize)
    ax.set_title('Bloch vector vs. $t$', fontsize = fontsize)


    fig = plt.figure()
    ax = fig.add_subplot(azim=-40, elev=30, projection='3d')
    sphere = qutip.Bloch(axes=ax)

    def animate(i):
        sphere.clear()
        sphere.add_vectors([x_data[i], y_data[i], z_data[i]], ['r'])
        sphere.add_points([x_data[:i + 1], y_data[:i + 1], z_data[:i + 1]])
        sphere.make_sphere()
        return ax

    ani = animation.FuncAnimation(
        fig, animate, range(len(x_data)), blit=False, repeat=False)
    ani.save(
        f'x_gate_trajectory.gif',
        fps=30)
    plt.show()



def simulate_two_level():
    r'''Simulate a two level qubit without noise.

    The Hamiltonian is the same as the pulished Hamiltonian by IBM Quantum.
    $$
    \begin{align*}
    H &= \frac{1}{2} \omega_q ({I-Z}) + \Omega \cos(2 \pi \nu_d t){X} \\
      &= \frac{1}{2} \times 2 \pi \nu_z ({I-Z}) + 2 \pi \nu_x \cos(2 \pi \nu_d t){X}
    \end{align*}
    $$
    '''
    dt = 1 / 4.5 # in nanosecond
    qubit_frequency = 4.6 # in GHz
    drive_frequency = 4.610 # in GHz
    omega = 0.6 # in 10^9 rads

    nu_z = qubit_frequency * dt
    nu_x = omega / 2 / np.pi * dt
    nu_d = drive_frequency * dt

    I = quantum_info.Operator.from_label('I')
    X = quantum_info.Operator.from_label('X')
    Y = quantum_info.Operator.from_label('Y')
    Z = quantum_info.Operator.from_label('Z')
    s_p = 0.5 * (X + 1j * Y)



    with qiskit.pulse.build() as schedule:
        gaussian = qiskit.pulse.library.Gaussian(
            duration=256 , amp=0.2, sigma=64)
        qiskit.pulse.play(gaussian, qiskit.pulse.DriveChannel(0))

    converter = qiskit_dynamics.pulse.InstructionToSignals(
        1, carriers={'d0': nu_d })
    gaussian_signal = converter.get_signals(schedule)[0]

    t_final = 256
    tau = 1
    drift = -2 * np.pi * nu_d * Z/2

    y0 = quantum_info.states.Statevector([1., 0.]) 

    n_steps = int(np.ceil(t_final / tau)) + 1
    t_eval = np.linspace(0., t_final, n_steps)
    # signals = [Signal(envelope=1., carrier_freq=nu_d)]
    signals = [gaussian_signal]

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=.5 * 2 * np.pi * nu_z * (I- Z),
        hamiltonian_operators=[2 * np.pi * nu_x * X],
        rotating_frame=drift,
        rwa_cutoff_freq=5.0,
    )
    sol = solver.solve(t_span=[0., t_final], y0=y0, signals=signals, t_eval=t_eval)
    plot_qubit_dynamics(sol, t_eval, X, Y, Z)


def demo_rotate_two_rounds():
    r'''Demo how to set the qubit frequency and the Hamiltonian properly.

    This demo rotates the state on the XY plane two rounds in 256 dt.

    The Hamiltonian is the same as the pulished Hamiltonian by IBM Quantum.
    $$
    \begin{align*}
    H &= \frac{1}{2} \omega_q ({I-Z}) + \Omega \cos(2 \pi \nu_d t){X} \\
      &= \frac{1}{2} \times 2 \pi \nu_z ({I-Z}) + 2 \pi \nu_x \cos(2 \pi \nu_d t){X}
    \end{align*}
    $$
    '''
    t_final = 256
    tau = 0.5

    rounds = 2
    dt = 1 / 4.5 # in nanosecond
    qubit_frequency = 4.6 # in GHz
    drive_frequency = qubit_frequency + rounds / t_final / dt # in GHz
    omega = 0.6 # in 10^9 rads

    nu_z = qubit_frequency * dt
    nu_x = omega / 2 / np.pi * dt
    nu_d = drive_frequency * dt

    I = quantum_info.Operator.from_label('I')
    X = quantum_info.Operator.from_label('X')
    Y = quantum_info.Operator.from_label('Y')
    Z = quantum_info.Operator.from_label('Z')

    y0 = quantum_info.states.Statevector([.5**.5, .5**.5]) 

    n_steps = int(np.ceil(t_final / tau)) + 1
    t_eval = np.linspace(0., t_final, n_steps)
    signals = [qiskit_dynamics.signals.Signal(0)]
    drift = -2 * np.pi * nu_d * Z/2

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=.5 * 2 * np.pi * nu_z * (I- Z),
        hamiltonian_operators=[2 * np.pi * nu_x * X],
        rotating_frame=drift,
        rwa_cutoff_freq=5.0,
    )
    sol = solver.solve(t_span=[0., t_final], y0=y0, signals=signals, t_eval=t_eval)
    plot_qubit_dynamics(sol, t_eval, X, Y, Z)


def draw_pulse_signal():
    '''Convert the pulse schedule to signal, and draw the signal.'''
    dt = 1 / 4.5 # in nanosecond
    qubit_frequency = 4.5 # in GHz
    with qiskit.pulse.build() as schedule:
        gaussian = qiskit.pulse.library.Gaussian(
            duration=256 , amp=0.2, sigma=64)
        qiskit.pulse.play(gaussian, qiskit.pulse.DriveChannel(0))

    converter = qiskit_dynamics.pulse.InstructionToSignals(
        1, carriers={'d0': qubit_frequency * dt })
    signal = converter.get_signals(schedule)[0]

    figure, ax = plt.subplots()
    signal.draw(0, 256, 25600, axis=ax)

    plt.show()

def main():
    demo_rotate_two_rounds()

if __name__ == '__main__':
    main()
