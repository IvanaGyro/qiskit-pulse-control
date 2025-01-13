import pathlib

from matplotlib import animation
import qiskit.quantum_info
from scipy.integrate._ivp import ivp
import numpy as np
from scipy import optimize
from qiskit.circuit import parameter
import matplotlib.pyplot as plt
import qiskit.pulse
import qiskit.pulse.library
import qiskit_dynamics
import qiskit_dynamics.pulse
from qiskit import quantum_info
import qutip

IMAGE_DIR = pathlib.Path('images')

gaussian_x_pi_pulse_amplitudes = {
    (4.6, 4.6, 0.6): 0.17139523332521042,
}


def to_spherical(x, y, z):
    r = (x**2 + y**2 + z**2)**0.5
    theta = np.arctan2((x**2 + y**2)**0.5, z)
    phi = np.arctan2(y, x)
    return (r, theta, phi)


def to_spherical_degree(x, y, z):
    r, theta, phi = to_spherical(x, y, z)
    return (r, theta / np.pi * 180, phi / np.pi * 180)


def plot_qubit_dynamics(sol: ivp.OdeResult,
                        t_eval,
                        X,
                        Y,
                        Z,
                        title=None,
                        filename=None):
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

    _, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': fontsize})
    plt.plot(t_eval, x_data, label=r'$\langle X \rangle$')
    plt.plot(t_eval, y_data, label=r'$\langle Y \rangle$')
    plt.plot(t_eval, z_data, label=r'$\langle Z \rangle$')
    plt.plot(t_eval, r_data, label='$|r|$')
    plt.legend(fontsize=fontsize)
    ax.set_xlabel('$t$', fontsize=fontsize)
    title = title or 'Bloch vector vs. $t$'
    ax.set_title(title, fontsize=fontsize)
    if filename is not None:
        plt.savefig(IMAGE_DIR / f'{filename}.png')

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
        fig,
        animate,
        np.linspace(0, len(x_data) - 1, 30, dtype=np.int32),
        blit=False,
        repeat=False)
    if filename is not None:
        ani.save(IMAGE_DIR / f'{filename}.gif', fps=30)
    else:
        plt.show()


def simulate_two_level():
    r'''Simulate a two level qubit without noise.

    The Hamiltonian is the same as the published Hamiltonian by IBM Quantum.
    $$
    \begin{align*}
    H &= \frac{1}{2} \omega_q ({I-Z}) + \Omega \cos(2 \pi \nu_d t){X} \\
      &= \frac{1}{2} \times 2 \pi \nu_z ({I-Z}) + 2 \pi \nu_x \cos(2 \pi \nu_d t){X}
    \end{align*}
    $$
    '''
    dt = 1 / 4.5  # in nanosecond
    qubit_frequency = 4.6  # in GHz
    drive_frequency = 4.610  # in GHz
    omega = 0.6  # in 10^9 rads

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
            duration=256, amp=0.2, sigma=64)
        qiskit.pulse.play(gaussian, qiskit.pulse.DriveChannel(0))

    converter = qiskit_dynamics.pulse.InstructionToSignals(
        1, carriers={'d0': nu_d})
    gaussian_signal = converter.get_signals(schedule)[0]

    t_final = 256
    tau = 1
    drift = -2 * np.pi * nu_d * Z / 2

    y0 = quantum_info.states.Statevector([1., 0.])

    n_steps = int(np.ceil(t_final / tau)) + 1
    t_eval = np.linspace(0., t_final, n_steps)
    # signals = [Signal(envelope=1., carrier_freq=nu_d)]
    signals = [gaussian_signal]

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=.5 * 2 * np.pi * nu_z * (I - Z),
        hamiltonian_operators=[2 * np.pi * nu_x * X],
        rotating_frame=drift,
        rwa_cutoff_freq=5.0,
    )
    sol = solver.solve(
        t_span=[0., t_final], y0=y0, signals=signals, t_eval=t_eval)
    plot_qubit_dynamics(sol, t_eval, X, Y, Z)


def find_rabi_rate(solver: qiskit_dynamics.Solver,
                   schedule: qiskit.pulse.ScheduleBlock,
                   converter: qiskit_dynamics.pulse.InstructionToSignals,
                   amplitude_parameter: parameter.Parameter,
                   filename: str = None) -> float:
    '''Find Rabi rate for the given model and pulse shape.

    Args:
        solver: the model for calibration
        schedule: the pulse schedule created with the `amplitude_parameter`
        converter: the converter to convert the pulse schedule to signals
        amplitude_parameter: the paramter which will be replaced by the
            amplitude value
        filename: The plot will be saved in the given filename. If filename
            if omitted, the plot will be displayed
    
    Returns:
        The Rabi frequency. Driving the pulse with the returned amplitude
        analogies to drive a 2*pi pulse.
    '''
    t_final = schedule.duration
    tau = 1
    n_steps = int(np.ceil(t_final / tau)) + 1
    t_eval = np.linspace(0., t_final, n_steps)

    y0 = quantum_info.states.Statevector([1., 0.])

    def evolve(amplitude: float):
        current_schedule = schedule.assign_parameters(
            {amplitude_parameter: amplitude}, inplace=False)
        signal = converter.get_signals(current_schedule)[0]
        sol: ivp.OdeResult = solver.solve(
            t_span=[0., t_final], y0=y0, signals=[signal], t_eval=t_eval)
        Z = quantum_info.Operator.from_label('Z')
        final_state: quantum_info.states.Statevector = sol.y[t_final]
        return final_state.expectation_value(Z).real

    amplitudes = np.linspace(0.02, 0.75, 25)
    expectations = [evolve(a) for a in amplitudes]

    points = sorted(zip(expectations, amplitudes))
    guess_max = points[0][0]
    guess_min = points[-1][0]
    guess_amplitude = (guess_max - guess_min) / 2
    guess_base = (guess_max + guess_min) / 2
    # XXX: not a good guess
    guess_period = points[1][1] - points[0][1]
    guess_period = 0.4
    cos_curve = lambda x, amplitude, base, period, phase: amplitude * np.cos(
        2 * np.pi * x / period + phase) + base
    fit_result = optimize.curve_fit(
        cos_curve,
        amplitudes,
        expectations,
        p0=[guess_amplitude, guess_base, guess_period, 0])
    curve_parameters, pcov = fit_result
    errors = np.sqrt(np.diag(pcov))
    period = curve_parameters[2]

    x_fit = np.linspace(min(amplitudes), max(amplitudes), 100)
    y_fit = cos_curve(x_fit, *curve_parameters)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(amplitudes, expectations, color='blue', label='Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
    plt.title(rf'Gaussian Pulse')
    plt.xlabel('Amplitude')
    plt.ylabel(r'Proportion of $|1\rangle$')

    ax = plt.gca()
    ax.text(
        0.5,
        -0.15,
        rf'period = {period:.3f} $\pm$ {errors[2]:.3f}',
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='center',
        transform=ax.transAxes,
        bbox={
            'boxstyle': 'square',
            'facecolor': (0, 0, 0, 0),
        })

    plt.tight_layout()
    if filename is not None:
        plt.savefig(IMAGE_DIR / f'{filename}.png')
    else:
        plt.show()

    return period


def calibrate_x_pulse(qubit_frequency: float = 4.6,
                      drive_frequency: float = 4.6,
                      omega: float = 0.6):
    '''Calibrate the X gate for a qubit.
    
    The Hamiltonian is truncated to two levels from the Hamiltonian published
    by IBM Quantum.
    $$
    \\begin{align*}
    H &= \\frac{1}{2} \\omega_q ({I-Z}) + \\Omega \\cos(2 \\pi \\nu_d t){X} \\\\
      &= \\frac{1}{2} \\times 2 \\pi \\nu_z ({I-Z}) + 2 \\pi \\nu_x \\cos(2 \\pi \\nu_d t){X}
    \\end{align*}
    $$

    Args:
        qubit_frequency: qubit's frequency in GHz
        drive_frequency: signal's frequency in GHz
        omega: A hardward constant in 10^9 rads. This is the value of $\\Omega$
            in the Hamiltonian above.
    '''
    dt = 1 / 4.5  # in nanosecond
    qubit_frequency = qubit_frequency or 4.6  # in GHz
    drive_frequency = drive_frequency or qubit_frequency  # in GHz
    omega = omega or 0.6  # in 10^9 rads

    I = quantum_info.Operator.from_label('I')
    X = quantum_info.Operator.from_label('X')
    Y = quantum_info.Operator.from_label('Y')
    Z = quantum_info.Operator.from_label('Z')

    nu_z = qubit_frequency * dt
    nu_x = omega / 2 / np.pi * dt
    nu_d = drive_frequency * dt
    drift = -2 * np.pi * nu_d * Z / 2

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=.5 * 2 * np.pi * nu_z * (I - Z),
        hamiltonian_operators=[2 * np.pi * nu_x * X],
        rotating_frame=drift,
        rwa_cutoff_freq=nu_d * 1.5,
    )

    amplitude_parameter = parameter.Parameter('amp')
    with qiskit.pulse.build() as x_pi_pulse:
        gaussian = qiskit.pulse.library.Gaussian(
            duration=256, amp=amplitude_parameter, sigma=64)
        qiskit.pulse.play(gaussian, qiskit.pulse.DriveChannel(0))

    converter = qiskit_dynamics.pulse.InstructionToSignals(
        1, carriers={'d0': nu_d})

    key = (qubit_frequency, drive_frequency, omega)
    plot_evolution = False
    if key in gaussian_x_pi_pulse_amplitudes:
        amplitude = key
    else:
        filename = f'simulation-x_gate-rabi-q_{qubit_frequency},d_{drive_frequency},o_{omega}'
        amplitude = find_rabi_rate(
            solver,
            x_pi_pulse,
            converter,
            amplitude_parameter,
            filename=filename)
        amplitude /= 2
        gaussian_x_pi_pulse_amplitudes[key] = amplitude
        plot_evolution = True
        print(f'{key}: {amplitude:.17f}')

    x_pi_pulse.assign_parameters({amplitude_parameter: amplitude})
    signal = converter.get_signals(x_pi_pulse)[0]

    y0 = np.eye(2, dtype=complex)
    t_final = x_pi_pulse.duration
    tau = 1
    n_steps = int(np.ceil(t_final / tau)) + 1
    t_eval = np.linspace(0., t_final, n_steps)

    if plot_evolution:
        sol: ivp.OdeResult = solver.solve(
            t_span=[0., t_final],
            y0=quantum_info.states.Statevector([1., 0.]),
            signals=[signal],
            t_eval=t_eval)
        title = rf'$\omega_q = {qubit_frequency} GHz$ $\omega_d = {drive_frequency} GHz$ $\Omega = {omega}$'
        filename = f'simulation-x_gate-q_{qubit_frequency},d_{drive_frequency},o_{omega}'
        plot_qubit_dynamics(
            sol, t_eval, X, Y, Z, title=title, filename=filename)

    sol: ivp.OdeResult = solver.solve(
        t_span=[0., t_final],
        y0=y0,
        signals=[signal],
        t_eval=t_eval,
        atol=1e-10,
        rtol=1e-10)
    # rotation_matrix = $\cos(\theta / 2) I - i \sin(\theta / 2) \hat{n} \cdot \vec{\sigma}$
    n_times = len(sol.y)
    x_data = np.zeros((n_times,))
    y_data = np.zeros((n_times,))
    z_data = np.zeros((n_times,))
    r_data = np.zeros((n_times,))

    for t_i, sol_t in enumerate(sol.y):
        rotation_matrix = sol_t
        # rotation_matrix = solver.model.rotating_frame.state_out_of_frame(
        #     t_i, sol_t)
        x = -np.imag(rotation_matrix[1][0] + rotation_matrix[0][1]) / 2
        y = np.real(rotation_matrix[1][0] - rotation_matrix[0][1]) / 2
        z = np.imag(rotation_matrix[1][1] - rotation_matrix[0][0]) / 2
        i = np.real(rotation_matrix[0][0] + rotation_matrix[1][1]) / 2
        # XXX: This restrict the rotation angle lower than pi. Why do we need
        #   to do this correction? Is this correction valid?
        if i < 0:
            x, y, z, i = -x, -y, -z, -i
        # final_state = sol_t.data
        # x = -np.imag(final_state[1])
        # y = np.real(final_state[1])
        # z = - np.imag(final_state[0])
        # i = np.real(final_state[0])
        v = np.array([x, y, z])
        r = np.sqrt(v.dot(v))
        if r != 0:
            v /= r
        x, y, z = v
        x_data[t_i] = x
        y_data[t_i] = y
        z_data[t_i] = z
        r_data[t_i] = np.arctan2(r, i) * 2
    x, y, z = x_data[-1], y_data[-1], z_data[-1]
    # print(f'axis (x, y, z): {(x, y, z)}')
    # print(f'axis (r, theta, phi)(degree): {to_spherical_degree(x, y, z)}')
    # print(f'rotation angle: {r_data[-1]}')

    fontsize = 16
    _, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': fontsize})
    plt.plot(t_eval, x_data, label=r'X')
    plt.plot(t_eval, y_data, label=r'Y')
    plt.plot(t_eval, z_data, label=r'Z')
    plt.plot(t_eval, r_data, label=r'$\theta$')
    plt.legend(fontsize=fontsize)
    ax.set_xlabel('$t$', fontsize=fontsize)
    title = rf'Rotation Axis: $\omega_q = {qubit_frequency} GHz$ $\omega_d = {drive_frequency} GHz$ $\Omega = {omega}$'
    filename = f'simulation-x_gate-axis-q_{qubit_frequency},d_{drive_frequency},o_{omega}'
    ax.set_title(title, fontsize=fontsize)
    plt.savefig(IMAGE_DIR / f'{filename}.png')
    # plt.show()


def demo_rotate_two_rounds():
    r'''Demo how to set the qubit frequency and the Hamiltonian properly.

    This demo rotates the state on the XY plane two rounds in 256 dt.

    The Hamiltonian is the same as the published Hamiltonian by IBM Quantum.
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
    dt = 1 / 4.5  # in nanosecond
    qubit_frequency = 4.6  # in GHz
    drive_frequency = qubit_frequency + rounds / t_final / dt  # in GHz
    omega = 0.6  # in 10^9 rads

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
    drift = -2 * np.pi * nu_d * Z / 2

    solver = qiskit_dynamics.Solver(
        static_hamiltonian=.5 * 2 * np.pi * nu_z * (I - Z),
        hamiltonian_operators=[2 * np.pi * nu_x * X],
        rotating_frame=drift,
        rwa_cutoff_freq=5.0,
    )
    sol = solver.solve(
        t_span=[0., t_final], y0=y0, signals=signals, t_eval=t_eval)
    plot_qubit_dynamics(sol, t_eval, X, Y, Z)


def draw_pulse_signal():
    '''Convert the pulse schedule to signal, and draw the signal.'''
    dt = 1 / 4.5  # in nanosecond
    qubit_frequency = 4.5  # in GHz
    with qiskit.pulse.build() as schedule:
        gaussian = qiskit.pulse.library.Gaussian(
            duration=256, amp=0.2, sigma=64)
        qiskit.pulse.play(gaussian, qiskit.pulse.DriveChannel(0))

    converter = qiskit_dynamics.pulse.InstructionToSignals(
        1, carriers={'d0': qubit_frequency * dt})
    signal = converter.get_signals(schedule)[0]

    figure, ax = plt.subplots()
    signal.draw(0, 256, 25600, axis=ax)

    plt.show()


def main():
    calibrate_x_pulse()


if __name__ == '__main__':
    main()
