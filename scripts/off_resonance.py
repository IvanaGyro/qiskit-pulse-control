import pathlib

from matplotlib import animation
from klepto import archives
import dataclasses
from klepto import keymaps
import qiskit.quantum_info
import qiskit_dynamics.arraylias
from scipy.integrate._ivp import ivp
import numpy as np
import numpy.typing
from scipy import optimize
from qiskit.circuit import parameter
import matplotlib.pyplot as plt
import qiskit.pulse
import qiskit.pulse.library
import qiskit_dynamics
import qiskit_dynamics.pulse
from qiskit import quantum_info
import qutip
from klepto import safe
import time

IMAGE_DIR = pathlib.Path('images') / 'off_resonance'
CACHE_DIR = pathlib.Path('cache')


@dataclasses.dataclass
class PulseFinal:
    amplitude: float
    axis_x: float = None
    axis_y: float = None
    axis_z: float = None
    angle: float = None
    is_good: bool = True


_profiler: dict[str, float] = {}


def start_timer(label: str):
    '''
    Start timing an operation.

    Parameters:
        label (str): A unique label to identify the timer.
    '''
    global _profiler
    _profiler[label] = time.time()


def stop_timer(label: str):
    '''
    Stop the timer for an operation and print the elapsed time.

    Parameters:
        label (str): The label used to start the timer.

    Returns:
        elapsed_time (float): The elapsed time in seconds.
    '''
    global _profiler
    if label not in _profiler:
        raise ValueError(f"No timer started with label '{label}'.")
    elapsed_time = time.time() - _profiler.pop(label)
    print(f"Timer '{label}' completed in {elapsed_time:.6f} seconds.")
    return elapsed_time


def decompose_unitary(unitary: numpy.typing.NDArray) -> tuple[float]:
    r'''Decompose 2x2 unitary operator to the composition of Pauli matrices.

    The input unitart matrix should be in this form:

    $$
    U = e^{i\phi} \cos(\theta / 2) I - i e^{i\phi} \sum_{i \in \{x, y, z\}} n_i \sin(\theta / 2) \sigma_i
    $$

    Returns:
        A tuple of which values are `(\cos(\theta / 2), n_x \sin(\theta / 2), n_y \sin(\theta / 2), n_z \sin(\theta / 2))`
    '''
    TOLERANCE = 1e-8
    if unitary.shape != (2, 2):
        raise ValueError(f'Unitary should be a 2x2 matrix. unitary: {unitary}')
    pauli_operators = quantum_info.SparsePauliOp.from_operator(unitary)
    if not pauli_operators.is_unitary():
        raise ValueError(f'Matrix is not an unitary. matrix: {unitary}')
    coefficients = {'I': 0, 'X': 0, 'Y': 0, 'Z': 0}
    coefficients.update(pauli_operators.to_list())
    global_phase = coefficients['I'] / np.abs(coefficients['I'])

    corrected_coefficient = coefficients['I'] / global_phase
    if abs(np.imag(corrected_coefficient)) > TOLERANCE:
        raise ValueError(
            f'Error higher than the tolerance. input unitary:{unitary}')
    coefficients['I'] = np.real(corrected_coefficient)
    for label in ('X', 'Y', 'Z'):
        corrected_coefficient = coefficients[label] / global_phase
        if abs(np.real(corrected_coefficient)) > TOLERANCE:
            raise ValueError(
                f'Error higher than the tolerance. input unitary:{unitary}')
        coefficients[label] = -np.imag(coefficients[label] / global_phase)
    return tuple(coefficients.values())


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
    Z = quantum_info.Operator.from_label('Z')

    def evolve(amplitude: float):
        current_schedule = schedule.assign_parameters(
            {amplitude_parameter: amplitude}, inplace=False)
        sol: ivp.OdeResult = solver.solve(
            t_span=[0., t_final],
            y0=y0,
            signals=current_schedule,
            t_eval=t_eval)
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
        p0=[guess_amplitude, guess_base, guess_period, 0],
        maxfev=10000)
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


@safe.inf_cache()
def two_level_solver(qubit_frequency: float = 4.6,
                     drive_frequency: float = 4.6,
                     omega: float = 0.6) -> qiskit_dynamics.Solver:
    '''Establish a solver for simulating a qubit.

    The solver is transformed into the rotating frame on the pulse's frequency,
    and applied RWA.

    The Hamiltonian in the lab frame is truncated to two levels from the
    Hamiltonian published by IBM Quantum.
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
    rotating_frame_frequency = nu_d
    drift = -2 * np.pi * rotating_frame_frequency * Z / 2

    # One of `channel_carrier_freqs` or `rwa_carrier_freqs` should be set to
    # make RWA works
    # `rwa_cutoff_freq` should be in the range:
    #   (abs(rotating_frame_frequency - rwa_carrier_freqs),
    #    rotating_frame_frequency + rwa_carrier_freqs)
    # If `rwa_carrier_freqs` is omitted, the value of `rwa_carrier_freqs`
    # will be replaced by `channel_carrier_freqs`.
    return qiskit_dynamics.Solver(
        static_hamiltonian=.5 * 2 * np.pi * nu_z * (I - Z),
        hamiltonian_operators=[2 * np.pi * nu_x * X],
        hamiltonian_channels=['d0'],
        channel_carrier_freqs={'d0': nu_d},
        dt=1,
        rotating_frame=drift,
        rwa_cutoff_freq=nu_d * 1.5,
    )


@safe.no_cache(
    cache=archives.dir_archive(name=CACHE_DIR / 'scripts' /
                               'calibrate_x_gaussian_pulse'),
    keymap=keymaps.hashmap(algorithm='md5'))
def calibrate_x_gaussian_pulse(qubit_frequency: float, drive_frequency: float,
                               omega: float, duration: int,
                               sigma: float) -> float:
    '''Calibrate the X gate for a qubit.

    Find the best amplitude of the Gaussian pulse with using Rabi oscillation.

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
        duration: duration of the pulse in dt.
        sigma: sigma value of the Gaussian pulse in dt.
    
    Returns:
        The amplitude of the Gaussian pulse.
    '''
    amplitude_parameter = parameter.Parameter('amp')
    with qiskit.pulse.build() as x_pi_pulse:
        gaussian = qiskit.pulse.library.Gaussian(
            duration=duration, amp=amplitude_parameter, sigma=sigma)
        qiskit.pulse.play(gaussian, qiskit.pulse.DriveChannel(0))

    solver = two_level_solver(qubit_frequency, drive_frequency, omega)
    filename = f'simulation-x_gate-rabi-q_{qubit_frequency},d_{drive_frequency},o_{omega}'
    amplitude = find_rabi_rate(
        solver, x_pi_pulse, amplitude_parameter, filename=filename)
    amplitude /= 2
    if amplitude > 1.0:
        raise ValueError('Rabi curve fitting failed.')
    return amplitude


@safe.no_cache(
    cache=archives.dir_archive(name=CACHE_DIR / 'scripts' /
                               'calibrate_and_evaluate_x_gaussian_pulse'),
    keymap=keymaps.hashmap(algorithm='md5'))
def calibrate_and_evaluate_x_gaussian_pulse(qubit_frequency: float,
                                            drive_frequency: float,
                                            omega: float,
                                            duration: int = 256,
                                            sigma: float = 64) -> PulseFinal:
    '''Calibrate the X gate for a qubit and calculate the effective rotation.

    Find the best amplitude of the Gaussian pulse with using Rabi oscillation,
    and calculate the equivalent rotation axis and angle of the pulse at each
    time step.

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
        duration: duration of the pulse in dt.
        sigma: sigma value of the Gaussian pulse in dt.
    
    Returns:
        The amplitude of the Gaussian pulse.
    '''
    start_timer('rabi')
    amplitude = calibrate_x_gaussian_pulse(qubit_frequency, drive_frequency,
                                           omega, duration, sigma)
    stop_timer('rabi')

    with qiskit.pulse.build() as x_pi_pulse:
        gaussian = qiskit.pulse.library.Gaussian(
            duration=duration, amp=amplitude, sigma=sigma)
        qiskit.pulse.play(gaussian, qiskit.pulse.DriveChannel(0))
    solver = two_level_solver(qubit_frequency, drive_frequency, omega)

    t_final = x_pi_pulse.duration
    tau = 1
    n_steps = int(np.ceil(t_final / tau)) + 1
    t_eval = np.linspace(0., t_final, n_steps)

    # plot evolution
    X = quantum_info.Operator.from_label('X')
    Y = quantum_info.Operator.from_label('Y')
    Z = quantum_info.Operator.from_label('Z')
    start_timer('state_evole')
    sol: ivp.OdeResult = solver.solve(
        t_span=[0., t_final],
        y0=quantum_info.states.Statevector([1., 0.]),
        signals=x_pi_pulse,
        t_eval=t_eval)
    title = rf'$\omega_q = {qubit_frequency} GHz$ $\omega_d = {drive_frequency} GHz$ $\Omega = {omega}$'
    filename = f'simulation-x_gate-q_{qubit_frequency},d_{drive_frequency},o_{omega}'
    stop_timer('state_evole')

    start_timer('plot_qubit_dynamics')
    plot_qubit_dynamics(sol, t_eval, X, Y, Z, title=title, filename=filename)
    stop_timer('plot_qubit_dynamics')

    # the tolerance should small enough to make the result enough close to an unitary
    # and to make the error of decompsed coefficient small enough
    start_timer('unitary_evolve')
    sol: ivp.OdeResult = solver.solve(
        t_span=[0., t_final],
        y0=np.eye(2, dtype=np.complex128),
        signals=x_pi_pulse,
        t_eval=t_eval,
        atol=1e-12,
        rtol=1e-12)
    stop_timer('unitary_evolve')

    # rotation_matrix = $\cos(\theta / 2) I - i \sin(\theta / 2) \hat{n} \cdot \vec{\sigma}$
    n_times = len(sol.y)
    x_data = np.zeros((n_times,))
    y_data = np.zeros((n_times,))
    z_data = np.zeros((n_times,))
    r_data = np.zeros((n_times,))

    start_timer('decompose_unitary_series')
    previous_axis = np.zeros(3)
    for t_i, sol_t in enumerate(sol.y):
        rotation_matrix = sol_t
        i, x, y, z = decompose_unitary(rotation_matrix)
        axis = np.array((x, y, z))
        if np.dot(previous_axis, axis) < 0:
            i, x, y, z = -i, -x, -y, -z
            axis = -axis
        previous_axis = axis
        # final_state = sol_t.data
        # x = -np.imag(final_state[1])
        # y = np.real(final_state[1])
        # z = - np.imag(final_state[0])
        # i = np.real(final_state[0])
        r = np.sqrt(axis.dot(axis))
        if r != 0:
            axis /= r
        x, y, z = axis
        x_data[t_i] = x
        y_data[t_i] = y
        z_data[t_i] = z
        r_data[t_i] = np.arctan2(r, i) * 2
    x, y, z = x_data[-1], y_data[-1], z_data[-1]
    stop_timer('decompose_unitary_series')

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
    return PulseFinal(amplitude, axis_x=x, axis_y=y, axis_z=z, angle=r_data[-1])


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
    rotating_frame_frequency = nu_d
    drift = -2 * np.pi * rotating_frame_frequency * Z / 2

    # rwa_cutoff_freq should be set in this range:
    # abs(rotating_frame_frequency - drive_frequency * dt) < rwa_cutoff_freq < rotating_frame_frequency + drive_frequency * dt
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


def plot_rabi_frequencies(drive_frequencies: list[float],
                          amplitudes: list[float], xs: list[float],
                          ys: list[float], zs: list[float],
                          angles: list[float]):
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(10, 30))

    ax0.set_xscale('symlog')
    ax0.set_xlabel('off resonance (Hz)')
    ax0.set_ylabel('amplitudes')
    ax0.grid()
    ax0.xaxis.grid(which='minor')
    ax0.scatter(drive_frequencies, amplitudes)

    ax1.set_xscale('symlog')
    ax1.set_xlabel('off resonance (GHz)')
    ax1.set_ylabel('x')
    ax1.grid()
    ax1.xaxis.grid(which='minor')
    ax1.scatter(drive_frequencies, xs)

    ax2.set_xscale('symlog')
    ax2.set_xlabel('off resonance (GHz)')
    ax2.set_ylabel('y')
    ax2.grid()
    ax2.xaxis.grid(which='minor')
    ax2.scatter(drive_frequencies, ys)

    ax3.set_xscale('symlog')
    ax3.set_xlabel('off resonance (GHz)')
    ax3.set_ylabel('z')
    ax3.grid()
    ax3.xaxis.grid(which='minor')
    ax3.scatter(drive_frequencies, zs)

    ax4.set_xscale('symlog')
    ax4.set_xlabel('off resonance (GHz)')
    ax4.set_ylabel('angle(rads)')
    ax4.grid()
    ax4.xaxis.grid(which='minor')
    ax4.scatter(drive_frequencies, angles)
    fig.savefig(IMAGE_DIR / 'simulation-x_gate-off_resonance.png')


def calibrate_x_pulse_with_off_resonance():
    qubit_frequency = 4.6
    omega = 0.6
    off_resonance_rate = 1e-9
    off_resonances = []
    while off_resonance_rate <= 0.0025:
        off_resonances.append(off_resonance_rate * qubit_frequency)
        off_resonance_rate *= 2
    off_resonances = [-r for r in off_resonances[::-1]] + [0] + off_resonances

    drive_frequencies = []
    amplitudes = []
    xs = []
    ys = []
    zs = []
    angles = []
    for r in off_resonances:
        off_resonance_frequency = qubit_frequency * r
        pulse_final: PulseFinal = calibrate_and_evaluate_x_gaussian_pulse(
            qubit_frequency=qubit_frequency,
            drive_frequency=qubit_frequency + off_resonance_frequency,
            omega=omega)
        if not pulse_final.is_good:
            continue
        drive_frequencies.append(off_resonance_frequency)
        amplitudes.append(pulse_final.amplitude)
        xs.append(pulse_final.axis_x)
        ys.append(pulse_final.axis_y)
        zs.append(pulse_final.axis_z)
        angles.append(pulse_final.angle)
    plot_rabi_frequencies(drive_frequencies, amplitudes, xs, ys, zs, angles)


def main():
    calibrate_and_evaluate_x_gaussian_pulse(
        qubit_frequency=4.6, drive_frequency=4.6, omega=0.6)


if __name__ == '__main__':
    calibrate_x_pulse_with_off_resonance()
