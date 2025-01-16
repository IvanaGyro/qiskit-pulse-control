import pathlib

from matplotlib import animation
import dataclasses
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


@dataclasses.dataclass
class PulseFinal:
    amplitude: float
    axis_x: float = None
    axis_y: float = None
    axis_z: float = None
    angle: float = None
    is_good: bool = True


gaussian_x_pi_pulse_amplitudes = {
    (4.6, 3.1799764377599997, 0.6):
        PulseFinal(0.20771771701029515, is_good=False),
    (4.6, 3.8899882188799997, 0.6):
        PulseFinal(0.21203950419996329, is_good=False),
    (4.6, 4.244994109439999, 0.6):
        PulseFinal(0.21236624571861701, is_good=False),
    (4.6, 4.42249705472, 0.6):
        PulseFinal(21.48039477256234520, is_good=False),
    (4.6, 4.511248527359999, 0.6):
        PulseFinal(0.17934235173812585, is_good=False),
    (4.6, 4.5556242636799995, 0.6):
        PulseFinal(0.35985948532730261, is_good=False),
    (4.6, 4.57781213184, 0.6):
        PulseFinal(0.21283439135763774, is_good=False),
    (4.6, 4.58890606592, 0.6):
        PulseFinal(
            0.16728223568865005,  # may be fitted
            axis_x=0.96488868198353928,
            axis_y=0.00001101614430950,
            axis_z=-0.26265915414984686,
            angle=4.67668869579144530),
    (4.6, 4.59445303296, 0.6):
        PulseFinal(
            0.17036195302211765,
            axis_x=0.94307236712410603,
            axis_y=-0.00000040448335726,
            axis_z=-0.33258759803512161,
            angle=3.62924957830966299),
    (4.6, 4.597226516479999, 0.6):
        PulseFinal(
            0.17113667889129439,
            axis_x=0.98032047891629082,
            axis_y=0.00000082644385342,
            axis_z=-0.19741266073038782,
            angle=3.27431242755625584),
    (4.6, 4.5986132582399994, 0.6):
        PulseFinal(
            0.17133934930919950,
            axis_x=0.99462819404859804,
            axis_y=0.00000183395329666,
            axis_z=-0.10351210364136700,
            angle=3.17781258523558297),
    (4.6, 4.59930662912, 0.6):
        PulseFinal(
            0.17135043147325416,
            axis_x=0.99860818459813916,
            axis_y=0.00000130181027158,
            axis_z=-0.05274176383013795,
            angle=3.15233431386912466),
    (4.6, 4.599653314559999, 0.6):
        PulseFinal(
            0.17137239636954632,
            axis_x=0.99964028803978422,
            axis_y=0.00000081534716577,
            axis_z=-0.02681966679645123,
            angle=3.14623734669254507),
    (4.6, 4.5998266572799995, 0.6):
        PulseFinal(
            0.17135627960575933,
            axis_x=0.99990476763079350,
            axis_y=0.00000046877719371,
            axis_z=-0.01380056770531304,
            angle=3.14429392298926391),
    (4.6, 4.59991332864, 0.6):
        PulseFinal(
            0.17136910924764454,
            axis_x=0.99997351267279033,
            axis_y=0.00000022019714692,
            axis_z=-0.00727832073986296,
            angle=3.14410683943996894),
    (4.6, 4.59995666432, 0.6):
        PulseFinal(
            0.17136852966519900,
            axis_x=0.99999193036721301,
            axis_y=0.00000015117609689,
            axis_z=-0.00401736237250795,
            angle=3.14398540076860034),
    (4.6, 4.59997833216, 0.6):
        PulseFinal(
            0.17141170460212682,
            axis_x=0.99999715382532550,
            axis_y=0.00000009840983789,
            axis_z=-0.00238586278703433,
            angle=3.14474728249174351),
    (4.6, 4.599989166079999, 0.6):
        PulseFinal(
            0.17135132519118426,
            axis_x=0.99999876557268386,
            axis_y=0.00000004465502690,
            axis_z=-0.00157125844676158,
            angle=3.14363065414222165),
    (4.6, 4.59999458304, 0.6):
        PulseFinal(
            0.17130237025958450,
            axis_x=0.99999932314247686,
            axis_y=0.00000006627287056,
            axis_z=-0.00116349240819024,
            angle=3.14272961505688775),
    (4.6, 4.599997291519999, 0.6):
        PulseFinal(
            0.17136212264917364,
            axis_x=0.99999953937184705,
            axis_y=0.00000006045539854,
            axis_z=-0.00095982086354704,
            angle=3.14382483873684926),
    (4.6, 4.5999986457599995, 0.6):
        PulseFinal(
            0.17136871072334489,
            axis_x=0.99999963211706988,
            axis_y=0.00000005423042891,
            axis_z=-0.00085776787168922,
            angle=3.14394535282259513),
    (4.6, 4.59999932288, 0.6):
        PulseFinal(
            0.17137843827692617,
            axis_x=0.99999967448595284,
            axis_y=0.00000003668257190,
            axis_z=-0.00080686305349487,
            angle=3.14412353493167052),
    (4.6, 4.59999966144, 0.6):
        PulseFinal(
            0.17136115941993188,
            axis_x=0.99999969475945960,
            axis_y=0.00000002457365010,
            axis_z=-0.00078133282726847,
            angle=3.14380653413064781),
    (4.6, 4.59999983072, 0.6):
        PulseFinal(
            0.17139013599485400,
            axis_x=0.99999970456073539,
            axis_y=0.00000004236706481,
            axis_z=-0.00076868617802037,
            angle=3.14433796907335505),
    (4.6, 4.59999991536, 0.6):
        PulseFinal(
            0.17140303457626052,
            axis_x=0.99999970933405880,
            axis_y=0.00000005573235023,
            axis_z=-0.00076245117538000,
            angle=3.14457463492636213),
    (4.6, 4.59999995768, 0.6):
        PulseFinal(
            0.17138503059944393,
            axis_x=0.99999971184115322,
            axis_y=0.00000000215637593,
            axis_z=-0.00075915585388688,
            angle=3.14424434069695824),
    (4.6, 4.59999997884, 0.6):
        PulseFinal(
            0.17139348209056163,
            axis_x=0.99999971306658986,
            axis_y=0.00000003652250462,
            axis_z=-0.00075753992408174,
            angle=3.14439938838032873),
    (4.6, 4.6, 0.6):
        PulseFinal(
            0.17139523332521042,
            axis_x=0.99999971421117984,
            axis_y=0.00000001901967653,
            axis_z=-0.00075602748517285,
            angle=3.14443161693991557),
    (4.6, 4.60000002116, 0.6):
        PulseFinal(
            0.17139619311637033,
            axis_x=0.99999971546555588,
            axis_y=0.00000000757997250,
            axis_z=-0.00075436649413242,
            angle=3.14444930990353244),
    (4.6, 4.60000004232, 0.6):
        PulseFinal(
            0.17139949422881037,
            axis_x=0.99999971656249020,
            axis_y=0.00000004098356502,
            axis_z=-0.00075291097584982,
            angle=3.14450975421611423),
    (4.6, 4.6000000846399995, 0.6):
        PulseFinal(
            0.17139103566873276,
            axis_x=0.99999971902875362,
            axis_y=0.00000003462113393,
            axis_z=-0.00074962818290336,
            angle=3.14435450755022172),
    (4.6, 4.600000169279999, 0.6):
        PulseFinal(
            0.17137946432454249,
            axis_x=0.99999972382802305,
            axis_y=0.00000008240163789,
            axis_z=-0.00074319840626505,
            angle=3.14414214616878152),
    (4.6, 4.600000338559999, 0.6):
        PulseFinal(
            0.17138427341894377,
            axis_x=0.99999973323800506,
            axis_y=0.00000003585779242,
            axis_z=-0.00073042721590207,
            angle=3.14423045376212196),
    (4.6, 4.60000067712, 0.6):
        PulseFinal(
            0.17137156771594148,
            axis_x=0.99999975156711951,
            axis_y=0.00000010228493657,
            axis_z=-0.00070488700435663,
            angle=3.14399720409465822),
    (4.6, 4.60000135424, 0.6):
        PulseFinal(
            0.17135159385934079,
            axis_x=0.99999978624432073,
            axis_y=0.00000000955419885,
            axis_z=-0.00065384349273839,
            angle=3.14363060257956839),
    (4.6, 4.60000270848, 0.6):
        PulseFinal(
            0.17136889280774895,
            axis_x=0.99999984758768767,
            axis_y=0.00000003145959179,
            axis_z=-0.00055210922887040,
            angle=3.14394775581885177),
    (4.6, 4.600005416959999, 0.6):
        PulseFinal(
            0.17135567103079066,
            axis_x=0.99999993945255861,
            axis_y=0.00000005464723331,
            axis_z=-0.00034798689088573,
            angle=3.14370490707063599),
    (4.6, 4.60001083392, 0.6):
        PulseFinal(
            0.17138879334161436,
            axis_x=0.99999999825588537,
            axis_y=-0.00000004435508268,
            axis_z=0.00005906121825303,
            angle=3.14431302482974795),
    (4.6, 4.600021667839999, 0.6):
        PulseFinal(
            0.17139150412566811,
            axis_x=0.99999961783900337,
            axis_y=-0.00000002156768680,
            axis_z=0.00087425502380218,
            angle=3.14436633847340996),
    (4.6, 4.60004333568, 0.6):
        PulseFinal(
            0.17134918485710438,
            axis_x=0.99999685851180675,
            axis_y=-0.00000001783647431,
            axis_z=0.00250658463193705,
            angle=3.14360979345909231),
    (4.6, 4.60008667136, 0.6):
        PulseFinal(
            0.17132695793905250,
            axis_x=0.99998334879397888,
            axis_y=-0.00000013015819195,
            axis_z=0.00577080018391834,
            angle=3.14329223601122765),
    (4.6, 4.60017334272, 0.6):
        PulseFinal(
            0.17141443121571648,
            axis_x=0.99992461041012548,
            axis_y=-0.00000031849798417,
            axis_z=0.01227898595395752,
            angle=3.14527784710552050),
    (4.6, 4.60034668544, 0.6):
        PulseFinal(
            0.17134004836817715,
            axis_x=0.99967935394335983,
            axis_y=-0.00000084607339136,
            axis_z=0.02532171595036271,
            angle=3.14547909249776936),
    (4.6, 4.600693370879999, 0.6):
        PulseFinal(
            0.17137010526164173,
            axis_x=0.99868709261645083,
            axis_y=-0.00000151672007476,
            axis_z=0.05122588251070076,
            angle=3.15236349995471565),
    (4.6, 4.60138674176, 0.6):
        PulseFinal(
            0.17134701517604703,
            axis_x=0.99478156056518496,
            axis_y=-0.00000312771055993,
            axis_z=0.10202767639083400,
            angle=3.17729225640700319),
    (4.6, 4.60277348352, 0.6):
        PulseFinal(
            0.17117583182146592,
            axis_x=0.98061444195699177,
            axis_y=-0.00000467954374518,
            axis_z=0.19594722810869061,
            angle=3.27369269693772713),
    (4.6, 4.60277348352, 0.6):
        PulseFinal(
            0.17117583182146592,
            axis_x=0.98061444195699177,
            axis_y=-0.00000467954374518,
            axis_z=0.19594722810869061,
            angle=3.27369269693772713),
    (4.6, 4.6055469670399996, 0.6):
        PulseFinal(
            0.17046240477265676,
            axis_x=0.94357818449845854,
            axis_y=0.00000022985645141,
            axis_z=0.33114982974258061,
            angle=3.62834888880173123),
    (4.6, 4.6110939340799995, 0.6):
        PulseFinal(
            0.16745349446464686,  # may be fitted
            axis_x=0.96522446677883700,
            axis_y=-0.00002065342042665,
            axis_z=0.26142250917804716,
            angle=4.67473425929162545),
    (4.6, 4.622187868159999, 0.6):
        PulseFinal(0.21353639154365156, is_good=False),
    (4.6, 4.64437573632, 0.6):
        PulseFinal(0.34549927267512398, is_good=False),
    (4.6, 4.68875147264, 0.6):
        PulseFinal(0.20415114334049114, is_good=False),
    (4.6, 4.777502945279999, 0.6):
        PulseFinal(0.22393824874729909, is_good=False),
    (4.6, 4.95500589056, 0.6):
        PulseFinal(0.21199432889687223, is_good=False),
    (4.6, 5.31001178112, 0.6):
        PulseFinal(0.20649346067476168, is_good=False),
    (4.6, 6.02002356224, 0.6):
        PulseFinal(0.20731739151143649, is_good=False),
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
        pulse_final = gaussian_x_pi_pulse_amplitudes[key]
        if not pulse_final.is_good or pulse_final.angle is not None:
            return
        amplitude = pulse_final.amplitude
    else:
        filename = f'simulation-x_gate-rabi-q_{qubit_frequency},d_{drive_frequency},o_{omega}'
        amplitude = find_rabi_rate(
            solver,
            x_pi_pulse,
            converter,
            amplitude_parameter,
            filename=filename)
        amplitude /= 2
        gaussian_x_pi_pulse_amplitudes[key] = PulseFinal(amplitude)
        if amplitude > 1.0:
            print('Rabi curve fitting failed.')
            print(f'{key}: PulseFinal({amplitude:.17f}, is_good=False),')
            return
        print(f'{key}: PulseFinal({amplitude:.17f}),')
        plot_evolution = True

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
    gaussian_x_pi_pulse_amplitudes[key] = PulseFinal(
        amplitude, axis_x=x, axis_y=y, axis_z=z, angle=r_data[-1])
    print(
        f'{key}: PulseFinal({amplitude:.17f},  axis_x={x:.17f}, axis_y={y:.17f}, axis_z={z:.17f}, angle={r_data[-1]:.17f}),'
    )

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
    qubit_frequency = 4.6
    omega = 0.6
    off_resonance_rate = 1e-9
    off_resonances = []
    while off_resonance_rate <= 0.1:
        off_resonances.append(off_resonance_rate * qubit_frequency)
        off_resonance_rate *= 2
    off_resonances = [-r for r in off_resonances[::-1]] + [0] + off_resonances
    for r in off_resonances:
        calibrate_x_pulse(
            qubit_frequency=qubit_frequency,
            drive_frequency=qubit_frequency + qubit_frequency * r,
            omega=omega)


if __name__ == '__main__':
    main()
