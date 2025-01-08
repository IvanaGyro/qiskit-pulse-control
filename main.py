from qiskit.pulse.library import Gaussian, Drag
from qiskit_ibm_runtime import SamplerV2
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit import pulse
import qutip
from matplotlib import animation
from qiskit.primitives import SamplerPubResult, BitArray
import qiskit_experiments
import numpy as np
from scipy import optimize
import math
import collections

from dataclasses import dataclass

import matplotlib.pyplot as plt

from qiskit_pulse_control import service
from qiskit_pulse_control import unified_job
from qiskit_pulse_control import meta


@dataclass
class GaussianPulseCalibration(meta.QiskitTask):
    amplitudes: list[float]
    sigmas: list[float]

    def __post_init__(self):
        self.gaussian_parameters = [
            (a, s) for a in self.amplitudes for s in self.sigmas
        ]

    def build_circuit(self, backend, amplitude, sigma):
        with pulse.build(backend) as gate_pulse:
            # duration must be a multiple of 16 cycles
            # duration should be at least 64dt and must be int
            duration = int((sigma * 6 + 15) // 16 * 16)
            duration = max(duration, 64)
            microwave = Gaussian(duration, amplitude, sigma)
            pulse.play(microwave, pulse.drive_channel(0))

            # gate_pulse.draw()
            # print(gate_pulse)

            gate = Gate(name='Gaussian', label='G', num_qubits=1, params=[])
            qc = QuantumCircuit(1, 1)
            qc.append(gate, [0])
            qc.add_calibration(gate, [0], gate_pulse)

            # Must do measurement to get the result from the sampler.
            qc.measure(0, 0)
            return qc

    def submit_job(self):
        backend = service.get_service().backend(self.backend)
        circuits = [
            self.build_circuit(backend, amplitude, sigma)
            for amplitude, sigma in self.gaussian_parameters
        ]
        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return unified_job.Job(job)

    def _plot_fitting_figure(self, amplitude: float, sigmas: list[float],
                             proportions: list[float]):
        points = sorted(zip(proportions, sigmas))
        guess_max = points[0][0]
        guess_min = points[-1][0]
        guess_amplitude = (guess_max - guess_min) / 2
        guess_base = (guess_max + guess_min) / 2
        # XXX: not a good guess
        guess_period = points[1][1] - points[0][1]
        guess_period = 21.0
        cos_curve = lambda x, amplitude, base, period, phase: amplitude * np.cos(
            2 * math.pi * x / period + phase) + base
        fit_result = optimize.curve_fit(
            cos_curve,
            sigmas,
            proportions,
            p0=[guess_amplitude, guess_base, guess_period, 0],
            maxfev=100000,
        )

        curve_parameters, pcov = fit_result
        errors = np.sqrt(np.diag(pcov))

        x_fit = np.linspace(min(sigmas), max(sigmas), 100)
        y_fit = cos_curve(x_fit, *curve_parameters)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.scatter(sigmas, proportions, color='blue', label='Data Points')
        plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
        plt.title(rf'Gaussian Pulse, $amplitude = {amplitude}$')
        plt.xlabel('Sigma')
        plt.ylabel(r'Proportion of $|1\rangle$')

        ax = plt.gca()
        ax.text(
            0.5,
            -0.15,
            rf'period = {curve_parameters[2]:.3f} $\pm$ {errors[2]:.3f}',
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='center',
            transform=ax.transAxes,
            bbox={
                'boxstyle': 'square',
                'facecolor': (0, 0, 0, 0),
            })

        plt.tight_layout()
        plt.show()

    def post_process(self, result):
        grouped_points = collections.defaultdict(lambda: {
            'sigma': [],
            'proportion': []
        })
        for circuit_result, (amplitude, sigma) in zip(result,
                                                      self.gaussian_parameters):
            bitarray: BitArray = circuit_result.data['c']
            proportion = bitarray.get_int_counts()[1] / bitarray.num_shots
            grouped_points[amplitude]['sigma'].append(sigma)
            grouped_points[amplitude]['proportion'].append(proportion)
            print(
                f'sigma:{sigma} amplitude:{amplitude} proportion:{proportion}')
        if len(self.amplitudes) >= len(self.sigmas):
            # only plot the figures for the task scanning sigmas
            return
        for amplitude, points in grouped_points.items():
            self._plot_fitting_figure(amplitude, points['sigma'],
                                      points['proportion'])


@dataclass
class RabiScanAmplitudes(meta.QiskitTask):
    amplitudes: list[float]
    sigma: float
    physical_qubit: int

    def build_circuit(self, backend, amplitude, sigma):
        with pulse.build(backend) as gate_pulse:
            # duration must be a multiple of 16 cycles
            # duration should be at least 64dt and must be int
            duration = int((sigma * 6 + 15) // 16 * 16)
            duration = max(duration, 64)
            microwave = Gaussian(duration, amplitude, sigma)
            pulse.play(microwave, pulse.drive_channel(0))

            # gate_pulse.draw()
            # print(gate_pulse)

            gate = Gate(name='Gaussian', label='G', num_qubits=1, params=[])
            qc = QuantumCircuit(1, 1)
            qc.append(gate, [0])
            # XXX: Does assigning physical qubit here work? qiskit_experiments
            # does this:
            # https://qiskit-community.github.io/qiskit-experiments/_modules/qiskit_experiments/library/characterization/rabi.html#Rabi
            qc.add_calibration(gate, (self.physical_qubit,), gate_pulse)

            # Must do measurement to get the result from the sampler.
            qc.measure(0, 0)
            return qc

    def submit_job(self):
        backend = service.get_service().backend(self.backend)
        circuits = [
            self.build_circuit(backend, amplitude, self.sigma)
            for amplitude in self.amplitudes
        ]
        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return unified_job.Job(job)

    def post_process(self, result):
        proportions = []
        amplitudes = []
        for circuit_result, amplitude in zip(result, self.amplitudes):
            bitarray: BitArray = circuit_result.data['c']
            proportion = bitarray.get_int_counts()[1] / bitarray.num_shots
            proportions.append(proportion)
            amplitudes.append(amplitude)

        points = sorted(zip(proportions, amplitudes))
        guess_max = points[0][0]
        guess_min = points[-1][0]
        guess_amplitude = (guess_max - guess_min) / 2
        guess_base = (guess_max + guess_min) / 2
        # XXX: not a good guess
        guess_period = points[1][1] - points[0][1]
        guess_period = 0.4
        cos_curve = lambda x, amplitude, base, period, phase: amplitude * np.cos(
            2 * math.pi * x / period + phase) + base
        fit_result = optimize.curve_fit(
            cos_curve,
            amplitudes,
            proportions,
            p0=[guess_amplitude, guess_base, guess_period, 0])
        curve_parameters, pcov = fit_result
        errors = np.sqrt(np.diag(pcov))

        x_fit = np.linspace(min(self.amplitudes), max(self.amplitudes), 100)
        y_fit = cos_curve(x_fit, *curve_parameters)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.scatter(amplitudes, proportions, color='blue', label='Data Points')
        plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
        plt.title(rf'Gaussian Pulse, $\sigma = {self.sigma}$')
        plt.xlabel('Amplitude')
        plt.ylabel(r'Proportion of $|1\rangle$')

        ax = plt.gca()
        ax.text(
            0.5,
            -0.15,
            rf'period = {curve_parameters[2]:.3f} $\pm$ {errors[2]:.3f}',
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='center',
            transform=ax.transAxes,
            bbox={
                'boxstyle': 'square',
                'facecolor': (0, 0, 0, 0),
            })

        plt.tight_layout()
        plt.show()


@dataclass
class CompareDragAndGaussian(meta.QiskitTask):
    duration: int
    amplitude: float
    sigma: float
    beta: float
    angle: float = 0.0

    def submit_job(self):
        backend = meta.get_backend(self.backend)

        circuit_with_gaussion = QuantumCircuit(1, 1)
        with pulse.build(backend) as gate_pulse:
            waveform = Gaussian(self.duration, self.amplitude, self.sigma,
                                self.angle)
            pulse.play(waveform, pulse.drive_channel(0))

            # gate_pulse.draw()
            gate = Gate(name='Gaussian', label='G', num_qubits=1, params=[])
            circuit_with_gaussion.append(gate, [0])
            circuit_with_gaussion.add_calibration(gate, [0], gate_pulse)
        circuit_with_gaussion.measure(0, 0)

        circuit_with_drag = QuantumCircuit(1, 1)
        with pulse.build(backend) as gate_pulse:
            waveform = Drag(self.duration, self.amplitude, self.sigma,
                            self.beta, self.angle)
            pulse.play(waveform, pulse.drive_channel(0))

            # gate_pulse.draw()
            gate = Gate(name='Drag', label='D', num_qubits=1, params=[])
            circuit_with_drag.append(gate, [0])
            circuit_with_drag.add_calibration(gate, [0], gate_pulse)
        circuit_with_drag.measure(0, 0)

        circuit_with_x = QuantumCircuit(1, 1)
        circuit_with_x.x(0)
        circuit_with_x.measure(0, 0)

        circuits = [circuit_with_gaussion, circuit_with_drag, circuit_with_x]
        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return unified_job.Job(job)

    def post_process(self, result):

        def get_population(circuit_result: SamplerPubResult):
            bitarray: BitArray = circuit_result.data['c']
            return bitarray.get_int_counts()[1] / bitarray.num_shots

        print(f'Gaussian: {get_population(result[0])}')
        print(f'Drag: {get_population(result[1])}')
        print(f'X gate: {get_population(result[2])}')


class ConcatTwoPulsesToMatchGranularity(meta.QiskitTask):
    '''
    This is invalid. Error message: Pulse not a multiple of 16 cycles
    '''

    def submit_job(self):
        backend = service.get_service().backend('ibm_sherbrooke')
        qc = QuantumCircuit(1, 1)
        with pulse.build(backend) as gate_pulse:
            pulse1 = Gaussian(duration=42, amp=1.0, sigma=32)
            pulse2 = Gaussian(duration=38, amp=1.0, sigma=32)
            pulse.play(pulse1, pulse.drive_channel(0))
            pulse.play(pulse2, pulse.drive_channel(0))

            # gate_pulse.draw()

            gate = Gate(
                name='Two Gaussian', label='GG', num_qubits=1, params=[])
            qc.append(gate, [0])
            qc.add_calibration(gate, [0], gate_pulse)

            # qc.draw('mpl')
        circuits = [qc]

        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return unified_job.Job(job)

    def post_process(self, result):
        print('Can concat two pulses to match the granularity.')


@dataclass
class TomographyOfXateSeries(meta.QiskitTask):
    x_gate_count: int
    physical_qubit: int

    def submit_job(self):
        backend = meta.get_backend(self.backend)
        circuit = QuantumCircuit(1, 1)
        for _ in range(self.x_gate_count):
            circuit.x(0)
        experiment_data = qiskit_experiments.library.StateTomography(
            circuit, physical_qubits=(self.physical_qubit,)).run(backend)
        print(f'submit TomographyOfXateSeries x_gate_count:{self.x_gate_count}')
        return unified_job.ExperimentJob(experiment_data)

    def post_process(self,
                     result: list[qiskit_experiments.framework.AnalysisResult]):
        for analysis_result in result:
            if analysis_result.name == 'state':
                density_matrix: np.typing.NDArray[
                    np.float64] = analysis_result.value.data
                break
        x = 2 * np.real(density_matrix[1][0])
        y = 2 * np.imag(density_matrix[1][0])
        z = 2 * np.real(density_matrix[0][0]) - 1
        print(f'Bloch vector:{(x, y, z)}')


@dataclass
class TrajectoryOfXgateSeries(meta.QiskitTask):
    x_gate_counts: list[int]
    physical_qubit: int

    def submit_job(self):
        xx_seriese_tomography_jobs = [
            TomographyOfXateSeries(self.backend, c,
                                   self.physical_qubit).submit_job()
            for c in self.x_gate_counts
        ]
        return xx_seriese_tomography_jobs

    def post_process(
            self,
            result: list[list[qiskit_experiments.framework.AnalysisResult]]):
        xs = []
        ys = []
        zs = []
        rs = []
        for analysis_results in result:
            density_matrix: np.typing.NDArray[np.float64] = None
            for analysis_result in analysis_results:
                if analysis_result.name == 'state':
                    density_matrix = analysis_result.value.data
                    break
            if density_matrix is None:
                raise ValueError(
                    'The analysis result does not contain the density matrix.')
            x = 2 * np.real(density_matrix[1][0])
            y = 2 * np.imag(density_matrix[1][0])
            z = 2 * np.real(density_matrix[0][0]) - 1
            xs.append(x)
            ys.append(y)
            zs.append(z)
            rs.append((x**2 + y**2 + z**2)**0.5)

        fig = plt.figure()
        ax = fig.add_subplot(azim=-40, elev=30, projection='3d')
        sphere = qutip.Bloch(axes=ax)

        def animate(i):
            sphere.clear()
            sphere.add_vectors([xs[i], ys[i], zs[i]], ['r'])
            sphere.add_points([xs[:i + 1], ys[:i + 1], zs[:i + 1]])
            sphere.make_sphere()
            return ax

        ani = animation.FuncAnimation(
            fig, animate, range(len(xs)), blit=False, repeat=False)
        ani.save(
            f'x_series_trajectory-{self.x_gate_counts[0]}_{self.x_gate_counts[-1]}.gif',
            fps=12)

        # Create the line plot
        plt.plot(self.x_gate_counts, rs)

        # Add titles and labels for clarity
        plt.title('Trajectory Of X-gate Series')
        plt.xlabel('x-gate count')
        plt.ylabel('bloch vector length')

        # Display the plot
        plt.show()


@dataclass
class RamseyCharacterization(meta.QiskitTask):
    delays: list[float]
    frequency_shift: float
    physical_qubit: int

    def submit_job(self):
        backend = meta.get_backend(self.backend)
        experiment = qiskit_experiments.library.T2Ramsey(
            physical_qubits=(self.physical_qubit,),
            delays=self.delays,
            backend=backend,
            osc_freq=self.frequency_shift)
        return unified_job.ExperimentJob(experiment.run())

    def post_process(self,
                     result: list[qiskit_experiments.framework.AnalysisResult]):
        for r in result:
            print(r)
        print(f'frequency shift:{result[1].value:.10f} Hz')

def main():
    service.set_token('MY_TOKEN')

    # GaussianPulseCalibration('ibm_sherbrooke',
    #                          amplitudes=[0.5, 0.75, 1],
    #                          sigmas=[16, 24, 32, 40, 48]).run()

    # GaussianPulseCalibration('ibm_sherbrooke',
    #                          amplitudes=[1],
    #                          sigmas=list(range(5, 101))).run()

    # GaussianPulseCalibration('ibm_sherbrooke',
    #                          amplitudes=[1],
    #                          sigmas=[20 + i * 0.1 for i in range(20)]).run()

    # CompareDragAndGaussian('ibm_sherbrooke',
    #                        duration=256,
    #                        amplitude=0.2002363461992037,
    #                        sigma=64,
    #                        beta=3.279359125685733,
    #                        angle=0.0).run()

    # sigmas = [31 + i * 0.1 for i in range(20)
    #          ] + [52 + i * 0.1 for i in range(20)]
    # GaussianPulseCalibration(
    #     'ibm_sherbrooke', amplitudes=[1], sigmas=sigmas).run()

    # ConcatTwoPulsesToMatchGranularity().run()

    # TomographyOfXateSeries(
    #     'fake_sherbrooke', x_gate_count=5, physical_qubit=5).run()

    # RabiScanAmplitudes(
    #     'ibm_sherbrooke',
    #     amplitudes=[0.03 * i for i in range(1, 26)],
    #     sigma=52.,
    #     physical_qubit=0).run()
    # RabiScanAmplitudes(
    #     'ibm_sherbrooke',
    #     amplitudes=[0.03 * i for i in range(1, 26)],
    #     sigma=64.,
    #     physical_qubit=0).run()
    # RabiScanAmplitudes(
    #     'ibm_sherbrooke',
    #     amplitudes=[0.03 * i for i in range(1, 26)],
    #     sigma=47.,
    #     physical_qubit=0).run()

    # RamseyCharacterization(
    #     'ibm_sherbrooke',
    #     delays=list(np.arange(1.00e-6, 50.0e-6, 2.00e-6)),
    #     frequency_shift=0.0,
    #     physical_qubit=0,
    # ).run()

    # RamseyCharacterization(
    #     'ibm_sherbrooke',
    #     delays=list(np.arange(1.00e-6, 50.0e-6, 5.00e-7)),
    #     frequency_shift=0.0,
    #     physical_qubit=0,
    # ).run()

    # TrajectoryOfXgateSeries(
    #     'fake_sherbrooke',
    #     x_gate_counts=list(range(0, 25, 1)),
    #     physical_qubit=1).run()

    # TrajectoryOfXgateSeries(
    #     'fake_sherbrooke',
    #     x_gate_counts=list(range(0, 1001, 2)),
    #     physical_qubit=1).run()

    plt.show()


if __name__ == '__main__':
    main()
