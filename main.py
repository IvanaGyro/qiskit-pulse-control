from qiskit.pulse.library import Gaussian, Drag
from qiskit_ibm_runtime import SamplerV2
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit import pulse
from qiskit.primitives import SamplerPubResult, BitArray

from dataclasses import dataclass

import matplotlib.pyplot as plt

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
        backend = meta.get_service().backend(self.backend)
        circuits = [
            self.build_circuit(backend, amplitude, sigma)
            for amplitude, sigma in self.gaussian_parameters
        ]
        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return unified_job.Job(job)

    def post_process(self, result):
        for circuit_result, (amplitude, sigma) in zip(result,
                                                      self.gaussian_parameters):
            bitarray: BitArray = circuit_result.data['c']
            population = bitarray.get_int_counts()[1] / bitarray.num_shots
            print(
                f'sigma:{sigma} amplitude:{amplitude} population:{population}')


@dataclass
class CompareDragAndGaussian(meta.QiskitTask):
    duration: int
    amplitude: float
    sigma: float
    beta: float
    angle: float = 0.0

    def submit_job(self):
        backend = meta.get_service().backend(self.backend)

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

        print(self)
        print(f'Gaussian: {get_population(result[0])}')
        print(f'Drag: {get_population(result[1])}')
        print(f'X gate: {get_population(result[2])}')


class ConcatTwoPulsesToMatchGranularity(meta.QiskitTask):
    '''
    This is invalid. Error message: Pulse not a multiple of 16 cycles
    '''

    def submit_job(self):
        backend = meta.get_service().backend('ibm_sherbrooke')
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


def main():
    meta.set_token('MY_TOKEN')

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

    sigmas = [31 + i * 0.1 for i in range(20)] + [52 + i * 0.1 for i in range(20)]
    GaussianPulseCalibration('ibm_sherbrooke', amplitudes=[1], sigmas=sigmas).run()

    # ConcatTwoPulsesToMatchGranularity().run()

if __name__ == '__main__':
    main()
