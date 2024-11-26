from qiskit.pulse import Schedule, Play, DriveChannel
from qiskit.pulse.library import Gaussian
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit import pulse
from pprint import pprint
import qiskit_ibm_runtime
from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder

from klepto.safe import no_cache
from klepto.archives import dir_archive, file_archive
from klepto.keymaps import hashmap
import json

from pathlib import Path

# import matplotlib.pyplot as plt

# Without setting the algorithm, klepto uses Python's built-in hash(), which
# doesn't generate the same value for the same input cross sessions. This
# unstable behavor leads the cache fails to load the results stored in the
# previous session.
stable_keymap = hashmap(algorithm='md5')

CACHE_DIR = Path('cache')

service = None


def patch_json():
    '''
    Some classes, like `DataBin`, in the job results always raise
    `NotImplementedError` when calling the `__setattr__` function. This design
    makes those classes not picklable. `klepto` doesn't provide the interface
    for customizing the dump process and the load process and only provide
    three protocol for dumping and loading, `pickle`, `json`, and `dill`. The
    official way to save and to load the job results is using json with
    `RuntimeEncoder` and `RuntimeDecoder`, so the simple workaround is to patch
    `json` and to use `json` as the protocol of `klepto`.
    '''
    original_json_dump = json.dump
    original_json_load = json.load

    def json_dump_for_job_results(*args, **kargs):
        return original_json_dump(*args, cls=RuntimeEncoder, **kargs)

    def json_load_for_job_results(*args, **kargs):
        return original_json_load(*args, cls=RuntimeDecoder, **kargs)

    json.dump = json_dump_for_job_results
    json.load = json_load_for_job_results


patch_json()


def get_service():
    global service
    if service is None:
        service = QiskitRuntimeService(
            instance='ibm-q/open/main',
            channel='ibm_quantum',
            token=
            'MY_TOKEN'
        )
    return service


class RetrieveJobError(Exception):
    """Exception raised when the requested job is not done."""

    def __init__(self, message,
                 job_id: qiskit_ibm_runtime.runtime_job_v2.JobStatus):
        super().__init__(message)
        self.job_id = job_id


@no_cache(cache=dir_archive(name=CACHE_DIR / 'job_results'),
          keymap=stable_keymap)
def retrieve_job_results(job_id):
    job = get_service().job(job_id)
    status = job.status()
    if status != "DONE":
        raise RetrieveJobError(f'The job is in the status: {status}', job_id)
    return job.result()


def handle_job(task_name):
    job_id_cache = no_cache(cache=file_archive(name=CACHE_DIR /
                                               f'{task_name}-job_id.pkl'),
                            keymap=stable_keymap)

    def decorator(submit_job_function):
        submit_job_function = job_id_cache(submit_job_function)

        # The job result is not picklable. Use the patched json a the protocol
        # as an workaround.
        @no_cache(
            cache=file_archive(name=CACHE_DIR / f'{task_name}-results.pkl',
                               protocol='json'),
            keymap=stable_keymap,
        )
        def wrapper():
            job_id = submit_job_function()
            return retrieve_job_results(job_id)

        return wrapper

    return decorator


def x_gaussian_pulse_with_different_amplitude_and_sigma():
    amplitudes = [0.5, 0.75, 1]
    sigmas = [16, 24, 32, 40, 48]
    combinations = [(s, a) for a in amplitudes for s in sigmas]

    def build_circuit_executing_gaussian_pulse(backend, sigma,
                                               amplitude) -> QuantumCircuit:
        with pulse.build(backend) as gate_pulse:
            # duration must be a multiple of 16 cycles
            # for gaussion duration should at least 64dt: (error: Pulse gaussian has too few samples (48 < 64) )
            # XXX: If the amplitude is half (0.5), does the pulse only rotate pi/2?
            microwave = Gaussian(duration=sigma * 6, amp=amplitude, sigma=sigma)
            pulse.play(microwave, pulse.drive_channel(0))

            # gate_pulse.draw()

            gate = Gate(name='custom_pulse',
                        label='CP',
                        num_qubits=1,
                        params=[])
            qc = QuantumCircuit(2, 2)

            # append the custom gate
            qc.append(gate, [0])
            qc.measure(0, 0)

            # define pulse of quantum gate
            qc.add_calibration(gate, [0], gate_pulse)

            # qc.draw('mpl')
            return qc

    @handle_job(x_gaussian_pulse_with_different_amplitude_and_sigma.__name__)
    def submit_job():
        backend = get_service().backend('ibm_sherbrooke')
        circuits = [
            build_circuit_executing_gaussian_pulse(backend, sigma, amplitude)
            for sigma, amplitude in combinations
        ]

        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return job.job_id()

    results = submit_job()
    for idx, result in enumerate(results):
        bitarray = result.data['c']
        sigma, amplitude = combinations[idx]
        print(f'sigma:{sigma} amplitude:{amplitude}')
        print(
            f'excited state proportion: {bitarray.get_int_counts()[1] / bitarray.num_shots}'
        )


def find_best_sigma_for_x_gate():
    sigmas = list(range(5, 101))

    def build_circuit_executing_gaussian_pulse(backend, sigma,
                                               amplitude) -> QuantumCircuit:
        with pulse.build(backend) as gate_pulse:
            # duration must be a multiple of 16 cycles
            # for gaussion duration should at least 64dt: (error: Pulse gaussian has too few samples (48 < 64) )
            # XXX: If the amplitude is half (0.5), does the pulse only rotate pi/2?
            duration = (sigma * 6 + 15) // 16 * 16
            duration = max(duration, 64)
            microwave = Gaussian(duration=duration, amp=amplitude, sigma=sigma)
            pulse.play(microwave, pulse.drive_channel(0))

            # gate_pulse.draw()

            gate = Gate(name='custom_pulse',
                        label='CP',
                        num_qubits=1,
                        params=[])
            qc = QuantumCircuit(2, 2)

            # append the custom gate
            qc.append(gate, [0])
            qc.measure(0, 0)

            # define pulse of quantum gate
            qc.add_calibration(gate, [0], gate_pulse)

            # qc.draw('mpl')
            return qc

    @handle_job(find_best_sigma_for_x_gate.__name__)
    def submit_job():
        backend = get_service().backend('ibm_sherbrooke')
        circuits = [
            build_circuit_executing_gaussian_pulse(backend, sigma, 1)
            for sigma in sigmas
        ]

        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return job.job_id()

    results = submit_job()
    for idx, result in enumerate(results):
        bitarray = result.data['c']
        proportion = bitarray.get_int_counts()[1] / bitarray.num_shots
        print(f'sigma:{sigmas[idx]} proportion:{proportion}')


# x_gaussian_pulse_with_different_amplitude_and_sigma()
find_best_sigma_for_x_gate()
