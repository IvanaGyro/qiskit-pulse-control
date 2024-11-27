from qiskit.pulse import Schedule, Play, DriveChannel
from qiskit.pulse.library import Gaussian, Drag
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit import pulse
from pprint import pprint
import qiskit_ibm_runtime
from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder
from qiskit.primitives import SamplerPubResult, BitArray

from klepto.safe import no_cache
from klepto.archives import dir_archive, file_archive
from klepto.keymaps import hashmap, picklemap
import json

from pathlib import Path
from dataclasses import dataclass

import abc
import inspect

import matplotlib.pyplot as plt

# Without setting the algorithm, klepto uses Python's built-in hash(), which
# doesn't generate the same value for the same input cross sessions. This
# unstable behavor leads the cache fails to load the results stored in the
# previous session.
stable_keymap = picklemap() + hashmap(algorithm='md5')

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


class _QiskitTaskMeta(abc.ABCMeta):
    ''' A Meta class for modifying the classes inherited the `QiskitTask` class.

    Wrap `submit_job()` and `__retrieve_result()` to save their return values
    into the disk.
    '''

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        if name == 'QiskitTask':
            # `QiskitTask` class cannot be initialized, so there is no need to
            # decorate its methods.
            return cls

        is_method = lambda key: inspect.isfunction(getattr(cls, key, None))
        if (not is_method('submit_job') or
                not is_method('_QiskitTask__retrieve_result')):

            raise TypeError('_QiskitTaskMeta can only be apply on '
                            'the subclass of QiskitTask.')

        def unwrap_and_rewrap(method):
            '''Get the original method and remove `self` from its signature.

            If the class inherits more than one level, its methods are wrapped
            by the cache of the super class. The cache should be added on the
            original method instead of the wrapped method.

            The default `self` argument of the method will be passed to the
            keymap as a keyword argument. This makes the keymap receives
            multiple `self` argument, including the default `self` argument of
            the keymap, and raises error. As an workaround, we wrap the method
            with `wrapper(*args, **kwargs)` before wrapping it with the cache.

            Args:
                method (function):
                    The method of the class.
            '''
            if hasattr(method, '__wrapped__'):
                # The method has wrapped with the cache of the super class
                # should have been wrapped with the wrapper below.
                return method.__wrapped__

            def wrapper(*args, **kwargs):
                return method(*args, **kwargs)

            return wrapper

        # We don't ignore `self` in the two cache below, so all the attributes
        # of the instance will be used to calculate the key for the cache. With
        # this practive, the key is always generated twice everytime `run()` is
        # called if the job result hasn't been cached. This is a trade-off to
        # save the time on calling `run()` after the result is cached.
        job_id_cache = no_cache(cache=dir_archive(name=CACHE_DIR / name /
                                                  'job_id'),
                                keymap=stable_keymap)
        cls.submit_job = job_id_cache(unwrap_and_rewrap(cls.submit_job))

        # The job result is not picklable. Use the patched json a the protocol
        # as an workaround.
        result_archive = dir_archive(name=CACHE_DIR / name / 'result',
                                     protocol='json')
        result_cache = no_cache(cache=result_archive, keymap=stable_keymap)
        cls._QiskitTask__retrieve_result = result_cache(
            unwrap_and_rewrap(cls._QiskitTask__retrieve_result))
        return cls


class QiskitTask(metaclass=_QiskitTaskMeta):
    ''' A helper class for handling the Qiskit tasks.

    Implement `submit_job()` and `post_process()` in the subclass, and call
    `run()` on the instance of the subclass. This class will help you submit the
    job, save the result, and execute the post process.

    See `class CompareDragAndGaussian` for the example.
    '''

    @abc.abstractmethod
    def submit_job(self) -> str:
        '''Create a qiskit job and send it to IBMQ

        You must implement this method.

        Returns:
            A string of the job id.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def post_process(self, result):
        '''Process the result returned by IMBQ

        You must implement this method. Put the code for printing the result
        or drawing the charts in this method.

        Args:
            result:
                The result of the job. The contnet in the result depends on the
                job submitted in `submit_job()`.
        '''
        raise NotImplementedError()

    def __retrieve_result(self):
        '''Submit the job and try to retrieve the result.

        This method is decorated by the metaclass. The result will be cached if
        this method successfully gets the result.
        '''
        job_id = self.submit_job()
        job = get_service().job(job_id)
        status = job.status()
        if status != "DONE":
            raise RetrieveJobError(f'The job is in the status: {status}',
                                   job_id)
        return job.result()

    def run(self):
        '''Run the task

        You should not implement this method for the subclass.

        Call this method to submit the job and process the job result. The job
        result will be cached in the cache folder `CACHE_DIR`. Calling this
        method of the instance with the same attribute values and the same class
        name will not run `submit_job()` again. Instead this method will read
        the cached job id and try to retrieve the job result if the job result
        has not been cached before. If the job result is retrieved successfully
        or has been cached before, this method will call `post_process()` with
        the job result as the parameter.

        Raises:
            RetrieveJobError: An error occured when the job status is not DONE.
        '''
        try:
            result = self.__retrieve_result()
        except RetrieveJobError as e:
            print(e)
        else:
            self.post_process(result)


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


@dataclass
class GaussianPulseCalibration(QiskitTask):
    backend: str
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
        backend = get_service().backend(self.backend)
        circuits = [
            self.build_circuit(backend, amplitude, sigma)
            for amplitude, sigma in self.gaussian_parameters
        ]
        sampler = SamplerV2(mode=backend)
        job = sampler.run(circuits)
        return job.job_id()

    def post_process(self, result):
        for circuit_result, (amplitude, sigma) in zip(result,
                                                      self.gaussian_parameters):
            bitarray: BitArray = circuit_result.data['c']
            population = bitarray.get_int_counts()[1] / bitarray.num_shots
            print(
                f'sigma:{sigma} amplitude:{amplitude} population:{population}')


@dataclass
class CompareDragAndGaussian(QiskitTask):
    backend: str
    duration: int
    amplitude: float
    sigma: float
    beta: float
    angle: float = 0.0

    def submit_job(self):
        backend = get_service().backend(self.backend)

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
        return job.job_id()

    def post_process(self, result):

        def get_population(circuit_result: SamplerPubResult):
            bitarray: BitArray = circuit_result.data['c']
            return bitarray.get_int_counts()[1] / bitarray.num_shots

        print(self)
        print(f'Gaussian: {get_population(result[0])}')
        print(f'Drag: {get_population(result[1])}')
        print(f'X gate: {get_population(result[2])}')



GaussianPulseCalibration('ibm_sherbrooke',
                         amplitudes=[0.5, 0.75, 1],
                         sigmas=[16, 24, 32, 40, 48]).run()

GaussianPulseCalibration('ibm_sherbrooke',
                         amplitudes=[1],
                         sigmas=list(range(5, 101))).run()

GaussianPulseCalibration('ibm_sherbrooke',
                         amplitudes=[1],
                         sigmas=[20 + i * 0.1 for i in range(20)]).run()

CompareDragAndGaussian('ibm_sherbrooke',
                       duration=256,
                       amplitude=0.2002363461992037,
                       sigma=64,
                       beta=3.279359125685733,
                       angle=0.0).run()
