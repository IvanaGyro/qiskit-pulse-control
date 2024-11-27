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
