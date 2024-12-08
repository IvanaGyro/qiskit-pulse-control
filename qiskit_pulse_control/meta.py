import abc
import dataclasses
import inspect
import json
from pathlib import Path

import dill
import qiskit_ibm_runtime
from klepto.archives import dir_archive
from klepto.keymaps import hashmap, picklemap
from klepto.safe import no_cache
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, RuntimeDecoder

# Without setting the algorithm, klepto uses Python's built-in hash(), which
# doesn't generate the same value for the same input cross sessions. This
# unstable behavor leads the cache fails to load the results stored in the
# previous session.
#
# `picklemap()` will just call `__repr__()` if the serializer is not assigned to
# generate the key. This makes the key from the instance which doesn't belong to
# the dataclass different across sessions.
stable_keymap = picklemap(serializer=dill) + hashmap(algorithm='md5')

CACHE_DIR = Path('cache')

service = None
_token = None


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
        if 'cls' in kargs:
            del kargs['cls']
        return original_json_dump(*args, cls=RuntimeEncoder, **kargs)

    def json_load_for_job_results(*args, **kargs):
        if 'cls' in kargs:
            del kargs['cls']
        return original_json_load(*args, cls=RuntimeDecoder, **kargs)

    json.dump = json_dump_for_job_results
    json.load = json_load_for_job_results


patch_json()


def set_token(token: str):
    global _token
    _token = token


def get_service(channel='ibm_quantum'):
    if channel == 'local':
        # Getting a local service is quick, so there is no need to cache.
        return QiskitRuntimeService(channel='local')
    global service
    if service is None:
        if _token is None:
            raise RuntimeError(
                'Please call set_token() before getting a service.')
        service = QiskitRuntimeService(
            instance='ibm-q/open/main', channel=channel, token=_token)
    return service


def get_backend(backend_name: str):
    if backend_name.startswith('fake_'):
        return get_service('local').backend(backend_name)
    if backend_name.startswith('ibm_'):
        return get_service().backend(backend_name)
    raise ValueError(f'invalid backend name:{backend_name}')


class RetrieveJobError(Exception):
    """Exception raised when the requested job is not done."""

    def __init__(self, message,
                 job_id: qiskit_ibm_runtime.runtime_job_v2.JobStatus):
        super().__init__(message)
        self.job_id = job_id


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
        job_id_cache = no_cache(
            cache=dir_archive(
                name=CACHE_DIR / name / 'job_id', protocol='json'),
            keymap=stable_keymap)
        cls.submit_job = job_id_cache(unwrap_and_rewrap(cls.submit_job))

        # The job result is not picklable. Use the patched json a the protocol
        # as an workaround.
        result_archive = dir_archive(
            name=CACHE_DIR / name / 'result', protocol='json')
        result_cache = no_cache(cache=result_archive, keymap=stable_keymap)
        cls._QiskitTask__retrieve_result = result_cache(
            unwrap_and_rewrap(cls._QiskitTask__retrieve_result))
        return cls


@dataclasses.dataclass
class QiskitTask(metaclass=_QiskitTaskMeta):
    ''' A helper class for handling the Qiskit tasks.

    Implement `submit_job()` and `post_process()` in the subclass, and call
    `run()` on the instance of the subclass. This class will help you submit the
    job, save the result, and execute the post process.

    See `class CompareDragAndGaussian` for the example.

    Attributes:
        backend (str):
            The name of the backend for running the job. The prefix
            of the fake backend is "fake" instead of "ibm", for example
            "fake_sherbrooke".
    '''
    backend: str

    @abc.abstractmethod
    def submit_job(self) -> str | qiskit_ibm_runtime.RuntimeJobV2:
        '''Create a qiskit job and send it to IBMQ

        You must implement this method.

        Returns:
            A string of the job id when the backend is an IBM device, otherwise
            the result of the job when the backend is a fake backend.
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
        job_id_or_result = self.submit_job()
        if self.is_fake_backend():
            job_result = job_id_or_result
            return job_result
        job_id = job_id_or_result
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
            # TODO: Show the error message of the job which status is "ERROR".
            print(e)
        else:
            self.post_process(result)

    def is_fake_backend(self) -> bool:
        '''Check if the assigned backend is a fake backend

        Returns:
            `True` if the backend name starts with "fake\_", otherwise `False`.
        '''
        return self.backend.startswith('fake_')
