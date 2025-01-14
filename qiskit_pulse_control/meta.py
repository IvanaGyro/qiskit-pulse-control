import abc
import dataclasses
import inspect
import pathlib
from typing import Union

import dill
from klepto.archives import dir_archive
from klepto.keymaps import hashmap, picklemap
from klepto.safe import no_cache
from qiskit_experiments import framework
import qiskit_ibm_runtime

from qiskit_pulse_control import service
from qiskit_pulse_control import unified_job

# Without setting the algorithm, klepto uses Python's built-in hash(), which
# doesn't generate the same value for the same input cross sessions. This
# unstable behavor leads the cache fails to load the results stored in the
# previous session.
#
# `picklemap()` will just call `__repr__()` if the serializer is not assigned to
# generate the key. This makes the key from the instance which doesn't belong to
# the dataclass different across sessions.
stable_keymap = picklemap(serializer=dill) + hashmap(algorithm='md5')

CACHE_DIR = pathlib.Path('cache')


def get_backend(backend_name: str):
    if backend_name.startswith('fake_'):
        return service.get_service('local').backend(backend_name)
    if backend_name.startswith('ibm_'):
        return service.get_service().backend(backend_name)
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
                # The method has wrapped by the cache of the super class.
                return method.__wrapped__

            def wrapper(*args, **kwargs):
                # XXX: Why we cannot just return `method`?
                return method(*args, **kwargs)

            return wrapper

        # We don't ignore `self` in the two cache below, so all the attributes
        # of the instance will be used to calculate the key for the cache. With
        # this practice, the key is always generated twice everytime `run()` is
        # called if the job result hasn't been cached. This is a trade-off to
        # save the time on calling `run()` after the result is cached.
        job_id_cache = no_cache(
            cache=dir_archive(
                name=CACHE_DIR / name / 'job_id', protocol='json'),
            keymap=stable_keymap)
        cls.submit_job = job_id_cache(unwrap_and_rewrap(cls.submit_job))

        # The job result is not picklable. As an workaround, use the patched
        # json as the protocol.
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
    def submit_job(
        self
    ) -> Union[str, qiskit_ibm_runtime.RuntimeJobV2, unified_job.Job,
               unified_job.ExperimentJob]:
        '''Create a qiskit job and send it to IBMQ

        You must implement this method.

        Returns (unified_job.Job | unified_job.ExperimentJob):
            A job converted from the jobs from the qiskit's ecosystem or a job
            wrapping an instance of `qiskit_experiments.framework.ExperimentData`.
            Returning a job id or a qiskit runtime job is deprecated. 
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

        def is_unified_job(any_object):
            return (isinstance(any_object, unified_job.Job) or
                    isinstance(any_object, unified_job.ExperimentJob))

        job_or_id_or_result = self.submit_job()
        if (isinstance(job_or_id_or_result, list) and
                all(is_unified_job(j) for j in job_or_id_or_result)):
            jobs = job_or_id_or_result
            return [self.__result_from_unified_job(j) for j in jobs]
        if is_unified_job(job_or_id_or_result):
            return self.__result_from_unified_job(job_or_id_or_result)

        # below is for backward compatibility
        if self.is_fake_backend():
            job_result = job_or_id_or_result
            return job_result
        job_id = job_or_id_or_result
        job = service.get_service().job(job_id)
        status = job.status()
        if status != "DONE":
            raise RetrieveJobError(f'The job is in the status: {status}',
                                   job_id)
        return job.result()

    def __result_from_unified_job(self, job: unified_job.Job |
                                  unified_job.ExperimentJob):
        if isinstance(job, unified_job.Job):
            if job.result is not None:
                return unified_job.JobResult(job.result)
            if not job.runtime_job.done():
                raise RetrieveJobError(
                    f'The job is in the status: {job.runtime_job.status()}',
                    job.id)
            # TODO: update `job.result` in the cache
            return unified_job.JobResult(job.runtime_job.result())

        if isinstance(job, unified_job.ExperimentJob):
            if job.analysis_result is not None:
                return unified_job.ExperimentResult(job.analysis_result)
            status = job.experiment_data.status()
            if not status == framework.ExperimentStatus.DONE:
                raise RetrieveJobError(
                    f'The experiment is in the status: {status}',
                    job.jobs[0].id)
            return unified_job.ExperimentResult(
                job.experiment_data.analysis_results())

        assert False

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
            return
        if (isinstance(result, list) and
                all(isinstance(r, unified_job.Result) for r in result)):
            self.post_process([r.value for r in result])
        elif isinstance(result, unified_job.Result):
            self.post_process(result.value)
        else:
            # TODO: This is for backward compability. Remove this.
            self.post_process(result)

    def is_fake_backend(self) -> bool:
        '''Check if the assigned backend is a fake backend

        Returns:
            `True` if the backend name starts with "fake\_", otherwise `False`.
        '''
        return self.backend.startswith('fake_')
