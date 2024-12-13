import functools

from qiskit import providers
from qiskit.primitives import primitive_job
from qiskit_ibm_runtime import runtime_job_v2


class Job:
    '''A wrapper for qiskit's job

    The packages in the qiskit's ecosystem may have its own job type. This class
    provide an universal interface for all job types.

    Attributes:
        status (`qiskit.providers.JobStatus`): status of the job
        id (str | None): The job ID provided by qiskit. `None` if the job is 
            created from a fake backend.
        result: job result if the job status is `JobStatus.DONE`, otherwise `None`
    '''

    @functools.singledispatchmethod
    def __init__(self, qiskit_job: runtime_job_v2.RuntimeJobV2):
        self.status = providers.JobStatus[qiskit_job.status()]
        self.id = qiskit_job.job_id()
        # expect the job is running or queueing
        self.result = None

    @__init__.register(primitive_job.PrimitiveJob)
    def _(self, qiskit_job: primitive_job.PrimitiveJob):
        '''Assume an instance of `PrimitiveJob` is always created from a fake
        backend.
        '''
        # Assume the job from the fake backend will finish within a reasonable
        # time.
        self.result = qiskit_job.result()
        self.status = qiskit_job.status()
        self.id = None
        assert (self.status == providers.JobStatus.DONE)

    @__init__.register(providers.JobStatus)
    def _(self, status: providers.JobStatus, id: str = None, result=None):
        self.status = status
        self.id = id
        self.result = result