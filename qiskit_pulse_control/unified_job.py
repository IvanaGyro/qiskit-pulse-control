import dataclasses
import functools
from typing import Any

from qiskit.primitives import primitive_job
from qiskit import providers
from qiskit_experiments import framework
from qiskit_ibm_runtime import runtime_job_v2

from qiskit_pulse_control import service


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
    def __init__(self, arg):
        raise TypeError(f'Not supported job type:{type(arg)}',)

    @__init__.register(runtime_job_v2.RuntimeJobV2)
    def _(self, qiskit_job: runtime_job_v2.RuntimeJobV2):
        # expect the job is running or queueing, so leave result as `None`
        self.__init__(
            providers.JobStatus[qiskit_job.status()], id=qiskit_job.job_id())
        self._runtime_job = qiskit_job

    @__init__.register(primitive_job.PrimitiveJob)
    def _(self, qiskit_job: primitive_job.PrimitiveJob):
        '''Assume an instance of `PrimitiveJob` is always created from a fake
        backend.
        '''
        # Assume the job from the fake backend will finish within a reasonable
        # time. After getting the result, the job status changes.
        result = qiskit_job.result()
        self.__init__(qiskit_job.status(), result=result)
        assert self.status == providers.JobStatus.DONE

    @__init__.register(providers.JobStatus)
    def _(self, status: providers.JobStatus, id: str = None, result=None):
        self.status = status
        self.id = id
        self.result = result
        self._runtime_job = None

    @property
    def runtime_job(self):
        '''The runtime job with the assigned job iD.'''
        if self._runtime_job is not None:
            return self._runtime_job
        if self.id is None:
            raise ValueError(
                'This Job is not a wrapper of a qiskit runtime job.')
        self._runtime_job = service.get_service().job(self.id)
        return self._runtime_job


class ExperimentJob:
    '''A wrapper for qiskit_experiments' experiment

    Different from qiskit_experiments' experiment, this class hold the
    information of jobs.

    Attributes:
        jobs (list[Job]):
            jobs wrapped by `Job` class.
        experiment (qiskit_experiments.framework.BaseExperiment):
            an instance of the subclass of `BaseExperiment`.
        analysis_result:
            The analysis result if the status of the experiment is done,
            otherwise `None`.
    '''

    @functools.singledispatchmethod
    def __init__(self, experiment_data: framework.ExperimentData):
        self.jobs = [Job(j) for j in experiment_data.jobs()]
        self.experiment = experiment_data.experiment
        self.analysis_result = None
        if all(job.status == providers.JobStatus.DONE for job in self.jobs):
            self.analysis_result = experiment_data.analysis_results()
        self._experiment_data = experiment_data

    @__init__.register(list)
    def _(self, jobs: list[Job],
          experiment: framework.BaseExperiment,
          analysis_result: framework.AnalysisResult | list[framework.AnalysisResult] = None):
        if not all(isinstance(job, Job) for job in jobs):
            raise ValueError('`jobs` should only contains `Job` instances.')
        self.jobs = jobs
        self.experiment = experiment
        self.analysis_result = analysis_result
        self._experiment_data = None

    @property
    def experiment_data(self) -> framework.ExperimentData:
        '''The cached experiment data or the recovered experiment data from the
        job results or the jobs.
        '''
        if self._experiment_data is not None:
            return self._experiment_data
        self._experiment_data = framework.ExperimentData(
            experiment=self.experiment)
        # TODO: check if the runtime jobs is done
        if all(job.status == providers.JobStatus.DONE for job in self.jobs):
            for job in self.jobs:
                # .add_data() cannot handle PrimitiveResult. The `job_id`
                # arugument is not important.
                self._experiment_data._add_result_data(job.result, '_')
            # update the analysis results
            self.experiment.analysis.run(
                self._experiment_data, replace_results=True)
            if self.analysis_result is None:
                self.analysis_result = self._experiment_data.analysis_results()
        else:
            for job in self.jobs:
                self._experiment_data.add_jobs(
                    [job.runtime_job for job in self.jobs])
        return self._experiment_data


@dataclasses.dataclass
class Result:
    value: Any


class JobResult(Result):
    '''Store the result from qiskit's runtime jobs
    
    This class helps the JSON encoder and decoder know how to encode and decode
    the result.
    '''
    pass


class ExperimentResult(Result):
    '''Store the analysis result from 
    `qiskit_experiments.framework.ExperimentData`

    This class helps the JSON encoder and decoder know how to encode and decode
    the result.
    '''
    pass
