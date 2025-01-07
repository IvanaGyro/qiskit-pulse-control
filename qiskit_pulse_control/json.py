from collections.abc import Callable
import collections
import json
from typing import Any

from qiskit import providers
from qiskit_experiments import framework
import qiskit_ibm_runtime

from qiskit_pulse_control import unified_job


class JobAndQiskitRuntimeEncoder(json.JSONEncoder):
    '''JSON encoder for encoding `job.Job` and qiskit's job result.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qiskit_runtime_encoder = qiskit_ibm_runtime.RuntimeEncoder()
        self._experiment_encoder = framework.ExperimentEncoder()

    def default(self, any_object: Any) -> Any:
        if isinstance(any_object, unified_job.Job):
            return self._encode_job(any_object)
        if isinstance(any_object, unified_job.ExperimentJob):
            # fix https://github.com/qiskit-community/qiskit-experiments/issues/1508
            copied_experiment = any_object.experiment.copy()
            experiment_kwargs = getattr(copied_experiment, '__init_kwargs__',
                                        collections.OrderedDict())
            if 'backend' in experiment_kwargs:
                del experiment_kwargs['backend']

            encoded_value = {
                'jobs': [self._encode_job(job) for job in any_object.jobs],
                'experiment':
                    self._experiment_encoder.encode(any_object.experiment),
            }
            if any_object.analysis_result is not None:
                analysis_result = self._experiment_encoder.encode(
                    any_object.analysis_result)
                encoded_value['analysis_result'] = analysis_result
            return {'__type__': 'experiment', '__value__': encoded_value}
        if isinstance(any_object, unified_job.JobResult):
            return {
                '__type__':
                    'JobResult',
                '__value__':
                    self._qiskit_runtime_encoder.encode(any_object.value),
            }
        if isinstance(any_object, unified_job.ExperimentResult):
            return {
                '__type__': 'ExperimentResult',
                '__value__': self._experiment_encoder.encode(any_object.value),
            }
        return self._qiskit_runtime_encoder.default(any_object)

    def _encode_job(self, job: unified_job.Job):
        encoded_value = {
            'status': job.status.name,
        }
        if job.id is not None:
            encoded_value['id'] = job.id
        if job.result is not None:
            encoded_value['result'] = self._qiskit_runtime_encoder.encode(
                job.result)
        return {'__type__': 'job', '__value__': encoded_value}


class JobAndQiskitRuntimeDecoder(json.JSONDecoder):
    '''JSON decoder for decoding `job.Job` and qiskit's job result.'''

    def __init__(self, *, object_hook: Callable = None, **kwargs):
        self._previous_object_hook = object_hook
        self._qiskit_runtime_decoder = qiskit_ibm_runtime.RuntimeDecoder()
        self._experiment_decoder = framework.ExperimentDecoder()
        super().__init__(object_hook=self.object_hook, **kwargs)

    def object_hook(self, any_object: Any) -> Any:
        if self._previous_object_hook is not None:
            any_object = self._previous_object_hook(any_object)
        if not isinstance(any_object, dict) or '__type__' not in any_object:
            return any_object

        type_name = any_object['__type__']
        value: dict = any_object['__value__']
        if type_name == 'job':
            return self._decode_job(value)
        if type_name == 'experiment':
            jobs = value['jobs']
            experiment = self._experiment_decoder.decode(value['experiment'])
            analysis_result = None
            if 'analysis_result' in value:
                analysis_result = self._experiment_decoder.decode(
                    value['analysis_result'])
            return unified_job.ExperimentJob(jobs, experiment, analysis_result)
        if type_name == 'JobResult':
            return unified_job.JobResult(
                self._qiskit_runtime_decoder.decode(value))
        if type_name == 'ExperimentResult':
            return unified_job.ExperimentResult(
                self._experiment_decoder.decode(value))
        # TODO: backward compability, return `any_object` instead
        return self._qiskit_runtime_decoder.object_hook(any_object)

    def _decode_job(self, encoded_job: dict) -> unified_job.Job:
        result = encoded_job.get('result')
        if isinstance(result, str):
            # TODO: backward compability, encoded_job['result'] should always be
            # a string
            result = self._qiskit_runtime_decoder.decode(result)
        return unified_job.Job(providers.JobStatus[encoded_job['status']],
                               encoded_job.get('id'), result)


def patch_json():
    '''
    Some classes, like `DataBin`, in the job results always raise
    `NotImplementedError` when calling the `__setattr__` function. This design
    makes those classes not picklable. `klepto` doesn't provide the interface
    for customizing the dump process and the load process and only provide
    three protocol for dumping and loading, `pickle`, `json`, and `dill`. The
    official way to save and to load the job results is using json with
    `qiskit_ibm_runtime.RuntimeEncoder` and `qiskit_ibm_runtime.RuntimeDecoder`,
    so the simple workaround is to patch `json` and to use `json` as the
    protocol of `klepto`.
    '''
    patched = hasattr(json, 'original_json_dump')
    if patched:
        return

    json.original_json_dump = json.dump
    json.original_json_load = json.load

    def json_dump_for_job_results(*args,
                                  cls=JobAndQiskitRuntimeEncoder,
                                  **kargs):
        return json.original_json_dump(*args, cls=cls, **kargs)

    def json_load_for_job_results(*args,
                                  cls=JobAndQiskitRuntimeDecoder,
                                  **kargs):
        return json.original_json_load(*args, cls=cls, **kargs)

    json.dump = json_dump_for_job_results
    json.load = json_load_for_job_results
