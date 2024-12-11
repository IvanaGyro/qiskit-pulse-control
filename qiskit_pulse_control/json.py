from collections.abc import Callable
import json
from typing import Any

from qiskit import providers
import qiskit_ibm_runtime

from qiskit_pulse_control import unified_job


class JobAndQiskitRuntimeEncoder(qiskit_ibm_runtime.RuntimeEncoder):
    '''JSON encoder for encoding `job.Job` and qiskit's job result.'''

    def default(self, any_object: Any) -> Any:
        if isinstance(any_object, unified_job.Job):
            return {
                '__type__': 'job',
                '__value__': {
                    'status':
                        any_object.status.name,
                    'id':
                        any_object.id,
                    'result':
                        super().default(any_object.result)
                        if any_object.result is not None else None
                }
            }
        return super().default(any_object)


class JobAndQiskitRuntimeDecoder(json.JSONDecoder):
    '''JSON decoder for decoding `job.Job` and qiskit's job result.'''

    def __init__(self, *, object_hook: Callable = None, **kwargs):
        self._previous_object_hook = object_hook
        self._qiskit_runtime_decoder = qiskit_ibm_runtime.RuntimeDecoder()
        super().__init__(object_hook=self.object_hook, **kwargs)

    def object_hook(self, any_object: Any) -> Any:
        if self._previous_object_hook is not None:
            any_object = self._previous_object_hook(any_object)
        if getattr(any_object, '__type__', None) == 'job':
            value = any_object['__value__']
            return unified_job.Job(
                providers.JobStatus[value['status']], value['id'], value['result'])
        return self._qiskit_runtime_decoder.object_hook(any_object)


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
