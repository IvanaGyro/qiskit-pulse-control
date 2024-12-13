import qiskit_ibm_runtime

_service = None
_token = None


def set_token(token: str):
    global _token
    _token = token


def get_service(channel='ibm_quantum'):
    if channel == 'local':
        # Getting a local service is quick, so there is no need to cache.
        return qiskit_ibm_runtime.QiskitRuntimeService(channel='local')
    global _service
    if _service is None:
        if _token is None:
            raise RuntimeError(
                'Please call set_token() before getting a service.')
        _service = qiskit_ibm_runtime.QiskitRuntimeService(
            instance='ibm-q/open/main', channel=channel, token=_token)
    return _service