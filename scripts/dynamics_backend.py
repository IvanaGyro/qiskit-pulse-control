from qiskit import circuit
from qiskit_dynamics import DynamicsBackend
from qiskit.result import models
import numpy
from qiskit import primitives
import qiskit_ibm_runtime
from qiskit import transpiler

from qiskit_pulse_control import meta
from qiskit_pulse_control import service


def patch_experiment_result_data():
    '''An workaround for https://github.com/qiskit-community/qiskit-dynamics/pull/367'''
    ExperimentResultData = models.ExperimentResultData
    original_init = ExperimentResultData.__init__

    def new_init(self,
                 counts=None,
                 snapshots=None,
                 memory=None,
                 statevector=None,
                 unitary=None,
                 **kwargs):
        if isinstance(memory, numpy.ndarray):
            # '010' => '0x2'
            memory = [hex(int(byte, 2)) for byte in memory]
        original_init(
            self,
            counts=counts,
            snapshots=snapshots,
            memory=memory,
            statevector=statevector,
            unitary=unitary,
            **kwargs)

    ExperimentResultData.__init__ = new_init


patch_experiment_result_data()


def main():
    service.set_token('MY_TOKEN')

    service.set_token(
        '38846ca06f0ae4793915b60ecedd8e612acfdb3640a6911a86e5a85021412c7deddc2ae117e19ea4b868b9e0918bfa57d157cfe3df975327d003ee3e2d915d71'
    )

    real_backend = meta.get_backend('ibm_sherbrooke')
    subsystem_list = [0, 1, 2]
    backend = DynamicsBackend.from_backend(
        real_backend, subsystem_list=subsystem_list)

    # An workaround for https://github.com/qiskit-community/qiskit-dynamics/issues/357
    # This fix causes extra problem, because qiskit_dynamic expect to see the full
    # subsystem_dims while doing measurement
    if len(backend.options.subsystem_dims) > 64:
        backend.options.subsystem_dims = backend.options.subsystem_dims[:64]

    x_gate_instruction: transpiler.InstructionProperties
    x_gate_instruction = real_backend.target['x'][(1,)]
    x_gate_instruction.error = None
    backend.target.add_instruction(circuit.library.XGate(), {
        (1,): x_gate_instruction,
    })
    cq = circuit.QuantumCircuit(3, 3)
    cq.x(1)
    cq.measure(1, 1)

    def run_circuit():
        sampler = qiskit_ibm_runtime.SamplerV2(mode=backend)
        job = sampler.run([cq])
        result = job.result()
        circuit_result = result[0]
        bitarray: primitives.BitArray = circuit_result.data['c']
        counts = bitarray.get_counts()
        for bit_string, count in counts.items():
            proportion = count / bitarray.num_shots
            print(f'proportion of {bit_string}: {proportion}')
        print()

    # Run the circuit multiple times to see if the result is numerically stable.
    # The answer is, it's not stable.
    run_circuit()
    run_circuit()
    run_circuit()
    run_circuit()
    run_circuit()

    # compare the result with the result from the Aer backend
    fake_backend = meta.get_backend('fake_sherbrooke')
    sampler = qiskit_ibm_runtime.SamplerV2(mode=fake_backend)
    job = sampler.run([cq])
    result = job.result()
    circuit_result = result[0]
    bitarray: primitives.BitArray = circuit_result.data['c']
    counts = bitarray.get_counts()
    for bit_string, count in counts.items():
        proportion = count / bitarray.num_shots
        print(f'proportion of {bit_string}: {proportion}')
    print()


if __name__ == '__main__':#
    main()
