import math
from typing import List

from qulacs import Observable, ParametricQuantumCircuit, QuantumState, gate
from qulacs.gate import DenseMatrix


def python_backprop(circ: ParametricQuantumCircuit, obs: Observable) -> List[float]:
    n = circ.get_qubit_count()

    num_gates = circ.get_gate_count()
    gate_list = []
    for i in range(num_gates):
        gate_list.append(circ.get_gate(i))

    obs_gate = DenseMatrix(range(n), obs.get_matrix().toarray())

    inverse_parametric_gate_position = [-1] * num_gates
    for i in range(circ.get_parameter_count()):
        inverse_parametric_gate_position[circ.get_parametric_gate_position(i)] = i
    ans = [0.0] * circ.get_parameter_count()

    inv_gate_list = []
    for i in range(num_gates - 1, -1, -1):
        gate_now = circ.get_gate(i)
        if inverse_parametric_gate_position[i] != -1:
            if gate_now.get_name() == "ParametricRX":
                rcpi = gate.RX(gate_now.get_target_index_list()[0], math.pi)
            elif gate_now.get_name() == "ParametricRY":
                rcpi = gate.RY(gate_now.get_target_index_list()[0], math.pi)
            elif gate_now.get_name() == "ParametricRZ":
                rcpi = gate.RZ(gate_now.get_target_index_list()[0], math.pi)
            else:
                raise RuntimeError()

            state = QuantumState(n)

            # <bistate|
            for g in gate_list[::-1]:
                g.get_inverse().update_quantum_state(state)
            obs_gate.get_inverse().update_quantum_state(state)
            for g in inv_gate_list[::-1]:
                g.get_inverse().update_quantum_state(state)

            # |state>
            for g in gate_list:
                g.update_quantum_state(state)
            rcpi.update_quantum_state(state)
            for g in inv_gate_list:
                g.update_quantum_state(state)

            # 期待値測定
            backobs = Observable(n)
            obs_str = ""
            for i in range(n):
                obs_str += f"Z {i} "
            backobs.add_operator(
                1.0,
                obs_str,
            )
            ans[inverse_parametric_gate_position[i]] = backobs.get_expectation_value(
                state
            )

        inv_gate = gate_now.get_inverse()
        inv_gate_list.append(inv_gate)
    return ans
