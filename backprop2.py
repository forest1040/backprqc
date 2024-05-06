import math
from typing import List
import numpy as np

from sympy.tensor.array import tensorproduct

from qulacs import Observable, ParametricQuantumCircuit, QuantumState, gate
from qulacs.state import inner_product



def backprop_inner_product(
    circ: ParametricQuantumCircuit, bistate:QuantumState, obs: Observable, coef: float) -> List[float]:
    n = circ.get_qubit_count()
    state = QuantumState(n)
    state.set_zero_state()
    circ.update_quantum_state(state)

    #H = np.array(obs.get_matrix())
    Z = np.array([[1, 0], [0, -1]])

    #H = obs.get_matrix()
    #print(H)
    H = tensorproduct(Z * coef, tensorproduct(Z, tensorproduct(Z, Z)))
    print("H:", H)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # ユニタリーゲートの構築
    U = np.zeros_like(H, dtype=np.complex128)
    for i in range(len(eigenvalues)):
        phase = np.exp(1j * np.angle(eigenvalues[i]))
        U += phase * np.outer(eigenvectors[:, i], np.conj(eigenvectors[:, i]))

    print("U:", U)

    U2 = gate.DenseMatrix(list(range(n)), U)

    num_gates = circ.get_gate_count()
    inverse_parametric_gate_position = [-1] * num_gates
    for i in range(circ.get_parameter_count()):
        inverse_parametric_gate_position[circ.get_parametric_gate_position(i)] = i
    ans = [0.0] * circ.get_parameter_count()

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
            rcpi.update_quantum_state(state)
            U2.update_quantum_state(bistate)
            ans[inverse_parametric_gate_position[i]] = (
                # inner_product(bistate, state).real / 2.0
                inner_product(bistate, state).real
            )
            rcpi.get_inverse().update_quantum_state(state)
            U2.get_inverse().update_quantum_state(bistate)
        agate = gate_now.get_inverse()
        agate.update_quantum_state(bistate)
        agate.update_quantum_state(state)
    return ans


# bistate: Observableとcirc適用済みの状態
def python_backprop(circ: ParametricQuantumCircuit, obs: Observable, coef:float) -> List[float]:
    n = circ.get_qubit_count()
    state = QuantumState(n)
    state.set_zero_state()
    circ.update_quantum_state(state)
    # bistate = QuantumState(n)
    # astate = QuantumState(n)
    # obs.apply_to_state(astate, state, bistate)
    # bistate.multiply_coef(2)
    return backprop_inner_product(circ, state, obs, coef)
