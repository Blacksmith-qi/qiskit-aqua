
""" The QDF algorithm """

from typing import Optional, Union, Dict, Any, Tuple
import logging
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua.algorithms.linear_solvers import HHL
from qiskit.ignis.verification.tomography import state_tomography_circuits, \
    StateTomographyFitter
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.reciprocals import Reciprocal
from qiskit.aqua.components.eigs import Eigenvalues
from .linear_solver_result import LinearsolverResult

logger = logging.getLogger(__name__)

class QDF(HHL):
    r""" The QDF algorithm

    TODO
    description

    """

    def __init__(
            self,
            matrix: np.ndarray,
            vector: np.ndarray,
            truncate_powerdim: bool = False,
            truncate_hermitian: bool = False,
            eigs: Optional[Eigenvalues] = None,
            eigs2: Optional[Eigenvalues] = None,
            init_state: Optional[Union[QuantumCircuit, InitialState]] = None,
            reciprocal: Optional[Reciprocal] = None,
            rotation: Optional[Reciprocal] = None,
            num_q: int = 0,
            num_a: int = 0,
            orig_size: Optional[int] = None,
            quantum_instance: Optional[
                Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        """
        Args:
            matrix: The input matrix of linear system of equations
            vector: The input vector of linear system of equations
            truncate_powerdim: Flag indicating expansion to 2**n matrix to be truncated
            truncate_hermitian: Flag indicating expansion to hermitian matrix to be truncated
            eigs: The eigenvalue estimation instance
            eigs2: The eigenvalue estimation instance for matrix^2
            init_state: The initial quantum state preparation
            reciprocal: The eigenvalue reciprocal and controlled rotation instance
            rotation: The eigenvalue direct controlled rotation
            num_q: Number of qubits required for the matrix Operator instance
            num_a: Number of ancillary qubits for Eigenvalues instance
            orig_size: The original dimension of the problem (if truncate_powerdim)
            quantum_instance: Quantum Instance or Backend
        Raises:
            ValueError: Invalid input
        """
        super().__init__(matrix,
                        vector,
                        truncate_powerdim,
                        truncate_hermitian,
                        eigs,
                        init_state,
                        rotation,
                        num_q,
                        num_a,
                        orig_size,
                        quantum_instance)
        self._eigs2 = eigs2
        self._rotation_inverse = reciprocal
 

    def construct_circuit(self, measurement: bool = False) -> QuantumCircuit:
        """Construct the QDF circuit.

        Args:
            measurement: indicate whether measurement on both ancillary qubits
                should be performed

        Returns:
            the QuantumCircuit object for the constructed circuit
        """

        qc = super().construct_circuit(measurement=False)
        q = self._io_register
        a = self._eigenvalue_register
        s = self._ancilla_register
        self._ancilla_index = int(q.size + a.size)

        qc.barrier(a)
        # EigenvalueEstimation (QPE)
        qc += self._eigs2.construct_circuit("circuit", q, a)
        a = self._eigs2._output_register

        # Reciprocal calculation with rotation
        qc += self._rotation_inverse.construct_circuit("circuit", a)

        # Inverse EigenvalueEstimation
        qc += self._eigs2.construct_inverse("circuit", self._eigs2._circuit)
        
        self._circuit = qc
        
        return qc

    def _statevector_simulation(self) -> None:
        """The statevector simulation.

        The QDF result gets extracted from the statevector. Only for
        statevector simulator available.
        """
        res = self._quantum_instance.execute(self._circuit)
        sv = np.asarray(res.get_statevector(self._circuit))
        # Extract solution vector from statevector
        vec = self._reciprocal.sv_to_resvec(sv, self._num_q, 
                                    self._ancilla_index, True)
        # remove added dimensions
        self._ret['probability_result'] = \
            np.real(self._resize_vector(vec).dot(self._resize_vector(vec).conj()))
        vec = vec / np.linalg.norm(vec)
        self._hhl_results(vec)






