
""" The QDF algorithm """

from typing import Optional, Union, Dict, Any, Tuple
import logging
from copy import Error, deepcopy
import numpy as np
from numpy import random
import scipy as sp
from collections import Counter
from numpy.core.records import array
import pickle

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.eigs.eigs_qpe import EigsQPE
from qiskit.quantum_info import DensityMatrix
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua.algorithms.linear_solvers import HHL, QDF
from qiskit.ignis.verification.tomography import state_tomography_circuits, \
    StateTomographyFitter
from qiskit.aqua.components.initial_states import InitialState, Custom
from qiskit.aqua.components.reciprocals import Reciprocal
from qiskit.aqua.components.eigs import Eigenvalues
from qiskit.aqua.operators import MatrixOperator

from qiskit.circuit.library import QFT

from .linear_solver_result import LinearsolverResult

logger = logging.getLogger(__name__)

class QQPS(HHL):
    r""" The Quadrati programming solver algorithm

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
            rotation1: Optional[Reciprocal] = None,
            rotation2: Optional[Reciprocal] = None,
            num_q: int = 0,
            num_a: int = 0,
            quantum_instance: Optional[
                Union[QuantumInstance, BaseBackend, Backend]] = None,
            save_name: str = None) -> None:
        """
        Args:
            matrix: The input matrix of linear system of equations
            vector: The input vector of linear system of equations
            truncate_powerdim: Flag indicating expansion to 2**n matrix to be truncated
            truncate_hermitian: Flag indicating expansion to hermitian matrix to be truncated
            eigs: The eigenvalue estimation instance
            eigs2: The eigenvalue estimation instance for matrix^2
            init_state: The initial quantum state preparation
            rotation: The eigenvalue direct controlled rotation
            rotation: The eigenvalue direct controlled rotation
            num_q: Number of qubits required for the matrix Operator instance
            num_a: Number of ancillary qubits for Eigenvalues instance
            quantum_instance: Quantum Instance or Backend
            save_name: Suffix for saved files. If None nothing is saved
        Raises:
            ValueError: Invalid input
        """

        size = np.shape(matrix)[0]

        super().__init__(matrix,
                        vector,
                        truncate_powerdim,
                        truncate_hermitian,
                        eigs,
                        init_state,
                        rotation1,
                        num_q,
                        num_a,
                        size,   
                        quantum_instance)
        self._eigs2 = eigs2
        self._rotation_inverse = rotation2
        self._save_name = save_name
        self._orig_columns = size
        self._orig_rows = size
 


    
    def _resize_vector(self, vec: np.ndarray) -> np.ndarray:
        if self._truncate_hermitian or self._truncate_powerdim:
            vec = vec[:self._orig_columns] #Take first M entries
        return vec
    
    def _resize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        M = self._orig_columns #Columns
        N = self._orig_rows# Rows
        if self._truncate_powerdim:
            if self._truncate_hermitian:
                matrix = matrix[: M + N,: M + N]
            else:
                matrix = matrix[: N, :M]
        
        if self._truncate_hermitian:
            matrix = matrix[M : M+N, 0:M]
        return matrix

    def _resize_in_vector(self, vec: np.ndarray) -> np.ndarray:
        M = self._orig_columns #Columns
        N = self._orig_rows# Rows
        if self._truncate_powerdim:
            if self._truncate_hermitian:
                vec = vec[: N + M]
            else:
                vec = vec[: N]
        if self._truncate_hermitian:
            vec = vec[-self._orig_rows:] #Take last N entries
        return vec

    def construct_circuit(self, 
                        measurement: bool = False,
                        measure_result: bool = False) -> QuantumCircuit:
        """Construct the QDF circuit.

        Args:
            measurement: indicate whether measurement on both ancillary qubits
                should be performed
            measure_result: result register is also measured

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

        # Measuring the ancilla qubits
        if measurement:
            c = ClassicalRegister(2, name='result_anc')
            qc.add_register(c)
            qc.measure(self._reciprocal._anc, c[0])
            qc.measure(self._rotation_inverse._anc, c[1])
            self._success_bit = c

        # Measureing the result register
        if measure_result:
            c = ClassicalRegister(self._io_register._size, name='result')
            qc.add_register(c)
            qc.measure(self._io_register, c)
        
        self._circuit = qc

        return qc

    def _statevector_simulation(self) -> None:
        """The statevector simulation.

        The QDF result gets extracted from the statevector. Only for
        statevector simulator available.
        """
        res = self._quantum_instance.execute(self._circuit)
        sv = np.asarray(res.get_statevector(self._circuit))

        print('DEBUG Info')
        print(self._ancilla_index)
        print(self._circuit)
        print(sv)

        # Extract solution vector from statevector
        vec = self._reciprocal.sv_to_resvec(sv, self._num_q, 
                                    self._ancilla_index, True)

        print(vec)
        print(len(vec))
        # remove added dimensions
        self._ret['probability_result'] = \
            np.real(self._resize_vector(vec).dot(self._resize_vector(vec).conj()))
        vec = vec / np.linalg.norm(vec)
        self._hhl_results(vec)


    def _hhl_results(self, vec: np.ndarray) -> None:
        res_vec = self._resize_vector(vec)
        in_vec = self._resize_in_vector(self._vector)
        matrix = self._resize_matrix(self._matrix)
        self._ret["output"] = res_vec
        self._in_vector = in_vec
        # Rescaling the output vector to the real solution vector

        #DEBUG
        print('DEBUG INFO')
        print(res_vec)
        print(in_vec)
        print(matrix)
        print(vec)

        # cost functino of problem
        def costfn(x):
            return 0.5 * x @ matrix @ x + res_vec.T @ x 

        # Calculating the real solution vector
        result_ref_all = sp.optimize.minimize(costfn, x0=np.ones(self._orig_columns))
        result_ref = result_ref_all.x

        # Difference between output and solution
        def diff(x, output, solution):
            return np.linalg.norm((x[0] + x[1]*1j)*output - solution)
        # Minimize this function
        if np.array_equal(res_vec, np.zeros(len(res_vec))):
            solution = res_vec
        else:
            res_min = sp.optimize.minimize(diff,
                                    x0=(1,1),
                                    args=(res_vec, result_ref))

            solution = res_vec * (res_min.x[0] + res_min.x[1]*1j)

        self._ret["solution"] = solution.copy()
        self._ret["solution_ref"] = result_ref

         



