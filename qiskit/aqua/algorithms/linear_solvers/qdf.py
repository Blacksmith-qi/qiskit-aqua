
""" The QDF algorithm """

from typing import Optional, Union, Dict, Any, Tuple
import logging
from copy import deepcopy
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
            orig_size: Optional[Tuple[int, int]] = None,
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
            orig_size: Orignal size of the matrix before resizing
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
                        orig_size[1],
                        quantum_instance)
        self._eigs2 = eigs2
        self._rotation_inverse = reciprocal
        self._orig_rows = orig_size[0]
        self._orig_columns = orig_size[1]
 

    @staticmethod
    def matrix_resize(matrix: np.ndarray,
                      vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
        """Resizes matrix if necessary

        Args:
            matrix: the input matrix of linear system of equations
            vector: the input vector of linear system of equations
        Returns:
            new matrix, vector, truncate_powerdim, truncate_hermitian
        Raises:
            ValueError: invalid input
        """
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector)

        if matrix.shape[0] != len(vector):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")

        truncate_powerdim = False
        truncate_hermitian = False
        orig_size = None
        if orig_size is None:
            orig_size = len(vector)
        
        is_hermitian = np.allclose(matrix, matrix.conj().T)
        if not is_hermitian:
            logger.warning("Input matrix is not hermitian. It will be "
                           "expanded to a hermitian matrix automatically.")
            # Use resizing from paper
            matrix, vector = QDF.expand_to_hermitian(matrix, vector)
            truncate_hermitian = True


        is_powerdim = np.log2(matrix.shape[0]) % 1 == 0
        if not is_powerdim:
            logger.warning("Input matrix does not have dimension 2**n. It "
                           "will be expanded automatically.")
            matrix, vector = HHL.expand_to_powerdim(matrix, vector)
            truncate_powerdim = True

        # Return hermitian konjugate to get later I(F.H) and A = I(F.H)^2 
        return matrix, vector, truncate_powerdim, truncate_hermitian

    @staticmethod
    def expand_to_hermitian(matrix: np.ndarray,
                            vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Expand a non-hermitian matrix A to a hermitian matrix by
        [[0, A], [A.H, 0]] and expand vector b to [0,b].

        Args:
            matrix: the input matrix
            vector: the input vector

        Returns:
            the expanded matrix, the expanded vector
        """
        #
        rows = matrix.shape[0]
        columns = matrix.shape[1]
        new_matrix_right = np.zeros([rows, rows])
        new_matrix_right = np.array(new_matrix_right, dtype=complex)
        
        new_matrix_left = np.zeros([columns, columns])
        new_matrix_left = np.array(new_matrix_left, dtype=complex)

        # concatenating to one matrix

        top_matrix = np.concatenate((new_matrix_left, matrix.conj().T), axis=1)
        bottom_matrix = np.concatenate((matrix, new_matrix_right), axis=1)

        matrix = np.concatenate((top_matrix, bottom_matrix))

        new_vector = np.zeros((1, rows + columns))
        new_vector = np.array(new_vector, dtype=complex)
        new_vector[0, columns:] = vector
        vector = new_vector.reshape(np.shape(new_vector)[1])
        return matrix, vector

    def _resize_vector(self, vec: np.ndarray) -> np.ndarray:
        if self._truncate_hermitian or self._truncate_powerdim:
            vec = vec[:self._orig_columns] #Take first M entries
        return vec
    
    def _resize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if self._truncate_hermitian or self._truncate_powerdim:
            M = self._orig_columns #Columns
            N = self._orig_rows# Rows
            matrix = matrix[M : M+N, 0:M]
        return matrix
    
    def _resize_in_vector(self, vec: np.ndarray) -> np.ndarray:
        if self._truncate_hermitian or self._truncate_powerdim:
            vec = vec[-self._orig_rows:] #Take first N entries
        return vec
    



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

    def _state_tomography(self) -> None:
        """The state tomography.

        The QDF result gets extracted via state tomography. Available for
        qasm simulator and real hardware backends.
        """

        # Preparing the state tomography circuits measuring the io register
        tomo_circuits = state_tomography_circuits(self._circuit,
                                                  self._io_register)
        # Copy without measured ancillae, needed for the tomo fitter 
        tomo_circuits_noanc = deepcopy(tomo_circuits)
        # Adding measuremets to the ancillae
        ca = ClassicalRegister(2)
        for circ in tomo_circuits:
            circ.add_register(ca)
            circ.measure(self._reciprocal._anc, ca[0])
            circ.measure(self._rotation_inverse._anc, ca[1])

        # Extracting the probability of successful run
        results = self._quantum_instance.execute(tomo_circuits)
        probs = []
        for circ in tomo_circuits:
            counts = results.get_counts(circ)
            s, f = 0, 0
            for k, v in counts.items():
                # Both ancillae in state 1
                if k[0] == "1" and k[2] == "1":
                    s += v
                else:
                    f += v
            probs.append(s / (f + s))
        probs = self._resize_vector(probs)
        self._ret["probability_result"] = np.real(probs)

        # Filtering the tomo data for valid results with ancillary measured
        # to 1, i.e. c1==1 and c2==1
        results_noanc = self._tomo_postselect(results)
        tomo_data = StateTomographyFitter(results_noanc, tomo_circuits_noanc)
        rho_fit = tomo_data.fit('lstsq')
        vec = np.sqrt(np.diag(rho_fit))
        self._hhl_results(vec)

    def _tomo_postselect(self, results: Any) -> Any:
        new_results = deepcopy(results)

        for resultidx, _ in enumerate(results.results):
            old_counts = results.get_counts(resultidx)
            new_counts = {}

            # change the size of the classical register
            # Dropping the last two classical registers
            new_results.results[resultidx].header.creg_sizes = [
                new_results.results[resultidx].header.creg_sizes[0]]
            new_results.results[resultidx].header.clbit_labels = \
                new_results.results[resultidx].header.clbit_labels[0:-2]
            new_results.results[resultidx].header.memory_slots = \
                new_results.results[resultidx].header.memory_slots - 2

            for reg_key in old_counts:
                reg_bits = reg_key.split(' ')
                # Both ancillae need to be in 1
                if reg_bits[0] == '1' and reg_bits[2] == '1':
                    new_counts[reg_bits[1]] = old_counts[reg_key]

            data_counts = new_results.results[resultidx].data.counts
            new_results.results[resultidx].data.counts = \
                new_counts if isinstance(data_counts, dict) else data_counts.from_dict(new_counts)

        return new_results

    def _hhl_results(self, vec: np.ndarray) -> None:
        res_vec = self._resize_vector(vec)
        in_vec = self._resize_in_vector(self._vector)
        matrix = self._resize_matrix(self._matrix)
        self._ret["output"] = res_vec
        # Rescaling the output vector to the real solution vector
        tmp_vec = matrix.dot(res_vec)
        f1 = np.linalg.norm(in_vec) / np.linalg.norm(tmp_vec)
        # "-1+1" to fix angle error for -0.-0.j
        f2 = sum(np.angle(in_vec * tmp_vec.conj() - 1 + 1)) 
        self._ret["solution"] = f1 * res_vec * np.exp(-1j * f2)





