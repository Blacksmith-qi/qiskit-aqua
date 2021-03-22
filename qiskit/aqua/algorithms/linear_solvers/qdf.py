
""" The QDF algorithm """

from typing import Optional, Union, Dict, Any, Tuple
import logging
from copy import Error, deepcopy
import numpy as np
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
from qiskit.aqua.algorithms.linear_solvers import HHL
from qiskit.ignis.verification.tomography import state_tomography_circuits, \
    StateTomographyFitter
from qiskit.aqua.components.initial_states import InitialState, Custom
from qiskit.aqua.components.reciprocals import Reciprocal
from qiskit.aqua.components.eigs import Eigenvalues
from qiskit.aqua.operators import MatrixOperator

from qiskit.circuit.library import QFT

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
            orig_size: Optional[Tuple[int, int]] = [None,None],
            mprime: Optional[int] = 2,
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
            reciprocal: The eigenvalue reciprocal and controlled rotation instance
            rotation: The eigenvalue direct controlled rotation
            num_q: Number of qubits required for the matrix Operator instance
            num_a: Number of ancillary qubits for Eigenvalues instance
            orig_size: Orignal size of the matrix before resizing
            mprime: Reduced numer of fit functions according to paper
            quantum_instance: Quantum Instance or Backend
            save_name: Suffix for saved files. If None nothing is saved
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
        self._matrix_old = None
        self._vector_old = None
        self._mprime = mprime
        self._idx_keep_dim = None
        self._save_name = save_name
 

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
            raise ValueError(f"Input vector dimension {len(vector)} does not match input "
                             f"matrix dimension {matrix.shape[0]}!")

        truncate_powerdim = False
        truncate_hermitian = False
        orig_size = None
        if orig_size is None:
            orig_size = len(vector)

        # Prevent Problems with 1x1 matrices
        if matrix.shape[0] == 1 and matrix.shape[1] == 1:
            logger.warning("Input matrix is only 1x1 It will be "
                           "expanded to a hermitian matrix automatically.")
            # Use resizing from paper
            matrix, vector = QDF.expand_to_hermitian(matrix, vector)
            truncate_hermitian = True
        
        if matrix.shape[0] != matrix.shape[1]:
            # matrix is not square
            is_hermitian = False
        else: 
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

    @staticmethod
    def preparation(
        matrix: np.ndarray,
        vector: np.ndarray,
        num_ancillae: int,
        swap_test: Optional[bool] = False,
        evo_time: Optional[tuple[float,float]] = [None, None],
        num_time_slices: Optional[int] = 50,
        negative_evals: Optional[bool] = False,
        expansion_mode: Optional[str] = 'suzuki',
        expansion_order: Optional[int] = 2) -> Tuple:
        """
        Prepares everything for the algorithm.
        - Manges expanding matrix depending on hermitan or powerdim.
        - Creates Instances of the eigenvalue circuits
        - Returns as second matrix A noramlly and I(F) if one wants to 
            perform the swap test

        Args:
            matrix: The input matrix of linear system of equations
            vector: The input vector of linear system of equations
            swap_test: Information if swap test will be used
            num_ancillae: The number of ancillary qubits to use for the measurement,
            evo_time: An optional evolution time which should scale the eigenvalue onto the range
                :math:`(0,1]` (or :math:`(-0.5,0.5]` for negative eigenvalues). Defaults to
                ``None`` in which case a suitably estimated evolution time is internally computed.
            num_time_slices: The number of time slices, has a minimum value of 1.
            negative_evals: If negative eigenvalues should be taken into account
            expansion_mode: The expansion mode ('trotter' | 'suzuki')
            expansion_order: The suzuki expansion order, has a minimum value of 1.

        Returns:
            matrix: The input matrix of linear system of equations
            vector: The input vector of linear system of equations
            truncate_powerdim: Flag indicating expansion to 2**n matrix to be truncated
            truncate_hermitian: Flag indicating expansion to hermitian matrix to be truncated
            eigs: The eigenvalue estimation instance
            eigs2: The eigenvalue estimation instance for matrix^2
            num_q: Number of qubits required for the matrix Operator instance
            num_a: Number of ancillary qubits for Eigenvalues instance
            orig_size: Orignal size of the matrix before resizing
 
        """
        orig_size = matrix.shape
        matrix_F_dagger, vector, truncate_powerdim, truncate_hermitian = \
            QDF.matrix_resize(matrix, vector)

        # Additional ancilla qubit to keep accuracy
        ne_qfts = [None, None]
        if negative_evals:
            num_ancillae += 1
            ne_qfts = [QFT(num_ancillae - 1), QFT(num_ancillae - 1).inverse()]
            evo_time = [time / 2 for time in evo_time]


        # Create Eigenvalue instaces
        eigs = EigsQPE(MatrixOperator(matrix_F_dagger),
                        QFT(num_ancillae, inverse=True),
                        num_time_slices = num_time_slices,
                        expansion_mode=expansion_mode,
                        num_ancillae = num_ancillae,
                        expansion_order=expansion_order,
                        negative_evals=negative_evals,
                        evo_time=evo_time[0],
                        ne_qfts=ne_qfts)
        if swap_test:

            # Create I(F) using existing function
            matrix_2, *unused = QDF.matrix_resize(matrix.conjugate().T,
                                        np.zeros([matrix.shape[1]]))
            evo_time_2 = evo_time[0]
        else:
            matrix_2 = matrix_F_dagger @ matrix_F_dagger
            evo_time_2 = evo_time[1]


        eigs2 = EigsQPE(MatrixOperator(matrix_2),
                        QFT(num_ancillae, inverse=True),
                        num_time_slices = num_time_slices,
                        expansion_mode=expansion_mode,
                        num_ancillae = num_ancillae,
                        expansion_order=expansion_order,
                        negative_evals=negative_evals,
                        evo_time=evo_time_2,
                        ne_qfts=ne_qfts)


        num_q, num_a = eigs.get_register_sizes()

        result = matrix_F_dagger,matrix_2, vector, truncate_powerdim, truncate_hermitian, \
                eigs, eigs2, num_q, num_a, orig_size
        return result

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
                if k[0] == "1" and k[1] == "1":
                    s += v
                else:
                    f += v
            probs.append(s / (f + s))
        probs = self._resize_vector(probs)
        self._ret["probability_result"] = np.real(probs)

        # Filtering the tomo data for valid results with ancillary measured
        # to 1, i.e. c1==1 and c2==1
        results_noanc = self._tomo_postselect(results)

        # Saving data to pickle
        if self._save_name is not None:
            pickle.dump(tomo_circuits, 
                        open("tomo_circuits_" + self._save_name +".pkl", 'wb'))
            pickle.dump(tomo_circuits_noanc,
                        open("tomo_circuits_noanc_" + self._save_name +".pkl", 'wb'))
            pickle.dump(results, 
                        open("results_" + self._save_name +".pkl", 'wb'))

        # Starting fit
        try:
            tomo_data = StateTomographyFitter(results_noanc, tomo_circuits_noanc)
            rho_fit = tomo_data.fit('cvx')
            print(rho_fit)
            vec = DensityMatrix(rho_fit).to_statevector(atol=0.8)
            fit_vec = vec.data
        except:
            print("Fitting to measurement failed")
            print("Setting result to zero")
            fit_vec = np.zeros(len(self._vector))
        self._hhl_results(fit_vec)

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
                if reg_bits[0] == '11':
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
        self._in_vector = in_vec
        # Rescaling the output vector to the real solution vector

        # Calculating the real solution vector
        result_ref = np.linalg.pinv(matrix) @ in_vec

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

        self._ret["solution"] = solution
        
        # Reconstructing original vector if number of fit functions
        # have been reduced
        if self._idx_keep_dim is not None:
            # Add removed dims
            res_old = []
            for idx in range(self._original_dimension):
                if idx in self._idx_keep_dim:
                    res_old.append(solution[0])
                    solution = np.delete(solution,0)
                else:
                    res_old.append(0)

            self._ret["solution_old"] = np.array(res_old)


    @staticmethod
    def _filter_results(counts):
        """
        Returns all mesurements with both ancilla qubit in 11.
        Ancilla qubits need to be first measurement e.g. '0101 11'

        Args:
            counts: Counts of a run on a real device or qasm simulator

        Returns:
            counts_filterd: all counts with ancillae in '11'
        """
        new_counts = {}
        for key, value in counts.items():
            # Both ancilla need to be in 1
            if key[-1] == "1" and key[-2] == "1":
                new_counts[key[:-3]] = value
        return new_counts


        

    def reduce_fitfunctions(self, num_fit_func: Optional[int] = None, 
                            factor: Optional[float] = 2,
                            new_evo_time: Optional[Tuple[float, float]] =
                                [None, None]) -> None:
        """
        Start of part 3 of the paper.
        Samples Algrithm O(num_fit_func) times and create histogram.
        Choose the most important num_fit_funtions and reduce dimension
        of matrix to the needed fit functions

        Args:
            num_fit_func: Number M' of the most important fit functions
            factor: Factor by which num_fit_func is multiplied to get 
                    number of runs
        
        Returns:
            nothing but modifys used matrix
        """

        if num_fit_func is None:
            # Use mprime from functino init
            num_fit_func = self._mprime
    

        if self._quantum_instance.is_statevector:
            raise Error("Quantum instance needs to be a real device or qasm simulator")
        else:
            self.construct_circuit(measurement=True, measure_result=True)


        number_of_runs = num_fit_func * factor

        self._quantum_instance._run_config.shots = number_of_runs 
        result = self._quantum_instance.execute(self._circuit)

        counts = result.get_counts(self._circuit)
        counts = QDF._filter_results(counts)

        # Selecting most important fit functions
        idx_keep_dim = []
        for idx in range(num_fit_func):
            # Get highes count key
            key = max(counts, key = counts.get)
            del counts[key]
            # Only keep dim if it is not a additional dim from
            # truncation
            dim = int(key,2)
            if dim < self._orig_columns:
                idx_keep_dim.append(dim)
        
        self._idx_keep_dim = idx_keep_dim


        self._matrix_old = self._matrix
        self._vector_old = self._vector

        # Resize the matrix
        new_matrix = self._resize_matrix(self._matrix)
        new_vector = self._resize_in_vector(self._vector)


        new_matrix = new_matrix[:,idx_keep_dim]
        self._keep_dim = idx_keep_dim

        self._matrix_new = new_matrix
        self._vector_new = new_vector

        # Initailize with new matrix
        matrix, matrix2, vector, truncate_powerdim, truncate_hermitian, \
            eigs, eigs2, num_q, num_a, orig_size = \
                QDF.preparation(new_matrix,
                                new_vector,
                                self._num_a)

        self._matrix = matrix
        self._vector = vector
        self._truncate_hermitian = truncate_hermitian
        self._truncate_powerdim = truncate_powerdim
        self._eigs = eigs
        self._eigs2 = eigs2
        self._num_q = num_q
        self._orig_rows = orig_size[0]
        self._orig_columns = orig_size[1]
        self._init_state = Custom(num_q, state_vector=vector)


    @staticmethod
    def create_matrix(dim: int ,
                    seed: Optional[int] = 0,
                    kappa: Optional[float] = 4) -> Tuple[np.ndarray, np.ndarray]:

        """
        Creates an sparse, square, hermitian matrix with given dimension
        and seed.
        Checks if diagonal is not all zero

        Args:
            dim: dimension of the matrix    
            seed: seed for random generator
            kappa: max ratio between smallest and biggest eval

        Returns:
            matrix: random matrix 
            vector: random vector with same seed
        """

        # Default density 
        density = 0.1
        # Increase density for 4x4 matrixes
        if dim == 4:
            density = 0.3
        elif dim == 16:
            density == 0.01

        # Set seed
        np.random.seed(seed)

        # Create cextor
        vector = np.random.rand(dim)

        # Create matrix
        Found = False
        Runs = 0
        while not Found:
            # Increase counter 
            Runs += 1
            if Runs > 10000:
                raise RuntimeError('Faild to create random matrix')

            pre_matrix = sp.sparse.random(dim, dim, density=density)
            pre_matrix = pre_matrix.toarray()
            # Make it hermitan
            matrix = 0.5 * (pre_matrix + pre_matrix.conj().T)
            # Check diag not zero
            for i in range(dim):
                matrix[i,i] += vector[i]/2
            eigvalues = np.linalg.eigvals(matrix)

            # neg evals
            if not 0 in abs(eigvalues):
                if max(abs(eigvalues)) / min(abs(eigvalues)) <= kappa:
                    matrix = matrix / max(abs(eigvalues)) * 2
                    # Fix spectrum
                    if not np.array_equal(np.diag(matrix),np.zeros(dim)):
                        Found = True

        return matrix, vector



            

