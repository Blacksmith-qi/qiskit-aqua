""" Black box implementation of QPE for getting eigenvalues of a matrix."""

from os import stat
from typing import Optional, List
import numpy as np
from numpy.matrixlib.defmatrix import matrix
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.initial_states import Custom
from .eigs import Eigenvalues

class BBEigs(Eigenvalues):
    """ Eigenvalues using a black box.

    Calculates the eigenvalues of the matrix classically and encodes the estimation
    in the given register
    """

    def __init__(self,
                 matrix: np.ndarray,
                 num_ancillae: int = 1,
                 negative_evals: bool = False) -> None:

        """
        Args:
            matrix: The matrix to estimate the eigenvalues
            num_ancillae: The number of ancillary qubits to use for the measurement,
                has a minimum value of 1.
            negative_evals: Set ``True`` to indicate negative eigenvalues need to be handled
        """

        super().__init__()

        # Check if matrix is np.array
        if type(matrix) is not np.ndarray:
            matrix = np.array(matrix)

        self._matrix = matrix
        self._num_ancillae = num_ancillae
        self._negative_evals = negative_evals
        self._num_q = int(np.log2(np.shape(matrix)[0]))

    def get_register_sizes(self):
        return self._num_q, self._num_ancillae

    def get_scaling(self):
        return 0

    def construct_circuit(self, mode, register, a=None):
        """Construct the eigenvalues estimation using the PhaseEstimationCircuit

        Args:
            mode (str): construction mode, 'matrix' not supported
            register (QuantumRegister): the register to use for the quantum state

        Returns:
            QuantumCircuit: object for the constructed circuit
        Raises:
            ValueError: QPE is only possible as a circuit not as a matrix
        """
        # calculation of the eigenvalues
        evals = np.linalg.eigvals(self._matrix)

        num_of_states_half = 2 ** (self._num_ancillae - 1)

        # Rescaling the eigenvalues
        evals_scale = evals / max(abs(evals)) * num_of_states_half

        if self._negative_evals:
            # Get free sign qubit
            evals_scale = evals_scale / 2

            # Turn sign qubit into 1 for negative evals
            new_evals = []
            for value in evals_scale:
                if value < 0:
                    new_evals.append(abs(value) + num_of_states_half)
                else:
                    new_evals.append(value)

            evals_scale = new_evals


        # Convert to int
        evals_scale = [int(np.round(value,0)) for value in evals_scale]

        # Reverse order of the bits due to qiskits convetion
        new_evals = []
        for value in evals_scale:
            bits_str = "{0:b}".format(value)
            # Append leading zeros
            bits_str = '0' * (self._num_ancillae - len(bits_str)) + bits_str
            # Reverse order
            bits_str_qiskit = bits_str[::-1]
            new_evals.append(int(bits_str_qiskit, 2))
        evals_scale = new_evals


        # Create circuit using CUSTOM
        state_vec = []
        for idx in range(2 * num_of_states_half):
            if idx in evals_scale:
                state_vec.append(1)
            else:
                state_vec.append(0)
        state_vec = np.array(state_vec)

        state = Custom(num_qubits=self._num_ancillae,
                        state_vector=state_vec)

        circuit = QuantumCircuit(a)
        circuit += state.construct_circuit(register=a)

        return circuit 