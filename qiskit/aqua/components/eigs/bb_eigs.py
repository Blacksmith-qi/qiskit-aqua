""" Black box implementation of QPE for getting eigenvalues of a matrix."""

from typing import Optional, List
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.initial_states import Custom
from .eigs import Eigenvalues

class BBEigs(Eigenvalues):
    """ Eigenvalues using a black box.

    Calculates the eigenvalues of the matrix classically and encodes the estimation
    in the given register
    """

    def __init__(self,
                 matrix: np.array,
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

        self._matrix = matrix
        self._num_ancillae = num_ancillae
        self._negative_evals = negative_evals
        self._num_q = int(np.log2(np.shape(matrix)[0]))

    def get_register_sizes(self):
        return self._num_q, self._num_ancillae

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
        
        # TODO Implement calculation of the eigenvalues


        # TODO Create circuit using CUSTOM