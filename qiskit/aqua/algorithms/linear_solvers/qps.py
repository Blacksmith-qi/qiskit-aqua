
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

class QPS(QDF):
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
        super().__init__(matrix,
                        vector,
                        truncate_powerdim,
                        truncate_hermitian,
                        eigs,
                        init_state,
                        rotation1,
                        num_q,
                        num_a,
                        orig_size[1],
                        quantum_instance)
        self._eigs2 = eigs2
        self._rotation_inverse = rotation2
        self._save_name = save_name
 

                



