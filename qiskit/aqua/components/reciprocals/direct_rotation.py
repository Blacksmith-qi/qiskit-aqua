""" Controlled rotatino withou the inverse. Needed for Quantum Data fitting """


from typing import Optional
import itertools
import logging
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.reciprocals import Reciprocal
from qiskit.aqua.utils.validation import validate_range

logger = logging.getLogger(__name__)


class DirectRotation(Reciprocal):
    """
    Thist class is used to perform a controlled rotation based on a register
    which contains the rotation ange in binary form.

    To perform the rotation and overcome the inlinearity of the rotation
    (ry(theta) rotates about sin(theta)) the Taylor expansion of the sin is
    used. 

    sin(x) /approx x - x^2/2 + ...

    Therefore the rotation is broken down to multiple small rotation where
    sin is approximal x

    """

    def __init__(
            self,
            lambda_max: Optional[float] = None,
            max_amplitude: Optional[float] = 0.25) -> None:
        r"""
        Args:
            lambda_max: The biggest expected eigenvalue
            max_amplitude: The wanted scale of the resulting amplitude of the
            biggest eigenvalue will be
        """

        super().__init__()
        self.lambda_max = lambda_max
        self.max_amplitude = max_amplitude


    def sv_to_resvec(self, statevector, num_q):
        return statevector

    def construct_circuit(self, mode, inreg):
        """Construct the Direct Rotation circuit.

        Args:
            mode (str): construction mode, 'matrix' not supported
            inreg (QuantumRegister): input register, typically output register of Eigenvalues

        Returns:
            QuantumCircuit: containing the Direct Rotation circuit.
         Raises:
            NotImplementedError: mode not supported
        """

        self._ev = inreg
        self._anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(self._ev, self._anc)
        self._circuit = qc
        self._reg_size = len(inreg)




        return self._circuit



