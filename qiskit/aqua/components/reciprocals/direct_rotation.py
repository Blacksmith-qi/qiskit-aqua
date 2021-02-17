""" Controlled rotatino withou the inverse. Needed for Quantum Data fitting """


from typing import Optional
import itertools
import logging
import numpy as np
from math import ceil, isclose

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
            error: Optional[float] = 0.001) -> None:
        r"""
        Args:
            lambda_max: The biggest expected eigenvalue
            error: Max error E = |angle - sind(angle)|
        """

        super().__init__()
        self._lambda_max = lambda_max
        self._error = error
        


    def sv_to_resvec(self, statevector, num_q, index_ancilla, qdf=False):
        half = int(len(statevector) / 2)
        if not qdf:
            # Ignore ancilla qubit
            start_idx = half 
        else:
            # Ignore 2 ancilla qubits
            start_idx = half + 2 ** index_ancilla 
        return statevector[start_idx:start_idx + 2 ** num_q]

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
        
        # Createing the circuit base
        self._ev = inreg
        self._anc = QuantumRegister(1, 'anc_direct')
        qc = QuantumCircuit(inreg, self._anc) 
        self._circuit = qc
        self._reg_size = len(inreg)


        # calculate angle to achive desired error
        max_angle = np.pi
        while(not isclose(max_angle, np.sin(max_angle), rel_tol=self._error)):
            max_angle = max_angle/2



        for bit in range(self._reg_size):
            qc_temp = QuantumCircuit(1)
            angle = 2 * max_angle  / 2 ** bit
            qc_temp.ry(angle, 0)
            added_rotation_gate = qc_temp.to_gate(
                                        label='rot bit ' + str(bit))
            controlled_gate = added_rotation_gate.control()
            self._circuit.append(controlled_gate, 
                            [self._ev[bit],
                            self._anc])



        return self._circuit



