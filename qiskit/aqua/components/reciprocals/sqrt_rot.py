# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Controlled direct rotation for the HHL algorithm based on partial table lookup"""

from typing import Optional
import itertools
import logging
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.utils.validation import validate_range

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class SqrtRotation(LookupRotation):

    """
    The Lookup Rotation for Non-Reciprocals.

    This method applies a variable sized binning to the values. Only a specified number of bits
    after the most-significant bit is taken into account when assigning rotation angles to the
    numbers prepared as states in the input register. Using precomputed angles, the reciprocal
    is multiplied to the amplitude via controlled rotations. While no resolution of the result
    is lost for small values, towards larger values the bin size increases. The accuracy of the
    result is tuned by the parameters.

    A calculation of reciprocals of eigenvalues is performed and controlled rotation of ancillary
    qubit via a lookup method. It uses a partial table lookup of rotation angles to rotate an
    ancillary qubit by arcsin(C * lambda).
    """

    def __init__(
            self,
            pat_length: Optional[int] = None,
            subpat_length: Optional[int] = None,
            scale: float = 0,
            negative_evals: bool = False,
            evo_time: Optional[float] = None,
            lambda_min: Optional[float] = None) -> None:
        r"""
        Args:
            pat_length: The number of qubits used for binning pattern. Specifies the number of bits
                following the most-significant bit that is used to identify a number. This leads to
                a binning of large values, while preserving the accuracy for smaller values. It
                should be chosen as :math:`min(k-1,5)` for an input register with k qubits to limit
                the error in the rotation to < 3%.
            subpat_length: The number of qubits used for binning sub-pattern. This parameter is
                computed in the circuit creation routine and helps reducing the gate count.
                For `pat_length<=5` it is chosen as
                :math:`\left\lceil(\frac{patlength}{2})\right\rceil`.
            scale: The scale of rotation angle, corresponds to HHL constant C,
                has values between 0 and 1. This parameter is used to scale the reciprocals such
                that for a scale C, the rotation is performed by an angle
                :math:`\arcsin{\frac{C}{\lambda}}`. If neither the `scale` nor the
                `evo_time` and `lambda_min` parameters are specified, the smallest resolvable
                Eigenvalue is used.
            negative_evals: Indicate if negative eigenvalues need to be handled
            evo_time: The evolution time. This parameter scales the Eigenvalues in the phase
                estimation onto the range (0,1] ( (-0.5,0.5] for negative Eigenvalues ).
            lambda_min: The smallest expected eigenvalue
        """
        validate_range('scale', scale, 0, 1)
        super().__init__(pat_length=pat_length,
                            subpat_length=subpat_length,
                            scale=scale,
                            negative_evals=negative_evals,
                            evo_time=evo_time,
                            lambda_min=lambda_min)
    
    def sv_to_resvec(self, statevector, num_q, index_ancilla, qdf=False):
        half = int(len(statevector) / 2)
        print(half)
        if not qdf:
            # Ignore ancilla qubit
            start_idx = half 
        else:
            # Ignore 2 ancilla qubits
            start_idx = half + int(half/2) 
        return statevector[start_idx:start_idx + 2 ** num_q]

    def construct_circuit(self, mode, inreg):  # pylint: disable=arguments-differ
        """Construct the Lookup Rotation circuit.

        Args:
            mode (str): construction mode, 'matrix' not supported
            inreg (QuantumRegister): input register, typically output register of Eigenvalues

        Returns:
            QuantumCircuit: containing the Lookup Rotation circuit.
         Raises:
            NotImplementedError: mode not supported
        """

        # initialize circuit
        if mode == 'matrix':
            raise NotImplementedError('The matrix mode is not supported.')
        if self._lambda_min:
            self._scale = self._lambda_min / 2 / np.pi * self._evo_time
        if self._scale == 0:
            self._scale = 2**-len(inreg)
        self._ev = inreg
        self._workq = QuantumRegister(1, 'work')
        self._msq = QuantumRegister(1, 'msq')
        self._anc = QuantumRegister(1, 'anc_direct')
        qc = QuantumCircuit(self._ev, self._workq, self._msq, self._anc)
        self._circuit = qc
        self._reg_size = len(inreg)
        if self._pat_length is None:
            if self._reg_size <= 6:
                self._pat_length = self._reg_size - \
                    (2 if self._negative_evals else 1)
            else:
                self._pat_length = 5
        if self._reg_size <= self._pat_length:
            self._pat_length = self._reg_size - \
                (2 if self._negative_evals else 1)
        if self._subpat_length is None:
            self._subpat_length = int(np.ceil(self._pat_length / 2))
        m = self._subpat_length
        n = self._pat_length
        k = self._reg_size
        neg_evals = self._negative_evals

        # get classically precomputed eigenvalue binning
        approx_dict = LookupRotation._classic_approx(k, n, m,
                                                     negative_evals=neg_evals)

        fo = None
        old_fo = None
        # for negative EV, we pass a pseudo register ev[1:] ign. sign bit
        ev = [self._ev[i] for i in range(len(self._ev))]

        for _, fo in enumerate(list(approx_dict.keys())):
            # read m-bit and (n-m) bit patterns for current first-one and
            # correct Lambdas
            pattern_map = approx_dict[fo]
            # set most-significant-qbit register and uncompute previous
            if self._negative_evals:
                if old_fo != fo:
                    if old_fo is not None:
                        self._set_msq(self._msq, ev[1:], int(old_fo - 1))
                    old_fo = fo
                    if fo + n == k:
                        self._set_msq(self._msq, ev[1:], int(fo - 1),
                                      last_iteration=True)
                    else:
                        self._set_msq(self._msq, ev[1:], int(fo - 1),
                                      last_iteration=False)
            else:
                if old_fo != fo:
                    if old_fo is not None:
                        self._set_msq(self._msq, self._ev, int(old_fo))
                    old_fo = fo
                    if fo + n == k:
                        self._set_msq(self._msq, self._ev, int(fo),
                                      last_iteration=True)
                    else:
                        self._set_msq(self._msq, self._ev, int(fo),
                                      last_iteration=False)
            # offset = start idx for ncx gate setting and unsetting m-bit
            # long bitstring
            offset_mpat = fo + (n - m) if fo < k - n else fo + n - m - 1
            for mainpat, subpat, lambda_ar in pattern_map:
                # set m-bit pattern in register workq
                self._set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)
                # iterate of all 2**(n-m) combinations for fixed m-bit
                for subpattern, lambda_ in zip(subpat, lambda_ar):

                    # calculate rotation angle
                    theta = 2 * np.arcsin(min(1, 2 * self._scale * np.sqrt(lambda_)))
                    # offset for ncx gate checking subpattern
                    offset = fo + 1 if fo < k - n else fo

                    # rotation is happening here
                    # 1. rotate by half angle
                    qc.mcry(theta / 2, [self._workq[0], self._msq[0]],
                            self._anc[0], None, mode='noancilla')
                    # 2. mct gate to reverse rotation direction
                    self._set_bit_pattern(subpattern, self._anc[0], offset)
                    # 3. rotate by inverse of halfangle to uncompute / complete
                    qc.mcry(-theta / 2, [self._workq[0], self._msq[0]],
                            self._anc[0], None, mode='noancilla')
                    # 4. mct gate to uncompute first mct gate
                    self._set_bit_pattern(subpattern, self._anc[0], offset)
                # uncompute m-bit pattern
                self._set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)

        last_fo = fo
        # uncompute msq register
        if self._negative_evals:
            self._set_msq(self._msq, ev[1:], int(last_fo - 1),
                          last_iteration=True)
        else:
            self._set_msq(self._msq, self._ev, int(last_fo),
                          last_iteration=True)

        # rotate by pi to fix sign for negative evals
        if self._negative_evals:
            qc.cry(2 * np.pi, self._ev[0], self._anc[0])
        
        qc.barrier(self._msq)
        qc.barrier(self._workq)
        self._circuit = qc
        return self._circuit