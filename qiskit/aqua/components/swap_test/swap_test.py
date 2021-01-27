"""
Contains class to perform swap test
"""

from typing import Optional
from qiskit import QuantumRegister, ClassicalRegister

class SwapTest():
    """
    Class to create and evaluate an swap test on two registers
    """

    def __init__(self,
                 qreg1: QuantumRegister,
                 qreg2: QuantumRegister,
                 creg: Optional[ClassicalRegister] = None) -> None:
        """
        Args:
            qreg1: First of the two registers to be swapped
            qreg2: Second of the two registers to be swapped
            creg: Classical register to store the result of the test
        """

        self.qreg1 = qreg1
        self.qreg2 = qreg2
        
        if creg is not None:
            self.creg = creg
        else:
            self.creg = ClassicalRegister(1)