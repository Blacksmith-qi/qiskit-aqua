"""
Contains class to perform swap test
"""

from typing import Optional, Union, Dict, Any, Tuple
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.aqua.algorithms.quantum_algorithm import QuantumAlgorithm
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.result import Result

from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance




class SwapTest(QuantumAlgorithm):
    """
    Class to create and evaluate an swap test on two registers
    """

    def __init__(self,
                 qreg1: QuantumRegister,
                 qreg2: QuantumRegister,
                 quantum_instance: Optional[
                    Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:

        """
        Args:
            qreg1: First of the two registers to be swapped
            qreg2: Second of the two registers to be swapped
        """

        super().__init__(quantum_instance)

        if qreg1.size is not qreg2.size:
            raise ValueError('Registers to swap have different size')

        self.qreg1 = qreg1
        self.qreg2 = qreg2
        self.regsize = qreg1.size
        
    

    def construct_circuit(self) -> QuantumCircuit:
        """ Constructs the swap test circuit
        
        Args:

        Returns:
            QuantumCircuit: object for the swap test circuit
        """
        
        qc = QuantumCircuit(self.qreg1, self.qreg2)
        # Adding c-nots
        for idx in range(self.regsize):
            qc.cnot(self.qreg2[idx], self.qreg1[idx])
            qc.h(self.qreg2[idx])

        qc.measure_all()

        self.qc = qc
        return self.qc

    def _run(self) -> Result:
        return super()._run()