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
                 start_circuit: QuantumCircuit,
                 qreg1: QuantumRegister,
                 qreg2: QuantumRegister,
                 error: Optional[float] = 0.01,
                 quantum_instance: Optional[
                    Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:

        """
        Args:
            start_circuit: Circuit to append the swap test onto
            qreg1: First of the two registers to be swapped
            qreg2: Second of the two registers to be swapped
            error: Error tolerance for the result
            quantum_instance: Quantum Instance or Backend
        """

        super().__init__(quantum_instance)

        if qreg1.size is not qreg2.size:
            raise ValueError('Registers to swap have different size')

        self._qreg1 = qreg1
        self._qreg2 = qreg2
        self._regsize = qreg1.size
        self._start_circuit = start_circuit
        self._circuit = None
        self._results = None
        self._error = error
        
    

    def construct_circuit(self) -> QuantumCircuit:
        """ Constructs the swap test circuit
        
        Args:

        Returns:
            QuantumCircuit: object for the swap test circuit
        """
        
        qc_0 = QuantumCircuit(self._qreg1, self._qreg2)

        qc = qc_0 + self._start_circuit

        # Adding c-nots
        for idx in range(self._regsize):
            qc.cnot(self._qreg2[idx], self._qreg1[idx])
            qc.h(self._qreg2[idx])

        qc.measure_all()

        self._circuit = qc
        return qc

    def _run(self) -> Result:
        if self._quantum_instance.is_statevector:
            raise TypeError('Statevector simulation not supported for swap test')
        if self._circuit is None:
            self.construct_circuit()

        # Change number of shots acording to error
        self._quantum_instance._run_config.shots = int(1 / self._error**2 )
        results = self._quantum_instance.execute(self._circuit)
        
        self._results = results

        return results

    def get_probaility(self) -> float:
        if self._results is None:
            self.run()

        counts = self._results.get_counts()

        # Starting post processing
        runs_pos = 0
        runs_neg = 0

        for state, hits in counts.items():
            number_11 = 0
            for idx in range(self._regsize):
                if state[idx] == '1' and state[idx + int(self._regsize)] == '1':
                    number_11 += 1
            print(state + '  ' + str(hits))
            # If pairs of 00 or 11 is even -> equals 0 in clas ancilla
            if number_11 % 2 == 0:
                runs_neg += hits
            else:
                runs_pos += hits

        prob = runs_pos / (runs_pos + runs_neg)
        return prob