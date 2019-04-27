# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

import logging

import numpy as np
from qiskit import QuantumCircuit

from qiskit.aqua.utils.arithmetic import next_power_of_2_base
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.aqua.circuits import StateVectorCircuit

logger = logging.getLogger(__name__)


class RawFeatureVector(FeatureMap):
    """
    Using raw feature vector as the initial state vector
    """

    CONFIGURATION = {
        'name': 'RawFeatureVector',
        'description': 'Raw feature vector',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'raw_feature_vector_schema',
            'type': 'object',
            'properties': {
                'feature_dimension': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, feature_dimension=2):
        """Constructor.

        Args:
            feature_vector: The raw feature vector
        """
        self.validate(locals())
        super().__init__()
        self._feature_dimension = feature_dimension
        self._num_qubits = next_power_of_2_base(feature_dimension)

    def construct_circuit(self, x, qr=None, inverse=False):
        """
        Construct the second order expansion based on given data.

        Args:
            x (numpy.ndarray): 1-D to-be-encoded data.
            qr (QauntumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.

        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be numpy array.")
        if x.ndim != 1:
            raise ValueError("x must be 1-D array.")
        if x.shape[0] != self._feature_dimension:
            raise ValueError("Unexpected feature vector dimension.")

        state_vector = np.pad(x, (0, (1 << self.num_qubits) - len(x)), 'constant')

        svc = StateVectorCircuit(state_vector)
        return svc.construct_circuit(register=qr)
