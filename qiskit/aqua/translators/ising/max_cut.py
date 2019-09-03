# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Convert max-cut instances into Pauli list
Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
Design the max-cut object `w` as a two-dimensional np.array
e.g., w[i, j] = x means that the weight of a edge between i and j is x
Note that the weights are symmetric, i.e., w[j, i] = x always holds.
"""

import logging
import warnings

import numpy as np

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_qubit_op(weight_matrix):
    """Generate Hamiltonian for the max-cut problem of a graph.

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.

    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=np.bool)
                z_p = np.zeros(num_nodes, dtype=np.bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([0.5 * weight_matrix[i, j], Pauli(z_p, x_p)])
                shift -= 0.5 * weight_matrix[i, j]
    return WeightedPauliOperator(paulis=pauli_list), shift


def max_cut_value(x, w):
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    x_mat = np.outer(x, (1 - x))
    return np.sum(w * x_mat)


def get_graph_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x


def random_graph(n, weight_range=10, edge_prob=0.3, savefile=None, seed=None):
    """ random graph """
    from .common import random_graph as redirect_func
    warnings.warn("random_graph function has been moved to "
                  "qiskit.aqua.translators.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(n=n, weight_range=weight_range, edge_prob=edge_prob,
                         savefile=savefile, seed=seed)


def parse_gset_format(filename):
    """ parse gset format """
    from .common import parse_gset_format as redirect_func
    warnings.warn("parse_gset_format function has been moved to "
                  "qiskit.aqua.translators.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(filename)


def sample_most_likely(state_vector):
    """ sample most likely """
    from .common import sample_most_likely as redirect_func
    warnings.warn("sample_most_likely function has been moved to "
                  "qiskit.aqua.translators.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(state_vector=state_vector)


def get_gset_result(x):
    """ returns gset result """
    from .common import get_gset_result as redirect_func
    warnings.warn("get_gset_result function has been moved to "
                  "qiskit.aqua.translators.ising.common, "
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return redirect_func(x)


def get_max_cut_qubitops(weight_matrix):
    """ returns max cut qubit ops """
    warnings.warn("get_max_cut_qubitops function has been changed to get_qubit_op"
                  "the method here will be removed after Aqua 0.7+",
                  DeprecationWarning)
    return get_qubit_op(weight_matrix)
