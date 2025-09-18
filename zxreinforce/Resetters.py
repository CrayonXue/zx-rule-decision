# Holds resetters for the ZX env returning 
# (colors, angles, selected_node, source, target, selected_edges)

import numpy as np
from typing import Optional, List, Set, Union
from fractions import Fraction
from typing import Union

import sys
sys.path.append('../pyzx_copy')
import pyzx_copy as zx_copy



class Resetter_Circuit():
    def __init__(self, 
                 num_qubits_min:int,
                 num_qubits_max:int,
                 min_gates:int,
                 max_gates:int,
                 p_t: float = 0.2,
                 p_h: float = 0.2,
                 seed:Optional[int]=None):
        """generate a random circuit: pyzx circuit then convert to graph"""
        self.num_qubits_min = num_qubits_min
        self.num_qubits_max = num_qubits_max
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.p_t = p_t
        self.p_h = p_h
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self)->tuple:
        n_qubits = self.rng.integers(low=self.num_qubits_min, high=self.num_qubits_max+1)
        n_gates = self.rng.integers(low=self.min_gates, high=self.max_gates+1)
        self.circuit: zx_copy.Circuit = zx_copy.generate.CNOT_HAD_PHASE_circuit(n_qubits, n_gates, p_had=self.p_h, p_t=self.p_t)
        self.graph: zx_copy.Graph = self.circuit.to_graph().copy()
        zx_copy.full_reduce(self.graph)
        return self.graph, self.circuit
    