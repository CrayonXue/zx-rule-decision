# resetter_graph_bank.py
import pickle
import numpy as np
import sys
sys.path.append('../pyzx_copy')
import pyzx_copy as zx_copy

class Resetter_GraphBank:
    def __init__(self, bank_path: str, seed: int | None = None, shuffle: bool = True):
        graph_bank_path = bank_path + '.pkl'
        circuit_bank_path = bank_path + '_circuits.pkl'
        with open(graph_bank_path, "rb") as f:
            self.graphs = pickle.load(f)  # list[pyzx Graph]
        with open(circuit_bank_path, "rb") as f:
            self.circuits = pickle.load(f)  # list[pyzx Circuit]

        if not isinstance(self.graphs, list) or len(self.graphs) == 0:
            raise ValueError("Empty or invalid graph bank.")
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self._idx = 0
        if self.shuffle:
            self.rng.shuffle(self.graphs)

    def reset(self):
        # sample
        if self.shuffle:
            idx = self.rng.integers(0, len(self.graphs))  
        else:
            idx = self._idx % len(self.graphs)
            self._idx += 1
        g = self.graphs[idx]
        c = self.circuits[idx]
        # Important: return a fresh copy so env mutations don’t leak
        return g.copy(),c
