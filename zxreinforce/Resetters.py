# resetter_graph_bank.py
import pickle
import numpy as np
import sys
sys.path.append('../pyzx_copy')
import pyzx_copy as zx_copy

class Resetter_GraphBank:
    def __init__(self, bank_path: str, seed: int | None = None, shuffle: bool = True):
        with open(bank_path, "rb") as f:
            self.graphs = pickle.load(f)  # list[pyzx Graph]
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
            g = self.graphs[self.rng.integers(0, len(self.graphs))]
        else:
            g = self.graphs[self._idx % len(self.graphs)]
            self._idx += 1
        # Important: return a fresh copy so env mutations don’t leak
        return g.copy(), None  # env never uses the circuit downstream
