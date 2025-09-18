import numpy as np
from gymnasium import spaces
from .own_constants import (N_NODE_ACTIONS, N_EDGE_ACTIONS)
class ZXGymWrapper:
    """
    Pads variable-size observations to fixed [max_nodes, max_edges] and exposes:
      obs: dict(
        node_ids [max_nodes, 1] (-1 padded),
        node_type [max_nodes, 5],
        node_phase [max_nodes, 10],
        node_selected [max_nodes, 1],
        qubit_on [max_nodes, 1] (int),
        edge_pairs [max_edges, 2] (int vertex positions, -1 padded),
        edge_selected [max_edges, 1],
        context [3],
        action_mask [A]   # A = 1 + max_nodes*N_NODE_ACTIONS + max_edges*N_EDGE_ACTIONS
      )
    Action is single Discrete(A) over:
      [ per-node actions ... | per-edge actions ... | STOP ]
    """
    def __init__(self, zx_env, max_nodes=256, max_edges=512, max_qubits=64):
        self.env = zx_env
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_qubits = max_qubits

        self.n_node_actions = N_NODE_ACTIONS
        self.n_edge_actions = N_EDGE_ACTIONS
        self.action_size = 1 + max_nodes*self.n_node_actions + max_edges*self.n_edge_actions

        self.observation_space = spaces.Dict({
            "node_ids": spaces.Box(-1, max_nodes-1, shape=(max_nodes, 1), dtype=np.int32),
            "node_type": spaces.Box(0, 1, shape=(max_nodes, 5), dtype=np.int32),
            "node_phase": spaces.Box(0, 1, shape=(max_nodes, 10), dtype=np.int32),
            "node_selected": spaces.Box(0, 1, shape=(max_nodes, 1), dtype=np.int32),
            "qubit_on": spaces.Box(0, max_qubits-1, shape=(max_nodes, 1), dtype=np.int32),
            "edge_pairs": spaces.Box(-1, max_nodes-1, shape=(max_edges, 2), dtype=np.int32),
            "edge_selected": spaces.Box(0, 1, shape=(max_edges, 1), dtype=np.int32),
            "context": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(self.action_size,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(self.action_size)

    def _pad(self, arr, target_shape, pad_value=0):
        out = np.full(target_shape, pad_value, dtype=arr.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        out[slices] = arr[slices]
        return out

    def _build_obs_and_mask(self, observation):
        node_ids, node_type, node_phase, qubit_on, node_sel, edge_pairs, edge_sel, n_nodes, n_edges, context = observation

        n = len(node_ids)
        e = len(edge_pairs)

        # Create mapping from node ID to positional index
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Convert edge pairs from node IDs to positional indices
        edge_pos = np.full((e, 2), -1, dtype=np.int32)
        for i, (u, v) in enumerate(edge_pairs):
            u_idx = id_to_idx.get(int(u), -1)
            v_idx = id_to_idx.get(int(v), -1)
            if u_idx != -1 and v_idx != -1:  # Only include valid edges
                edge_pos[i, 0] = u_idx
                edge_pos[i, 1] = v_idx

        obs = {
            "node_ids": self._pad(node_ids.reshape(-1,1), (self.max_nodes, 1), -1),
            "node_type": self._pad(node_type, (self.max_nodes, 5), 0),
            "node_phase": self._pad(node_phase, (self.max_nodes, 10), 0),
            "node_selected": self._pad(node_sel.reshape(-1,1), (self.max_nodes, 1), 0),
            "qubit_on": self._pad(qubit_on.reshape(-1,1), (self.max_nodes, 1), 0),
            "edge_pairs": self._pad(edge_pos, (self.max_edges, 2), -1),  # Use converted indices
            "edge_selected": self._pad(edge_sel.reshape(-1,1), (self.max_edges, 1), 0),
            "context": context.astype(np.float32),
        }

        # Action mask: valid node actions for first n nodes, valid edge actions for first e edges, STOP always valid
        mask = np.zeros(self.action_size, dtype=np.int8)
        # Node actions
        for i in range(n):
            base = i * self.n_node_actions
            mask[base:base+self.n_node_actions] = 1
        # Shift by node section
        edge_offset = self.max_nodes * self.n_node_actions
        for j in range(e):
            base = edge_offset + j * self.n_edge_actions
            mask[base:base+self.n_edge_actions] = 1
        # STOP
        mask[-1] = 1

        obs["action_mask"] = mask
        return obs

    def reset(self, seed=None):
        observation = self.env.reset()
        return self._build_obs_and_mask(observation)

    def step(self, action):
        # Map padded action to env action index
        # If STOP (last index)
        if action == self.action_size - 1:
            env_action = N_EDGE_ACTIONS * self.env.n_edges + N_NODE_ACTIONS * self.env.n_spiders
        else:
            if action < self.max_nodes * self.n_node_actions:
                node_idx = action // self.n_node_actions
                act_idx = action % self.n_node_actions
                # invalid node index -> force STOP
                if node_idx >= self.env.n_spiders:
                    env_action = N_EDGE_ACTIONS * self.env.n_edges + N_NODE_ACTIONS * self.env.n_spiders
                else:
                    env_action = node_idx * N_NODE_ACTIONS + act_idx
            else:
                a2 = action - self.max_nodes * self.n_node_actions
                edge_idx = a2 // self.n_edge_actions
                act_idx = a2 % self.n_edge_actions
                if edge_idx >= self.env.n_edges:
                    env_action = N_EDGE_ACTIONS * self.env.n_edges + N_NODE_ACTIONS * self.env.n_spiders
                else:
                    env_action = N_NODE_ACTIONS * self.env.n_spiders + edge_idx * N_EDGE_ACTIONS + act_idx

        observation, reward, done = self.env.step(env_action)
        obs = self._build_obs_and_mask(observation)
        info = {}
        return obs, float(reward), bool(done), info