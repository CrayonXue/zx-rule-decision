
# This file contains the ZXCalculus environment class and functions to apply actions to the diagram
import copy
import pickle

import numpy as np

from collections import deque

from .own_constants import (INPUT, OUTPUT, GREEN, RED, HADAMARD, N_NODE_ACTIONS, N_EDGE_ACTIONS, encode_phase)

import sys
sys.path.append('../pyzx_copy')
import pyzx_copy as zx_copy
from pyzx_copy.utils import VertexType, toggle_vertex
from pyzx_copy.utils import heuristic_fixed_pairs_node_disjoint, broken_paths_for_unrouted_pairs, to_networkx_graph
from pyzx_copy.basicrules import pi_commute_Z,pi_commute_X,fuse

class ZXCalculus():
    """Class for single ZX-calculus environment"""
    def __init__(self,
                 max_steps:int=1000, 
                 resetter=None,
                 count_down_from:int=20,
                 step_penalty:float=0.01,
                 length_penalty:float=0.001,
                 extra_state_info:bool=False,
                 adapted_reward:bool=False):
        """max_steps: maximum number of steps per trajectory,
        resetter: object that can reset the environment,
        count_down_from: start stop counter from this number,
        """
        
        self.max_steps = max_steps        
        self.resetter = resetter
        self.count_down_from = count_down_from
        self.extra_state_info = extra_state_info
        self.adapted_reward = adapted_reward
        self.step_penalty = step_penalty
        self.length_penalty = length_penalty

 

    def get_observation(self)-> tuple:
        """ observation
        Get data,mask from the environment's graph"""
        nodes_features = []
        for v in self.graph.vertices():
            if v in self.graph.inputs():
                node_type = INPUT
                node_phase = encode_phase(None)
            elif v in self.graph.outputs():
                node_type = OUTPUT
                node_phase = encode_phase(None)
            elif self.graph.type(v) == VertexType.Z:
                node_type = GREEN
                node_phase = encode_phase(self.graph.phase(v))
            elif self.graph.type(v) == VertexType.X:
                node_type = RED
                node_phase = encode_phase(self.graph.phase(v))
            elif self.graph.type(v) == VertexType.H_BOX:
                node_type = HADAMARD
                node_phase = encode_phase(None)
            
            
            node_feasture = node_type + node_phase + [self.graph.qubit(v)] #5+10+1
            nodes_features.append(node_feasture)
        x = torch.tensor(nodes_features, dtype=torch.float32)

        edge_list = list(self.graph.edges())

        if not edge_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            uv = [(self.id2pos[int(u)], self.id2pos[int(v)]) for u, v in edge_list]
            edge_index = torch.tensor(uv, dtype=torch.long).t().contiguous()

        # Precompute feasibility mask (variable length: N*node_act + E*edge_act + 1)
        mask = torch.from_numpy(self.get_action_mask().astype(np.int8))  # [A_var]
        data = Data(
            x=x, 
            edge_index=edge_index)
        return data, mask

    def is_terminal(self) -> bool:
        return (self.current_left_nodes_by_continuous == 0
                and self.current_high_degree_nodes == 0)

    def reset(self)-> tuple:
        '''
        returns: (data,mask)
        Resets the environment:
        1. Sample ZX_diagram.
        2. Set the step counter to zero.
        Return new initial observation
        '''
        self.graph, _ = self.resetter.reset()
        self.step_counter = 0

        self.current_left_nodes_by_continuous, self.current_left_nodes_by_continuous_and_broken = self.num_nodes_left
        self.current_high_degree_nodes = self.count_high_degree_nodes()
        self.last_score = self.compute_dense_reward()
        return self.get_observation()
    


    def step(self, action: int) -> tuple[int, int]:
        '''action: int, action to be applied to the environment,
        returns: (data,mask, reward, done)
        '''

        ## Add the step counter for stopping criteria
        self.step_counter += 1
        self.graph = self.graph.copy()
        # Check if trajectory is over
        if self.step_counter >= self.max_steps: 
            # Return observation and reward, end_episode
            data,mask = self.reset()
            return data,mask, 0, 1
        
        else:
            # Applies the action    
            success = self.apply_action(action)
            if not success:
                # Penalize invalid or no-op actions
                data, mask = self.get_observation()
                return data, mask, -0.1, 0 # Small penalty, not done
            data, mask = self.get_observation()

            ## Calculate the reward
            self.current_left_nodes_by_continuous, self.current_left_nodes_by_continuous_and_broken = self.num_nodes_left
            self.current_high_degree_nodes = self.count_high_degree_nodes()

            done = 1 if self.is_terminal() else 0

            # reward = delta_left_continuous + 0.2*delta_left_cont_and_broken + self.delta_high_degree_nodes()
            # reward = -self.current_left_nodes_by_continuous - 0.2*self.current_left_nodes_by_continuous_and_broken - self.current_high_degree_nodes
            new_score = self.compute_dense_reward()
            reward = new_score - self.last_score
            self.last_score = new_score  # Update the last_score for the next step
            reward = reward - self.step_penalty # Apply penalties
            
            if done:
                reward+=1
            return data,mask, reward, done
    

    # =================Reward calculation=========================
    def compute_dense_reward(self) -> float:
        # Your existing dense reward formula, but callable at any time
        n = max(1, self.n_spiders)
        # return (
        #     0.45*(self.n_spiders - self.current_left_nodes_by_continuous)/n
        #     + 0.10*(self.n_spiders - self.current_left_nodes_by_continuous_and_broken)/n
        #     + 0.45*(self.n_spiders - self.current_high_degree_nodes)/n
        # )
        return (
            0.5*(self.n_spiders - self.current_left_nodes_by_continuous)/n
            + 0.5*(self.n_spiders - self.current_high_degree_nodes)/n
        )


    @property
    def n_spiders(self)->int:
        """Number of nodes in diagram"""
        return len(self.graph.vertices())
    
    @property
    def n_edges(self):
        """Number of edges in diagram"""
        return len(list(self.graph.edges()))

    @property
    def id2pos(self):
        return {int(v_id): i for i, v_id in enumerate(self.graph.vertices())}

    def count_high_degree_nodes(self, degree_threshold: int = 3) -> int:
        return sum(1 for v in self.graph.vertices() if len(self.graph.neighbors(v)) > degree_threshold)


    @property
    def num_nodes_left(self):

        G_nx = to_networkx_graph(self.graph)
        pairs = list(zip(self.graph.inputs(), self.graph.outputs()))  # fixed (si, ti)

        # path connecting inputs to outputs
        continuous_paths_dict = heuristic_fixed_pairs_node_disjoint(G_nx, pairs, iterations=10, restarts=5, seed=42)
        used_nodes = set().union(*continuous_paths_dict.values())
        left_nodes_by_continuous = set(self.graph.vertices()) - used_nodes # nodes not in any continuous path

        broken = broken_paths_for_unrouted_pairs(
            G_nx, continuous_paths_dict, pairs,
            include_endpoints_as_used=True,  # set False if endpoints of accepted paths may be reused
            edge_cost=1e-3,                  # tie-breaker toward shorter routes
            forbid_cost=1.0                  # cost per used (forbidden) node
        )
        broken_paths = []
        for (s,t), info in broken.items():
            broken_paths.append(sum(info["segments"], []))
        left_nodes_by_continuous_and_broken = left_nodes_by_continuous - set().union(*broken_paths) # nodes not in any continuous or broken path

        return len(left_nodes_by_continuous),len(left_nodes_by_continuous_and_broken)


    # ==================Actions=========================
    def apply_action(self, action:int):
        self.graph = self.graph.copy()
        self.node_actions_fn = [self.unfuse_rule, self.color_change_rule, self.split_hadamard, self.pi_rule]

        if action < N_NODE_ACTIONS * self.n_spiders:
            # Action is node action
            node_idx = action//N_NODE_ACTIONS
            action_idx = action % N_NODE_ACTIONS
            if action_idx < 2**5:
                success = self.unfuse_rule(node_idx,action_idx)
            else:
                success = self.node_actions_fn[action_idx%(2**5)+1](node_idx)
        else:
            raise ValueError(f"Action {action} not recognized")
        self.graph = self.graph.copy()
        return success

    # ----------------node action-----------------------
    def unfuse_rule(self, node_idx, action_idx):
        """Unfuse action"""
        v = list(self.graph.vertices())[node_idx]
        if not (self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X):
            return False
        child_idx = len(self.graph.vertices())
        self.graph.add_vertex(ty=self.graph.type(v), 
                    phase=0,
                    qubit=self.graph.qubit(v),
                    index=child_idx)
        
        # move selected neighbours of action node to child node
        neighbours = list(self.graph.neighbors(v))
        n=0
        for neighbor in neighbours:
            if get_bit(action_idx, n):
                self.graph.remove_edge((v, neighbor))
                self.graph.add_edge((child_idx, neighbor))
            n += 1
        self.graph.add_edge((v, child_idx))

        # reset graph's rows and qubits
        child_neighbors = list(self.graph.neighbors(child_idx))
        neighbor_qubit = [self.graph.qubit(child_neighbors[i]) for i in range(len(child_neighbors))]
        neighbor_row = [self.graph.row(child_neighbors[i]) for i in range(len(child_neighbors))]
        child_qubit = determine_qubit(neighbor_qubit)
        child_row = determine_row(neighbor_row)
        self.graph.set_qubit(child_idx, child_qubit)
        self.graph.set_row(child_idx, child_row)
        # move all rows >= child_row up by 1
        for v in self.graph.vertices():
            if self.graph.row(v) >= child_row and v != child_idx:
                self.graph.set_row(v, self.graph.row(v)+1)
        return True



    def color_change_rule(self, node_idx):
        v = list(self.graph.vertices())[node_idx]
        if not (self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X):
            return False
        self.graph.set_type(v, toggle_vertex(self.graph.type(v)))
        neighbours = list(self.graph.neighbors(v))
        for n in neighbours:
            self.graph.remove_edge((v,n))
            h = self.graph.add_vertex(ty=VertexType.H_BOX,qubit=determine_qubit([self.graph.qubit(n),self.graph.qubit(v)]),row=determine_row([self.graph.row(n),self.graph.row(v)]))
            self.graph.add_edge((v,h))
            self.graph.add_edge((h,n))
        return True

    def split_hadamard(self, node_idx):
        """Hadamard unfuse action"""
        # Change middle node
        v = list(self.graph.vertices())[node_idx]
        if self.graph.type(v) != VertexType.H_BOX:
            return False
        neighbours = list(self.graph.neighbors(v))
        if len(neighbours) != 2:
            raise ValueError(f"Hadamard node v={v} requires exactly two neighbors.")

        for w in self.graph.vertices():
            if self.graph.row(w) >= self.graph.row(v) and w != v:
                self.graph.set_row(w, self.graph.row(w)+3)
        u1 = self.graph.add_vertex(ty=VertexType.X, phase=1/2, qubit=self.graph.qubit(v), row=self.graph.row(v))
        u2 = self.graph.add_vertex(ty=VertexType.Z, phase=1/2, qubit=self.graph.qubit(v), row=self.graph.row(v)+1)
        u3 = self.graph.add_vertex(ty=VertexType.X, phase=1/2, qubit=self.graph.qubit(v), row=self.graph.row(v)+2)

        self.graph.add_edge((neighbours[0], u1))
        self.graph.add_edge((neighbours[1], u3))
        self.graph.add_edge((u1, u2))
        self.graph.add_edge((u2, u3))
        self.graph.remove_vertex(v)
        
        return True

    def pi_rule(self, node_idx):
        """node action"""
        v = list(self.graph.vertices())[node_idx]
        if (self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X):
            if self.graph.type(v) == VertexType.Z:
                pi_commute_Z(self.graph, v)
            elif self.graph.type(v) == VertexType.X:
                pi_commute_X(self.graph, v)
            return True
        else:
            return False

    # ----------------edge action-----------------------
    def fuse_rule(self, edge_idx):
        """Fuse action"""
        s,t = list(self.graph.edges())[edge_idx]
        return fuse(self.graph, s, t)

    def bialgebra_rule(self, edge_idx):
        v0,v1 = list(self.graph.edges())[edge_idx]

        v0t = self.graph.type(v0)
        v1t = self.graph.type(v1)
        v0p = self.graph.phase(v0)
        v1p = self.graph.phase(v1)
        n_spiders = len(list(self.graph.vertices()))
        if (v0p == 0 and v1p == 0 and
        ((v0t == VertexType.Z and v1t == VertexType.X) or (v0t == VertexType.X and v1t == VertexType.Z))):
            v0n = [n for n in self.graph.neighbors(v0) if not n == v1]
            v1n = [n for n in self.graph.neighbors(v1) if not n == v0]

            if (len(v0n) == 2 and len(v1n) == 2 and
                self.graph.num_edges(v0, v1) == 1 and # there is exactly one edge between v0 and v1
                self.graph.num_edges(v0, v0) == 0 and # there are no self-loops on v0
                self.graph.num_edges(v1, v1) == 0): # there are no self-loops on v1
                v01 = v0n[1]
                v11 = v1n[1]

                self.graph.set_type(v0, v1t)
                self.graph.set_type(v1, v0t)
                self.graph.remove_edge((v0,v01))
                self.graph.remove_edge((v1,v11))
                a = self.graph.add_vertex(ty=v1t, phase=0)
                b = self.graph.add_vertex(ty=v0t, phase=0)
                self.graph.add_edge((v01,a))
                self.graph.add_edge((v11,b))
                self.graph.add_edge((a,b))
                self.graph.add_edge((v0,b))
                self.graph.add_edge((v1,a))
                return True
            else:
                return False
        else:
            return False

    # ==================Action mask=========================
    def get_action_mask(self) -> np.ndarray:
        cut = N_NODE_ACTIONS*self.n_spiders  
        mask = np.zeros(cut, dtype=np.int32)
        for i in range(cut):
            node_idx = i // N_NODE_ACTIONS
            action_idx = i % N_NODE_ACTIONS
            v = list(self.graph.vertices())[node_idx]
            if action_idx < 2**5: # unfuse_rule
                if self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X:
                    mask[i] = 1
            elif action_idx % 2**5== 0: # color_change_rule
                if self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X:
                    mask[i] = 1
            elif action_idx % 2**5 == 1: # split_hadamard
                if self.graph.type(v) == VertexType.H_BOX:
                    mask[i] = 1
            elif action_idx % 2**5== 2: # pi_rule
                if self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X:
                    mask[i] = 1
        return mask


def save(colors:np.ndarray, angles:np.ndarray, 
         source:np.ndarray, target:np.ndarray, idx:int):
    """saves the current state of the environment at step idx"""
    with open(f"colors{idx}.pkl", 'wb') as f:
        pickle.dump(colors, f)
    with open(f"angles{idx}.pkl", 'wb') as f:
        pickle.dump(angles, f)
    with open(f"source{idx}.pkl", 'wb') as f:
        pickle.dump(source, f)
    with open(f"target{idx}.pkl", 'wb') as f:
        pickle.dump(target, f)




# The following functions are stand-alone functions to potentially make them jit compatible in the future
#Actions--------------------------------------------------------------------------------------------


from collections import Counter
from typing import List
import numpy as np
import torch 
from torch_geometric.data import Data

def determine_qubit(neighbor_qubit:List[int]) -> int:
    """Determine the qubit for the new child node based on its neighbors' qubits.
    
    If all neighbors are on the same qubit, return that qubit.
    If neighbors are on different qubits, return None (indicating ambiguity).
    """
    counts = Counter(neighbor_qubit)
    max_freq = max(counts.values())
    candidates = [item for item, freq in counts.items() if freq == max_freq]
    chosen_qubit = np.random.choice(candidates)
    return chosen_qubit

def determine_row(neighbor_row:List[int]) -> int:
    """Determine the row for the new child node based on its neighbors' rows.
    
    The new node's row is set to one more than the maximum row among its neighbors.
    """
    return max(neighbor_row)

def pyzx_to_pyg(zx_graph, num_qubits: int):
    """
    Encode qubit number index in the node feature for boundary vertex
    """
    node_features = []
    vertices = sorted(list(zx_graph.vertices()))
    
    inputs = zx_graph.inputs()
    outputs = zx_graph.outputs()
    
    # Acquiring qubit index for each boundary vertex
    # from PyZX
    input_q_map = {v: zx_graph.qubit(v) for v in inputs}
    output_q_map = {v: zx_graph.qubit(v) for v in outputs}
    
    boundary_feature_dim = 2 * num_qubits + 1
    
    for v in vertices:
        # Spider type (Z/X) - [2]
        stype = zx_graph.type(v)
        type_encoding = [1.0, 0.0] if stype == VertexType.Z else [0.0, 1.0]

        # Boundary and position encoding - [2 * N_qubits + 1]
        boundary_encoding = [0.0] * boundary_feature_dim
        if v in inputs:
            q_idx = input_q_map[v]
            boundary_encoding[q_idx] = 1.0
        elif v in outputs:
            q_idx = output_q_map[v]
            boundary_encoding[num_qubits + q_idx] = 1.0
        else: # Internal node
            boundary_encoding[-1] = 1.0

        # Phase encoding (sin/cos) - [2]
        phase = float(zx_graph.phase(v)) * np.pi
        phase_encoding = [np.sin(phase), np.cos(phase)]
        
        node_features.append(type_encoding + boundary_encoding + phase_encoding)

    x = torch.tensor(node_features, dtype=torch.float)

    edge_list = list(zx_graph.edges())
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def get_bit(n: int, idx: int) -> int:
    """Return the bit at index idx (LSB=0) for integer n. Works for n >= 0."""
    if idx < 0:
        raise ValueError("idx must be non-negative")
    return (n >> idx) & 1