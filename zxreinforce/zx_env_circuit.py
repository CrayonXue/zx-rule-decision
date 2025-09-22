
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
                 add_reward_per_step:float=-0.05,
                 resetter=None,
                 count_down_from:int=20,
                 dont_allow_stop:bool=False,
                 extra_state_info:bool=False,
                 adapted_reward:bool=False):
        """max_steps: maximum number of steps per trajectory,
        add_reward_per_step: reward added per step,
        resetter: object that can reset the environment,
        count_down_from: start stop counter from this number,
        dont_allow_stop: if True, stop action is only allowed if no other action is available,
        """
        
        self.add_reward_per_step = add_reward_per_step
        self.max_steps = max_steps        
        self.resetter = resetter
        self.count_down_from = count_down_from
        self.dont_allow_stop = dont_allow_stop
        self.extra_state_info = extra_state_info
        self.adapted_reward = adapted_reward

 

    def get_observation(self)-> tuple:
        """ observation: (type_array, phase_array, qubit_on_array,selected_node, edge_array, selected_edges)
        Get observation from the environment's graph"""

        type_list = []
        phase_list = []
        qubit_on_list = [] # qubit index of each node applied to
        for v in self.graph.vertices():
            qubit_on_list.append(self.graph.qubit(v))
            if v in self.graph.inputs():
                type_list.append(INPUT)
                phase_list.append(encode_phase(None))
            elif v in self.graph.outputs():
                type_list.append(OUTPUT)
                phase_list.append(encode_phase(None))
            elif self.graph.type(v) == VertexType.Z:
                type_list.append(GREEN)
                phase_list.append(encode_phase(self.graph.phase(v)))
            elif self.graph.type(v) == VertexType.X:
                type_list.append(RED)
                phase_list.append(encode_phase(self.graph.phase(v)))
            elif self.graph.type(v) == VertexType.H_BOX:
                type_list.append(HADAMARD)
                phase_list.append(encode_phase(None))

        self.node_array = np.array(list(self.graph.vertices()), dtype=np.int32)
        self.type_array = np.array(type_list, dtype=np.int32)
        self.phase_array = np.array(phase_list, dtype=np.int32)
        self.qubit_on_array = np.array(qubit_on_list, dtype=np.float32)
        self.edge_array = np.array(list(self.graph.edges()), dtype=np.int32)
        self.id2pos = {int(v_id): i for i, v_id in enumerate(self.node_array)}

        minigame = np.sum(self.selected_node)
        # Count down at end of trajectory only
        if self.max_steps - self.step_counter < self.count_down_from:
            count_down = self.max_steps - self.step_counter
        else:
            count_down = self.count_down_from

        if self.extra_state_info:
            info_state = self.max_diff
        else:
            info_state = 0.
        
        context_features = np.array([count_down, info_state, minigame], dtype=np.float32)
        # context_features = np.append(context_features, rel_action_counts)

        self.mask = self.get_action_mask()
        observation = [self.node_array,
                       self.type_array, 
                       self.phase_array,
                       self.qubit_on_array,
                       self.selected_node,
                       self.edge_array,
                       self.selected_edges,
                       self.n_spiders, 
                       self.n_edges,
                       self.mask,
                       context_features]

        return observation

    def reset(self)-> tuple:
        '''
        returns: (observation), 
        where observation is (type_array, phase_array, qubit_on_array,selected_node, edge_array, selected_edges)
        Resets the environment:
        1. Sample ZX_diagram.
        2. Set the step counter to zero.
        Return new initial observation
        '''
        self.graph, self.initial_circuit = self.resetter.reset()

        self.selected_edges = np.zeros(self.n_edges, dtype=np.int32)
        self.selected_node = np.zeros(self.n_spiders, dtype=np.int32)
        self.step_counter = 0
        self.max_diff = 0

        # For keep track of previous spiders for reward function
        self.previous_left_nodes_by_continuous, self.previous_left_nodes_by_continuous_and_broken = self.num_nodes_left
        self.current_left_nodes_by_continuous, self.current_left_nodes_by_continuous_and_broken = self.num_nodes_left
        self.previous_high_degree_nodes = self.count_high_degree_nodes()
        self.current_high_degree_nodes = self.count_high_degree_nodes()
        return self.get_observation()
    


    def step(self, action: int) -> tuple[int, int]:
        '''action: int, action to be applied to the environment,
        returns: (observation, reward, done)
        '''

        ## Add the step counter for stopping criteria
        self.step_counter += 1
        self.graph = self.graph.copy()
        # Check if trajectory is over
        if (self.step_counter >= self.max_steps or
            (action == N_EDGE_ACTIONS * self.n_edges + N_NODE_ACTIONS * self.n_spiders and not self.dont_allow_stop)
            ): 
            # Return observation and reward, end_episode
            observation = self.reset()
            return observation, 0, 1
        elif action == N_EDGE_ACTIONS * self.n_edges + N_NODE_ACTIONS * self.n_spiders and self.dont_allow_stop:
            observation = self.get_observation()
            return observation, 0, 0
        else:
            # Applies the action    
            success = self.apply_action(action)
            if len(self.selected_edges) != self.n_edges:
                self.selected_edges = np.zeros(self.n_edges, dtype=np.int32)
            if len(self.selected_node) != self.n_spiders:
                self.selected_node = np.zeros(self.n_spiders, dtype=np.int32)
            observation = self.get_observation()

    
            ## Calculate the reward
            self.previous_left_nodes_by_continuous, self.previous_left_nodes_by_continuous_and_broken = self.current_left_nodes_by_continuous, self.current_left_nodes_by_continuous_and_broken
            self.current_left_nodes_by_continuous, self.current_left_nodes_by_continuous_and_broken = self.num_nodes_left
            delta_left_continuous, delta_left_cont_and_broken = self.delta_left_nodes()
            self.previous_high_degree_nodes = self.current_high_degree_nodes
            self.current_high_degree_nodes = self.count_high_degree_nodes()

            # reward = delta_left_continuous + 0.2*delta_left_cont_and_broken + self.delta_high_degree_nodes()
            reward = -self.current_left_nodes_by_continuous - 0.2*self.current_left_nodes_by_continuous_and_broken - self.current_high_degree_nodes

            if reward + self.max_diff >= 0:
                reward_tilde = reward + self.max_diff
                self.max_diff = 0
            else:
                reward_tilde = 0
                self.max_diff = reward + self.max_diff

            # return observation, reward, done
            if self.adapted_reward:
                # rew_returned = reward_tilde + self.add_reward_per_step + success*0.1
                rew_returned = reward_tilde + self.add_reward_per_step
            else:
                # rew_returned = reward + self.add_reward_per_step + success*0.1
                rew_returned = reward + self.add_reward_per_step

            return observation, rew_returned, 0
    

    # =================Reward calculation=========================
    def delta_left_nodes(self)->int:
        """returns reward"""
        return self.previous_left_nodes_by_continuous - self.current_left_nodes_by_continuous, self.previous_left_nodes_by_continuous_and_broken - self.current_left_nodes_by_continuous_and_broken

    def delta_high_degree_nodes(self)->int:
        """returns reward"""
        return self.previous_high_degree_nodes - self.current_high_degree_nodes

    @property
    def n_spiders(self)->int:
        """Number of nodes in diagram"""
        return len(self.graph.vertices())
    
    @property
    def n_edges(self):
        """Number of edges in diagram"""
        return len(list(self.graph.edges()))

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
        self.node_actions_fn = [self.select_node, self.unfuse_rule, self.color_change_rule, self.split_hadamard, self.pi_rule]
        self.edge_actions_fn = [self.select_edge, self.fuse_rule, self.bialgebra_rule]
        if action < N_NODE_ACTIONS * self.n_spiders:
            # Action is node action
            node_idx = action//N_NODE_ACTIONS
            action_idx = action % N_NODE_ACTIONS
            success = self.node_actions_fn[action_idx](node_idx)
        elif action < N_NODE_ACTIONS *  self.n_spiders + N_EDGE_ACTIONS *self.n_edges:
            # Action is edge action
            action = action - (N_NODE_ACTIONS * self.n_spiders)
            edge_idx = action // N_EDGE_ACTIONS
            action_idx = action % N_EDGE_ACTIONS
            success = self.edge_actions_fn[action_idx](edge_idx)
        else:
            raise ValueError(f"Action {action} not recognized")
        self.graph = self.graph.copy()
        return success

    # ----------------node action-----------------------
    def select_node(self, node_idx):
        """select one node"""
        self.selected_node[node_idx] = 1-self.selected_node[node_idx]
        return True


    def unfuse_rule(self, node_idx):
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
        for neighbor in neighbours:
            pos = self.id2pos.get(int(neighbor), None)
            if pos is not None and self.selected_node[pos] == 1:
                self.graph.remove_edge((v, neighbor))
                self.graph.add_edge((child_idx, neighbor))
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
        if self.graph.type(v) == VertexType.Z:
            return pi_commute_Z(self.graph, v)
        elif self.graph.type(v) == VertexType.X:
            return pi_commute_X(self.graph, v)
        else:
            return False

    # ----------------edge action-----------------------
    def select_edge(self, edge_idx):
        """mark edge action after start Unfuse"""
        self.selected_edges[edge_idx] = 1-self.selected_edges[edge_idx]
        return True

    def fuse_rule(self, edge_idx):
        """Fuse action"""
        s,t = list(self.graph.edges())[edge_idx]
        if fuse(self.graph, s, t):
            return True
        else:
            return False

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
            if len(v0n) >2:
                self.selected_node = np.zeros(n_spiders, dtype=np.int32)
                for n in v0n[1:]:
                    pos = self.id2pos.get(int(n), None)
                    self.selected_node[pos] = 1
                self.unfuse_rule(v0)
            if len(v1n) >2:
                self.selected_node = np.zeros(n_spiders, dtype=np.int32)
                for n in v1n[1:]:
                    pos = self.id2pos.get(int(n), None)
                    self.selected_node[pos] = 1
                self.unfuse_rule(v1)

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
        cut1 = N_NODE_ACTIONS*self.n_spiders  
        cut2 = cut1 + N_EDGE_ACTIONS*self.n_edges
        mask = np.zeros(cut2+1, dtype=np.int32)
        for i in range(cut1):
            node_idx = i // N_NODE_ACTIONS
            action_idx = i % N_NODE_ACTIONS
            v = list(self.graph.vertices())[node_idx]
            if action_idx == 0: # select_node
                mask[i] = 1
            elif action_idx == 1: # unfuse_rule
                if self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X:
                    mask[i] = 1
            elif action_idx == 2: # color_change_rule
                if self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X:
                    mask[i] = 1
            elif action_idx == 3: # split_hadamard
                if self.graph.type(v) == VertexType.H_BOX:
                    mask[i] = 1
            elif action_idx == 4: # pi_rule
                if self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X:
                    mask[i] = 1
        for i in range(cut1, cut2):
            edge_idx = (i - cut1) // N_EDGE_ACTIONS
            action_idx = (i - cut1) % N_EDGE_ACTIONS
            s,t = list(self.graph.edges())[edge_idx]
            if action_idx == 0: # select_edge
                mask[i] = 1
            elif action_idx == 1: # fuse_rule
                if self.graph.type(s) == self.graph.type(t) and (self.graph.type(s) == VertexType.Z or self.graph.type(s) == VertexType.X):
                    mask[i] = 1
            elif action_idx == 2: # bialgebra_rule
                v0t = self.graph.type(s)
                v1t = self.graph.type(t)
                v0p = self.graph.phase(s)
                v1p = self.graph.phase(t)
                if (v0p == 0 and v1p == 0 and
                    ((v0t == VertexType.Z and v1t == VertexType.X) or (v0t == VertexType.X and v1t == VertexType.Z))):
                    v0n = [n for n in self.graph.neighbors(s) if not n == t]
                    v1n = [n for n in self.graph.neighbors(t) if not n == s]
                    if (len(v0n) >= 2 or len(v1n) >= 2 and
                        self.graph.num_edges(s, t) == 1 and # there is exactly one edge between s and t
                        self.graph.num_edges(s, s) == 0 and # there are no self-loops on s
                        self.graph.num_edges(t, t) == 0): # there are no self-loops on t
                        mask[i] = 1
        # STOP action
        if self.dont_allow_stop:
            if np.sum(mask[:-1]) == 0:
                mask[-1] = 1
        else:
            mask[-1] = 1
        return mask


def save(colors:np.ndarray, angles:np.ndarray, selected_node:np.ndarray, 
         source:np.ndarray, target:np.ndarray, selected_edges:np.ndarray, idx:int):
    """saves the current state of the environment at step idx"""
    with open(f"colors{idx}.pkl", 'wb') as f:
        pickle.dump(colors, f)
    with open(f"angles{idx}.pkl", 'wb') as f:
        pickle.dump(angles, f)
    with open(f"source{idx}.pkl", 'wb') as f:
        pickle.dump(source, f)
    with open(f"target{idx}.pkl", 'wb') as f:
        pickle.dump(target, f)
    with open(f"selected_node{idx}.pkl", 'wb') as f:
        pickle.dump(selected_node, f)
    with open(f"selected_edges{idx}.pkl", 'wb') as f:
        pickle.dump(selected_edges, f)


# The following functions are stand-alone functions to potentially make them jit compatible in the future
#Actions--------------------------------------------------------------------------------------------


from collections import Counter
from typing import List
import numpy as np
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

