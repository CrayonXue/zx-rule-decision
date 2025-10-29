
# This file contains the ZXCalculus environment class and functions to apply actions to the diagram

import pickle
import numpy as np
from collections import Counter
from typing import List
import numpy as np
import torch 
from torch_geometric.data import Data



import sys
sys.path.append('../pyzx_copy')
from pyzx_copy.utils import VertexType, toggle_vertex, heuristic_fixed_pairs_node_disjoint, broken_paths_for_unrouted_pairs, to_networkx_graph
from pyzx_copy.basicrules import pi_commute_Z,pi_commute_X,fuse
from .own_constants import (INPUT, OUTPUT, GREEN, RED, HADAMARD, N_NODE_ACTIONS, encode_phase)

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
        self.CNOT_reward = 0.001  # reward per reduced CNOT count
        self.T_reward = 2  # reward per reduced T count

    def _rebuild_order_cache(self):
        self.graph.my_normalize()  # rearrange nodes position(qubit, row) in the graph
        # vertices
        verts = sorted(map(int, self.graph.vertices()),
                        key=lambda v: (self.graph.qubit(v), self.graph.row(v), int(v)))
        self._verts = verts
        self._id2pos = {int(v): i for i, v in enumerate(verts)}

        # neighbors (sorted once per vertex)
        self._nbrs = {}
        for v in verts:
            nbrs = list(map(int, self.graph.neighbors(v)))
            nbrs.sort(key=lambda n: (self.graph.qubit(n), self.graph.row(n), int(n)))
            self._nbrs[v] = nbrs

        self._nbrs_same_color = {}
        for v in verts:
            nbrs = self._nbrs[v]
            same_color_nbrs = [n for n in nbrs if self.graph.type(n) == self.graph.type(v)]
            self._nbrs_same_color[v] = same_color_nbrs
        
        self._nbrs_diff_color = {}
        for v in verts:
            nbrs = self._nbrs[v]
            type_v = self.graph.type(v)
            if type_v in (VertexType.Z, VertexType.X):
                diff_color_nbrs = [n for n in nbrs if (self.graph.type(n) != type_v and self.graph.type(n) in (VertexType.Z, VertexType.X))]
            else:
                diff_color_nbrs = []
            self._nbrs_diff_color[v] = diff_color_nbrs

        # edges
        E = []
        for (u, v) in self.graph.edges():
            u, v = int(u), int(v)
            if u == v: continue
            a, b = (u, v) if self._id2pos[u] < self._id2pos[v] else (v, u)
            E.append((a, b))
        self._edges = sorted(set(E), key=lambda p: (self._id2pos[p[0]], self._id2pos[p[1]]))

    def get_observation(self)-> tuple:
        """ observation
        Get data,mask from the environment's graph"""
        nodes_features = []
        verts = self._verts
        # Precompute normalizers
        if len(verts) == 0:
            max_row = 1
            n_qubits = 1
        else:
            max_row = max(self.graph.row(u) for u in verts) or 1
            qubits = [self.graph.qubit(u) for u in verts]
            n_qubits = (max(qubits) if qubits else 0) + 1

        for v in verts:
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
            
            degree = len(self._nbrs[v])
            is_boundary = int(v in self.graph.inputs() or v in self.graph.outputs())
            row_norm = self.graph.row(v) / max_row
            qubit_norm = self.graph.qubit(v) / max(1, n_qubits - 1)

            node_feasture = node_type + node_phase + [degree, is_boundary,row_norm,qubit_norm] #5+10+4
            nodes_features.append(node_feasture)
        x = torch.tensor(nodes_features, dtype=torch.float32)


        id2pos = self._id2pos
        edge_list = self._edges

        if not edge_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)
        else:
            uv=[]
            ea=[]
            for (u,v) in edge_list:
                u, v = int(u), int(v)
                ui, vi = id2pos[u], id2pos[v]

                def feat(a, b):
                    dq = abs(self.graph.qubit(a) - self.graph.qubit(b)) / max(1, n_qubits - 1)
                    dr = abs(self.graph.row(a) - self.graph.row(b)) / max_row
                    h_incident = int(self.graph.type(a) == VertexType.H_BOX or self.graph.type(b) == VertexType.H_BOX)
                    same_color = int(
                        (self.graph.type(a) in (VertexType.Z, VertexType.X)) and
                        (self.graph.type(a) == self.graph.type(b))
                    )
                    return [dq, dr, h_incident, same_color]

                # u->v
                uv.append((ui, vi))
                ea.append(feat(u, v))
                # v->u
                uv.append((vi, ui))
                ea.append(feat(v, u))                

            edge_index = torch.tensor(uv, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(ea, dtype=torch.float32)                  # [E, 4]
        # Precompute feasibility mask (variable length: N*node_act + E*edge_act + 1)
        mask = torch.from_numpy(self.get_action_mask().astype(np.int8))  # [A_var]
        data = Data(
            x=x, 
            edge_index=edge_index,
            edge_attr=edge_attr)
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
        
        self.graph, self.circuit = self.resetter.reset()
        initial_circuit = self.circuit.to_basic_gates()
        self.initial_cnot = initial_circuit.stats_dict()['cnot']
        self.initial_t = initial_circuit.stats_dict()['tcount']
        self._rebuild_order_cache()
        self.step_counter = 0

        self.current_left_nodes_by_continuous = self.num_nodes_left
        self.previous_left_nodes_by_continuous = self.current_left_nodes_by_continuous
        self.current_high_degree_nodes = self.count_high_degree_nodes(degree_threshold=3)
        self.previous_high_degree_nodes = self.current_high_degree_nodes
        self.current_cnot_proxy = self.count_high_degree_nodes(degree_threshold=2) - self.count_high_degree_nodes(degree_threshold=3)
        self.previous_cnot_proxy = self.current_cnot_proxy
        self.current_t_count = self.count_T_nodes()
        self.previous_t_count = self.current_t_count

        return self.get_observation()
    


    def step(self, action: int) -> tuple[int, int]:
        '''action: int, action to be applied to the environment,
        returns: (data,mask, reward, done)
        '''

        ## Add the step counter for stopping criteria
        self.step_counter += 1
        # self.graph = self.graph.copy()
        # Check if trajectory is over
        if self.step_counter >= self.max_steps: 
            # Return observation and reward, end_episode
            data,mask = self.reset()
            t_reduced = self.initial_t - self.current_t_count
            cnot_reduced = self.initial_cnot - self.current_cnot_proxy
            info = {"terminated": False, "TimeLimit.truncated": True, "T_reduced": t_reduced, "CNOT_reduced": cnot_reduced}
            # print("info", info)
            return data,mask, 0, 1, info
        
        else:
            # Applies the action    
            self.apply_action(action)
            self._rebuild_order_cache()
            data, mask = self.get_observation()

            ## Calculate the reward
            # self.current_left_nodes_by_continuous, self.current_left_nodes_by_continuous_and_broken = self.num_nodes_left
            self.current_left_nodes_by_continuous = self.num_nodes_left
            self.current_high_degree_nodes = self.count_high_degree_nodes(degree_threshold=3)
            self.current_t_count = self.count_T_nodes()
            self.current_cnot_proxy = self.count_high_degree_nodes(degree_threshold=2) - self.count_high_degree_nodes(degree_threshold=3)

            done = 1 if self.is_terminal() else 0

            
            # reward = self.delta_left_continuous() + self.delta_left_cont_and_broken() + self.delta_high_degree_nodes()
            reward = self.delta_left_continuous() + self.delta_high_degree_nodes()
            reward += self.T_reward * self.delta_t_nodes() # reward for T reduction compared to initial T count
            reward += self.CNOT_reward * self.delta_cnot_proxy()
            # reward = -self.current_left_nodes_by_continuous - 0.2*self.current_left_nodes_by_continuous_and_broken - self.current_high_degree_nodes
            reward = reward - self.step_penalty # Apply penalties

            t_reduced = self.initial_t - self.current_t_count
            cnot_reduced = self.initial_cnot - self.current_cnot_proxy
            if done:
                reward+=3.0  # bonus for finishing
                reward += t_reduced if t_reduced>0 else 0  # bonus for T reduction
                reward += cnot_reduced if cnot_reduced>0 else 0  # bonus for CNOT reduction
            info = {"terminated": done, "TimeLimit.truncated": False, "T_reduced": t_reduced, "CNOT_reduced": cnot_reduced}
            # print("info", info)
            return data, mask, reward, done, info


    # =================Reward calculation=========================
    def delta_left_continuous(self):
        delta = self.previous_left_nodes_by_continuous - self.current_left_nodes_by_continuous
        self.previous_left_nodes_by_continuous = self.current_left_nodes_by_continuous
        # if delta > 0:
        #     return delta
        # else:
        #     return 0
        return delta

    def delta_left_cont_and_broken(self):
        delta = self.previous_left_nodes_by_continuous_and_broken - self.current_left_nodes_by_continuous_and_broken
        self.previous_left_nodes_by_continuous_and_broken = self.current_left_nodes_by_continuous_and_broken
        # if delta > 0:
        #     return delta
        # else:
        #     return 0
        return delta

    def delta_high_degree_nodes(self):
        delta = self.previous_high_degree_nodes - self.current_high_degree_nodes
        self.previous_high_degree_nodes = self.current_high_degree_nodes
        # if delta > 0:
        #     return delta
        # else:
        #     return 0
        return delta
    
    def delta_t_nodes(self):
        delta = self.previous_t_count - self.current_t_count
        self.previous_t_count = self.current_t_count
        return delta
    
    def delta_cnot_proxy(self):
        delta = self.previous_cnot_proxy - self.current_cnot_proxy
        self.previous_cnot_proxy = self.current_cnot_proxy
        return delta

    @property
    def n_spiders(self)->int:
        """Number of nodes in diagram"""
        return len(self._verts)
    
    @property
    def n_edges(self):
        """Number of edges in diagram"""
        return len(list(self.graph.edges()))

    @property
    def id2pos(self):
        return {int(v_id): i for i, v_id in enumerate(self._verts)}

    def count_high_degree_nodes(self, degree_threshold: int = 3) -> int:
        return sum(1 for v in self._verts if len(self._nbrs[v]) > degree_threshold)
    
    def count_T_nodes(self) -> int:
        return sum(1 for v in self._verts
               if self.graph.type(v) in (VertexType.Z, VertexType.X) and is_t_like(self.graph.phase(v)))
        

    @property
    def nodes_degrees(self):
        return [len(self._nbrs[v]) for v in self._verts]

    @property
    def num_nodes_left(self):

        G_nx = to_networkx_graph(self.graph)
        pairs = list(zip(self.graph.inputs(), self.graph.outputs()))  # fixed (si, ti)

        # path connecting inputs to outputs
        continuous_paths_dict = heuristic_fixed_pairs_node_disjoint(G_nx, pairs, iterations=10, restarts=5, seed=42)
        used_nodes = set().union(*continuous_paths_dict.values())
        left_nodes_by_continuous = set(self.graph.vertices()) - used_nodes # nodes not in any continuous path
        return len(left_nodes_by_continuous)

    @property
    def num_nodes_left_cb(self):

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



    def ordered_vertices(self):
        # Use a persistent, deterministic order for vertices
        # Prefer integer ID if stable; otherwise sort by (qubit, row, id)
        return sorted(self.graph.vertices(), key=lambda v: (self.graph.qubit(v), self.graph.row(v), int(v)))

    def ordered_neighbors(self, v):
        nbrs = list(self.graph.neighbors(v))
        nbrs.sort(key=lambda n: (self.graph.qubit(n), self.graph.row(n), int(n)))
        return nbrs

    # ==================Actions=========================
    def apply_action(self, action:int):
        N = len(self._verts)
        UNFUSE_SPACE = (1 << 5)  # 0-31; 32
        COLOR_OFFSET = UNFUSE_SPACE # 32
        SPLIT_OFFSET = COLOR_OFFSET + 1 # 33
        PI_OFFSET    = SPLIT_OFFSET + 1 # 34
        ID_OFFSET    = PI_OFFSET + 1 # 35
        ID_H_OFFSET  = ID_OFFSET + 1 # 36
        FUSE_SPACE  = ID_H_OFFSET + 1 + 5 # 37 - 41 ; 42
        BIAG_SPACE   = FUSE_SPACE + 5 # 42 - 46 ; 47
        assert N_NODE_ACTIONS == BIAG_SPACE, "Update offsets if you add actions."

        if action < N * N_NODE_ACTIONS:
            node_pos = action // N_NODE_ACTIONS
            a = action % N_NODE_ACTIONS
            if a < UNFUSE_SPACE:
                return self.unfuse_rule(node_pos, a)
            elif a == COLOR_OFFSET:
                return self.color_change_rule(node_pos)
            elif a == SPLIT_OFFSET:
                return self.split_hadamard(node_pos)
            elif a == PI_OFFSET:
                return self.pi_rule(node_pos)
            elif a == ID_OFFSET:
                return self.remove_identity_rule(node_pos)
            elif a == ID_H_OFFSET:
                return self.remove_identity_Hadamard_rule(node_pos)
            elif a < FUSE_SPACE:
                return self.fuse_rule(node_pos, a - (ID_H_OFFSET + 1))
            elif a < BIAG_SPACE:
                return self.bialgebra_rule(node_pos, a - (FUSE_SPACE))
            else:
                return False
        else:
            return False


    # ----------------node action-----------------------
    def fuse_rule(self, node_pos, action_idx):
        """Fuse action"""
        v = self._verts[node_pos]
        if self.graph.type(v) not in (VertexType.Z, VertexType.X):
            return False
        same_color_nbrs = self._nbrs_same_color[v]
        if len(same_color_nbrs) < 1:
            return False
    
        u = same_color_nbrs[action_idx % len(same_color_nbrs)]
        return fuse(self.graph, v, u)

    def unfuse_rule(self, node_pos, action_idx):
        """Unfuse action"""
        v = self._verts[node_pos]
        if self.graph.type(v) not in (VertexType.Z, VertexType.X):
            return False
        nbrs = self._nbrs[v]
        k = min(len(nbrs), 5)
        if action_idx >= (1 << k):
            return False
        
        child = self.graph.add_vertex(ty=self.graph.type(v), phase=0, qubit=self.graph.qubit(v))

        # Move first-k neighbors according to bits
        for bit, n in enumerate(nbrs[:k]):
            if get_bit(action_idx, bit):
                self.graph.remove_edge((v, n))
                self.graph.add_edge((child, n))
        self.graph.add_edge((v, child))


        # Update child attrs deterministically
        child_nbrs = list(self.graph.neighbors(child))  # unsorted OK for qubit/row aggregation
        neighbor_qubit = [self.graph.qubit(u) for u in child_nbrs]
        neighbor_row   = [self.graph.row(u) for u in child_nbrs]
        self.graph.set_qubit(child, determine_qubit(neighbor_qubit))
        self.graph.set_row(child,   determine_row(neighbor_row))

        return True

    def remove_identity_rule(self, node_pos):
        """Remove identity action"""
        v = self._verts[node_pos]
        if self.graph.type(v) not in (VertexType.Z, VertexType.X):
            return False
        if self.graph.phase(v) != 0:
            return False
        nbrs = self._nbrs[v]
        if len(nbrs) != 2:
            return False
        u1, u2 = nbrs
        self.graph.remove_vertex(v)
        self.graph.add_edge((u1, u2))
        return True
    
    def remove_identity_Hadamard_rule(self, node_pos):
        """Remove identity Hadamard action"""
        v = self._verts[node_pos]
        if self.graph.type(v) != VertexType.H_BOX:
            return False
        nbrs = self._nbrs[v]
        u1, u2 = nbrs
        if self.graph.type(u1) == VertexType.H_BOX:
            u1_nbrs = self._nbrs[u1]
            u3 = [n for n in u1_nbrs if n != v][0]
            self.graph.remove_vertex(v)
            self.graph.remove_vertex(u1)
            self.graph.add_edge((u2, u3))
            return True
        elif self.graph.type(u2) == VertexType.H_BOX:
            u2_nbrs = self._nbrs[u2]
            u3 = [n for n in u2_nbrs if n != v][0]
            self.graph.remove_vertex(v)
            self.graph.remove_vertex(u2)
            self.graph.add_edge((u1, u3))
            return True
        else:
            return False

    def color_change_rule(self, node_pos):
        v = self._verts[node_pos]
        if not (self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X):
            return False
        self.graph.set_type(v, toggle_vertex(self.graph.type(v)))
        neighbours = self._nbrs[v]
        for n in neighbours:
            self.graph.remove_edge((v,n))
            h = self.graph.add_vertex(ty=VertexType.H_BOX,qubit=determine_qubit([self.graph.qubit(n),self.graph.qubit(v)]),row=determine_row([self.graph.row(n),self.graph.row(v)]))
            self.graph.add_edge((v,h))
            self.graph.add_edge((h,n))
        return True

    def split_hadamard(self, node_idx):
        """Hadamard unfuse action"""
        # Change middle node
        v = self._verts[node_idx]
        if self.graph.type(v) != VertexType.H_BOX:
            return False
        neighbours = self._nbrs[v]
        if len(neighbours) != 2:
            raise ValueError(f"Hadamard node v={v} requires exactly two neighbors.")

        for w in self._verts:
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
        v = self._verts[node_idx]
        if (self.graph.type(v) == VertexType.Z or self.graph.type(v) == VertexType.X):
            if self.graph.type(v) == VertexType.Z:
                pi_commute_Z(self.graph, v)
            elif self.graph.type(v) == VertexType.X:
                pi_commute_X(self.graph, v)
            return True
        else:
            return False


    def bialgebra_rule(self, node_pos, action_idx):
        v = self._verts[node_pos]
        if self.graph.type(v) not in (VertexType.Z, VertexType.X):
            return False
        diff_color_nbrs = self._nbrs_diff_color[v]
        u = diff_color_nbrs[action_idx % len(diff_color_nbrs)]

        v0,v1 = v,u

        v0t = self.graph.type(v0)
        v1t = self.graph.type(v1)
        v0p = self.graph.phase(v0)
        v1p = self.graph.phase(v1)
        if ((v0t == VertexType.Z and v1t == VertexType.X) or (v0t == VertexType.X and v1t == VertexType.Z)):
            v0n = [n for n in self._nbrs[v0] if not n == v1]
            v1n = [n for n in self._nbrs[v1] if not n == v0]

            if v0p != 0 or len(v0n) !=2:
                if len(v0n) == 0:
                    child1 = self.graph.add_vertex(ty=v0t, phase=v0p)
                    child2 = self.graph.add_vertex(ty=v0t, phase=0)
                    self.graph.set_phase(v0, 0)
                    self.graph.add_edge((v0, child1))
                    self.graph.add_edge((v0, child2))
                    v0n = [child1, child2]
                elif len(v0n) ==1:
                    child = self.graph.add_vertex(ty=v0t, phase=v0p)
                    self.graph.set_phase(v0, 0)
                    self.graph.add_edge((v0, child))
                    v0n.append(child)  
                else:
                    child = self.graph.add_vertex(ty=v0t, phase=v0p)
                    self.graph.set_phase(v0, 0)
                    self.graph.add_edge((v0, child))
                    for n in v0n[1:]:
                        self.graph.remove_edge((v0,n))
                        self.graph.add_edge((child,n))
                    v0n = v0n[:1] + [child]
            elif v1p != 0 or len(v1n) !=2:
                if len(v1n) == 0:
                    child1 = self.graph.add_vertex(ty=v1t, phase=v1p)
                    child2 = self.graph.add_vertex(ty=v1t, phase=0)
                    self.graph.set_phase(v1, 0)
                    self.graph.add_edge((v1, child1))
                    self.graph.add_edge((v1, child2))
                    v1n = [child1, child2]
                elif len(v1n) ==1:
                    child = self.graph.add_vertex(ty=v1t, phase=v1p)
                    self.graph.set_phase(v1, 0)
                    self.graph.add_edge((v1, child))
                    v1n.append(child)  
                else:
                    child = self.graph.add_vertex(ty=v1t, phase=v1p)
                    self.graph.set_phase(v1, 0)
                    self.graph.add_edge((v1, child))
                    for n in v1n[1:]:
                        self.graph.remove_edge((v1,n))
                        self.graph.add_edge((child,n))
                    v1n = v1n[:1] + [child]

            
            elif (len(v0n) == 2 and len(v1n) == 2 and
                v0p == 0 and v1p == 0 and
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
        N = len(self._verts)
        total = N * N_NODE_ACTIONS 
        mask = np.zeros(total, dtype=np.int32)

        UNFUSE_SPACE = (1 << 5)  # 0-31; 32
        COLOR_OFFSET = UNFUSE_SPACE # 32
        SPLIT_OFFSET = COLOR_OFFSET + 1 # 33
        PI_OFFSET    = SPLIT_OFFSET + 1 # 34
        ID_OFFSET    = PI_OFFSET + 1 # 35
        ID_H_OFFSET  = ID_OFFSET + 1 # 36
        FUSE_OFFSET  = ID_H_OFFSET + 1 # 37 - 41 
        BIAG_OFFSET   = FUSE_OFFSET + 5 # 42 - 46 

        for i, v in enumerate(self._verts):
            base = i * N_NODE_ACTIONS
            deg = len(self._nbrs[v])
            deg_same_color = len(self._nbrs_same_color[v])
            deg_diff_color = len(self._nbrs_diff_color[v])
            if self.graph.type(v) in (VertexType.Z, VertexType.X):
                k = min(deg, 5)
                mask[base : base + (1 << k)] = 1  # contiguous block for unfuse patterns
                mask[base + COLOR_OFFSET] = 1
                mask[base + PI_OFFSET] = 1
                if deg==2 and self.graph.phase(v) == 0:
                    mask[base + ID_OFFSET] = 1
                # fuse
                m = min(deg_same_color, 5)
                mask[base + FUSE_OFFSET : base + FUSE_OFFSET + m] = 1  # contiguous block for fuse patterns

                # bialgebra
                m2 = min(deg_diff_color, 5)
                mask[base + BIAG_OFFSET : base + BIAG_OFFSET + m2] = 1  # contiguous block
 
            elif self.graph.type(v) == VertexType.H_BOX:
                mask[base + SPLIT_OFFSET] = 1
                u1,u2 = self._nbrs[v]
                if self.graph.type(u1) == VertexType.H_BOX or self.graph.type(u2) == VertexType.H_BOX:
                    mask[base + ID_H_OFFSET] = 1
                 

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

def is_t_like(phase):
    # Treat phase in multiples of 0.5 with tolerance
    try:
        p = float(phase)
    except Exception:
        p = phase  # if fraction-like, handle accordingly
    return not any(np.isclose(p, k * 0.5, atol=1e-6) for k in (0,1,2,3))


def determine_qubit(neighbor_qubit:List[int]) -> int:
    """Determine the qubit for the new child node based on its neighbors' qubits.
    
    If all neighbors are on the same qubit, return that qubit.
    If neighbors are on different qubits, return None (indicating ambiguity).
    """
    c = Counter(neighbor_qubit)
    maxf = max(c.values())
    return int(min(q for q, f in c.items() if f == maxf))


def determine_row(neighbor_row:List[int]) -> int:
    """Determine the row for the new child node based on its neighbors' rows.
    
    The new node's row is set to one more than the maximum row among its neighbors.
    """
    return max(neighbor_row)


def get_bit(n: int, idx: int) -> int:
    """Return the bit at index idx (LSB=0) for integer n. Works for n >= 0."""
    if idx < 0:
        raise ValueError("idx must be non-negative")
    return (n >> idx) & 1

