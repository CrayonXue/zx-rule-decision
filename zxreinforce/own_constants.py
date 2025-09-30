import sys
sys.path.append('../pyzx_copy')
import pyzx_copy as zx_copy

from fractions import Fraction
from typing import Union

# One hot encoding of colors
INPUT = [1,0,0,0,0]
OUTPUT = [0,1,0,0,0]   
GREEN = [0,0,1,0,0]   
RED = [0,0,0,1,0]   
HADAMARD = [0,0,0,0,1]    

# Number of node/edge actions
N_NODE_ACTIONS = 1*32+3
N_EDGE_ACTIONS = 2


# one-hot encoding for the angles: 0, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
# Assign each angle a unique index in the one-hot vector
ANGLE_ENCODING = {
    0:           [1,0,0,0,0,0,0,0,0,0],
    0.25:        [0,1,0,0,0,0,0,0,0,0],  # pi/4
    0.5:         [0,0,1,0,0,0,0,0,0,0],  # pi/2
    0.75:        [0,0,0,1,0,0,0,0,0,0],  # 3pi/4
    1:           [0,0,0,0,1,0,0,0,0,0],  # pi
    1.25:        [0,0,0,0,0,1,0,0,0,0],  # 5pi/4
    1.5:         [0,0,0,0,0,0,1,0,0,0],  # 3pi/2
    1.75:        [0,0,0,0,0,0,0,1,0,0],  # 7pi/4
    'Poly':      [0,0,0,0,0,0,0,0,1,0],  # polynomial with symbolic variables
    None:        [0,0,0,0,0,0,0,0,0,1]   # No angle
}

# Update ANGLE_ENCODING to map all pyzx_copy.symbolic.Poly phases to "Poly"
def encode_phase(phase):
    if isinstance(phase, zx_copy.symbolic.Poly):
        return ANGLE_ENCODING['Poly']
    elif phase==None:
        return ANGLE_ENCODING[None]
    elif isinstance(phase, Union[Fraction,int,float]):
        return ANGLE_ENCODING[phase%2]
    else:
        return ANGLE_ENCODING[None]