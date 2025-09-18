# smoke_test.py
from zxreinforce.Resetters import Resetter_Circuit
from zxreinforce.zx_env_circuit import ZXCalculus
from zxreinforce.zx_gym_wrapper import ZXGymWrapper

resetter = Resetter_Circuit(2, 3, 5, 10, p_t=0.2, p_h=0.2)
env = ZXCalculus(resetter=resetter, adapted_reward=True)
w = ZXGymWrapper(env, max_nodes=64, max_edges=128)

obs = w.reset()
total = 0.0
for _ in range(20):
    # pick a random valid action from mask
    import numpy as np
    mask = obs["action_mask"]
    valid_actions = np.flatnonzero(mask)
    a = int(np.random.choice(valid_actions))
    obs, r, d, _ = w.step(a)
    total += r
    if d:
        obs = w.reset()
print("OK, random 20-step return:", total)