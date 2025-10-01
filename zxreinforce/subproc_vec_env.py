# subproc_vec_env.py
import multiprocessing as mp
import numpy as np

# Optional: robust pickling of closures
class CloudpickleWrapper:
    def __init__(self, fn):
        self.fn = fn
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)
    def __setstate__(self, state):
        import pickle
        self.fn = pickle.loads(state)
    def __call__(self):
        return self.fn()

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs = env.reset()  # (Data, mask)
                remote.send(obs)
            elif cmd == "step":
                action = int(data)
                data_, mask_, r, d, info = env.step(action)
                # unify semantics: always return next obs, and if done happened, immediately reset
                if d:
                    # for your env: time-limit path already resets inside step; success path does not.
                    # do a reset anyway to be consistent
                    data2, mask2 = env.reset()
                    # send a single tuple including info
                    remote.send(((data2, mask2), float(r), True, info))
                else:
                    # send a single tuple including info
                    remote.send(((data_, mask_), float(r), False, info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except KeyboardInterrupt:
        pass

class SubprocVecEnv:
    def __init__(self, env_fns, start_method="spawn"):
        self.n = len(env_fns)
        ctx = mp.get_context(start_method)
        self.parent_conns, self.child_conns = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.ps = []
        for c, p, fn in zip(self.child_conns, self.parent_conns, env_fns):
            proc = ctx.Process(target=_worker, args=(c, p, CloudpickleWrapper(fn)))
            proc.daemon = True
            proc.start()
            c.close()  # child end closed in parent
            self.ps.append(proc)

    def reset(self):
        for pr in self.parent_conns:
            pr.send(("reset", None))
        results = [pr.recv() for pr in self.parent_conns]
        # list of tuples: (Data, mask)
        return results

    def step(self, actions):
        # actions: iterable length n
        for pr, a in zip(self.parent_conns, actions):
            pr.send(("step", int(a)))
        results = [pr.recv() for pr in self.parent_conns]
        # results: [((Data, mask), r, d, info), ...]
        next_obs = [{"data": o[0], "mask": o[1]} for (o, _, _, _) in results]
        rewards = np.array([r for (_, r, _, _) in results], dtype=np.float32)
        dones   = np.array([d for (_, _, d, _) in results], dtype=np.bool_)
        # be robust if keys missing in info
        def _mk_info(info):
            return {
                "terminated": bool(info.get("terminated", False)) if isinstance(info, dict) else False,
                "TimeLimit.truncated": bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False,
            }
        infos = [_mk_info(info) for (_, _, _, info) in results]
        return next_obs, rewards, dones, infos

    def close(self):
        for pr in self.parent_conns:
            pr.send(("close", None))
        for p in self.ps:
            p.join(timeout=1)