"""A large sharded reference library (nablamimic.bones.convert output) and fixed-size working sets of it.

The full library (BONES-SEED: ~52M frames at 50 Hz) does not fit on the GPU next to training, so training keeps a working
set: a few shards (each a random sample of the library) concatenated into a MotionLibrary of exactly `capacity` frames.
The frames after the last clip repeat its final frame and are never valid (last[f] = f), so every working set gives env
tables of the same shapes (TrackingEnv.set_library) and swapping one needs no recompilation.
"""

import json
from pathlib import Path

import numpy as np

from .motion import MotionLibrary

KEYS = ("qpos", "qvel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")


FLAGS = ("name_filtered", "elevated", "penetration", "floating")  # SuperMimic's BONES-SEED filters (bones/convert.py)


class ShardedLibrary:
    def __init__(self, root, filtered=True):
        """filtered: leave out the clips the converter flagged (FLAGS), as SuperMimic's training set does."""
        self.root = Path(root)
        idx = json.loads((self.root / "index.json").read_text())
        self.fps, self.bodies = idx["fps"], tuple(idx["bodies"])
        self.clips = {k: c for k, c in idx["clips"].items() if not (filtered and any(c.get(f) for f in FLAGS))}
        self.shards = sorted({c["shard"] for c in self.clips.values()})
        self.shard_frames = {s: 0 for s in self.shards}
        for c in self.clips.values():
            self.shard_frames[c["shard"]] += c["length"]

    def capacity(self, k):
        """Frames that hold any k shards."""
        return int(sum(sorted(self.shard_frames.values())[-k:]))

    def working_set(self, shards, capacity):
        """The clips of `shards` as one MotionLibrary of exactly `capacity` frames (clip order as stored)."""
        parts, names, starts, off = {k: [] for k in KEYS}, [], [], 0
        for s in shards:
            z = np.load(self.root / f"{s}.npz")
            keep = [i for i, nm in enumerate(z["names"]) if str(nm) in self.clips]
            arr = {k: z[k] for k in KEYS}
            for i in keep:
                a, n = int(z["starts"][i]), int(z["lengths"][i])
                for k in KEYS:
                    parts[k].append(arr[k][a:a + n])
                names.append(str(z["names"][i]))
                starts.append(off)
                off += n
        assert off <= capacity, f"{off} frames do not fit {capacity}"
        starts = np.asarray(starts)
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        last = np.empty(capacity, np.int64)
        ends = np.append(starts[1:], off) - 1
        for s0, e in zip(starts, ends):
            last[s0:e + 1] = e
        last[off:] = np.arange(off, capacity)  # padding: never valid
        pad = capacity - off
        arrays = {k: np.concatenate([v, np.repeat(v[-1:], pad, 0)]) for k, v in arrays.items()}
        return MotionLibrary(qpos=arrays["qpos"], qvel=arrays["qvel"], body_pos=arrays["body_pos"], last=last,
                             names=names, starts=starts, fps=self.fps, bodies=self.bodies,
                             body_quat=arrays["body_quat"], body_lin_vel=arrays["body_lin_vel"],
                             body_ang_vel=arrays["body_ang_vel"])
