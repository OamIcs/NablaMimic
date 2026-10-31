"""The checkpoints of a run that the nominal LAFAN1 protocol evaluates: prints their update numbers (or nothing).

  python -m nablamimic.eval_plan RUN

train.py runs: every checkpoint of the first DENSE_H hours of training (wall_s in log.jsonl, to compare the early
phase), then every 1000 updates, and the last checkpoint. ppo.py runs (algo ppo; checkpoints by wall clock, dense in the
first hour): every checkpoint after 0. Checkpoints after a divergence (DIVERGED) are skipped; a run stopped at SHAC's
gradient guard (HALTED) ends with the checkpoint written there (the gradient step at the guard was not applied).
"""

import argparse
import json
from pathlib import Path

DENSE_H = 1.0  # every checkpoint up to this many hours of training


def plan(d):
    d = Path(d)
    its = sorted(int(p.stem[3:]) for p in (d / "ckpt").glob("it_*.pkl"))
    if (d / "DIVERGED").exists():
        nan_it = int((d / "DIVERGED").read_text().strip())
        its = [i for i in its if i < nan_it]
    if not its:
        return []
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("algo") == "ppo":
        return [i for i in its if i > 0]
    wall_at = {}
    log = d / "log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith('{"iter"'):
                x = json.loads(line)
                wall_at[x["iter"]] = x["wall_s"]
    early = {i for i in its if 0 < i and wall_at.get(i, float("inf")) <= DENSE_H * 3600}
    return sorted({i for i in its if i % 1000 == 0 and i > 0} | early | {its[-1]})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nablamimic.eval_plan",
                                 description="Print the update numbers of the checkpoints the nominal LAFAN1 protocol "
                                             "evaluates for a run (nothing if there are none).")
    ap.add_argument("run", help="run directory (config.json, ckpt/, log.jsonl)")
    run = ap.parse_args(argv).run
    if not (Path(run) / "config.json").exists():
        ap.error(f"{run} is not a run directory (no config.json)")
    sel = plan(run)
    if sel:
        print(*sel)


if __name__ == "__main__":
    main()
