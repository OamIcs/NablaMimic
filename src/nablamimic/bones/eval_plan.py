"""Checkpoints of a BONES-SEED run for bones/benchmark.py: prints the update numbers of one evaluation part.

  python -m nablamimic.bones.eval_plan RUN full [K/N]    # nearest checkpoints to 0.5, 1, 2, 3, 4, 6 h (and the last)
  python -m nablamimic.bones.eval_plan RUN subset [K/N]  # the subset curve: every 500 updates in the first hour of
                                                         # training, then every 2000, and the last
K/N: part K of N of the list (to split it over several jobs).
"""

import argparse
import json
from pathlib import Path

HOURS = (0.5, 1, 2, 3, 4, 6)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nablamimic.bones.eval_plan",
                                 description="Print the update numbers of the checkpoints of one part of the BONES-SEED "
                                             "benchmark of a run.")
    ap.add_argument("run", help="run directory (ckpt/, log.jsonl)")
    ap.add_argument("mode", choices=["full", "subset"],
                    help="full: the nearest checkpoints to 0.5, 1, 2, 3, 4, 6 h of training and the last; subset: every "
                         "500 updates in the first hour, then every 2000, and the last")
    ap.add_argument("part", nargs="?", default=None, metavar="K/N", help="part K of N of the list")
    args = ap.parse_args(argv)
    d, mode = Path(args.run), args.mode
    if not (d / "log.jsonl").exists():
        ap.error(f"{d} is not a run directory (no log.jsonl)")
    its = sorted(int(p.stem[3:]) for p in (d / "ckpt").glob("it_*.pkl"))
    wall = {}
    for line in (d / "log.jsonl").read_text().splitlines():
        if line.startswith('{"iter"'):
            x = json.loads(line)
            wall[x["iter"]] = x["wall_s"] / 3600
    h = {i: wall.get(i, wall.get(i - i % 10, 0.0)) for i in its if i > 0}
    if not h:
        return
    if mode == "full":
        sel = sorted({min(h, key=lambda i: abs(h[i] - t)) for t in HOURS if t <= max(h.values()) + 0.05} | {max(h)})
    else:
        sel = sorted({i for i in h if (h[i] <= 1.0 and i % 500 == 0) or i % 2000 == 0} | {max(h)})
    if args.part:  # part K of N
        k, n = map(int, args.part.split("/"))
        sel = sel[k - 1::n]
    print(*sorted(sel))


if __name__ == "__main__":
    main()
