"""Configuration files (configs/*.args), the argv locked into a run's config.json, and the source revision."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def read_params(path, seen=()):
    """Arguments from a configuration file: whitespace-separated as on a command line, '#' starts a comment, a line
    '@FILE' includes another file (relative to this one's directory); the last value of an option wins."""
    path = Path(path).resolve()
    assert path not in seen, f"configuration include cycle at {path}"
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("@"):
            out += read_params(path.parent / line[1:].strip(), seen + (path,))
        elif line:
            out += shlex.split(line)
    return out


def resolve_params(argv):
    """--config FILE [overrides]: at a run's first start the arguments are FILE's followed by the overrides (the last
    value wins), locked into OUT/config.json; later starts (--resume) use the locked arguments, and the command line may
    then carry only --config, --out and --resume (edits of FILE are ignored with a warning). Without --config, argv as
    given. Returns (argv, info) with info = None or {"argv", "config_args", "locked"}."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre.add_argument("--out")
    pre.add_argument("--resume")
    p0, rest = pre.parse_known_args(argv)
    if p0.config is None:
        return argv, None
    assert p0.out, "--out is required"
    cfgp = Path(p0.out) / "config.json"
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    file_args = read_params(p0.config)
    assert not {"--out", "--config", "--resume"} & set(file_args), "configuration files must not set --out/--config/--resume"
    if "argv" in cfg:
        assert not rest, f"{cfgp} is locked: overrides are taken at the first start only, got {rest}"
        if file_args != cfg.get("config_args", file_args):
            print(f"warning: {p0.config} changed after the run was locked; using the locked arguments", flush=True)
        use, info = cfg["argv"], {"argv": cfg["argv"], "config_args": cfg.get("config_args"), "locked": True}
    else:
        use = file_args + rest
        info = {"argv": use, "config_args": file_args, "locked": False}
    core = ["--config", p0.config, "--out", p0.out] + (["--resume", p0.resume] if p0.resume else [])
    return use + core, info


def latest_checkpoint(out):
    """The newest OUT/ckpt/it_*.pkl (for --resume latest), None if there is none."""
    ck = sorted(Path(out).glob("ckpt/it_*.pkl"))
    return str(ck[-1]) if ck else None


def git_state(repo=REPO):
    """(commit, dirty, diff) of the source tree, ignoring configs/; (None, False, "") outside a git checkout."""
    def run(*c):
        r = subprocess.run(["git", "-C", str(repo), *c], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    top, commit = run("rev-parse", "--show-toplevel"), run("rev-parse", "HEAD")
    if commit is None or Path(top.strip()).resolve() != Path(repo).resolve():
        return None, False, ""
    diff = run("diff", "HEAD", "--", ".", ":(exclude)configs") or ""
    return commit.strip(), bool(diff.strip()), diff
