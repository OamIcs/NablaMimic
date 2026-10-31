"""Persistent XLA compilation cache shared by all tools.

Compiling a training step takes minutes; with the cache a repeated launch with the same shapes and code reuses the
executable (no effect on results). Directory: $JAX_COMPILATION_CACHE_DIR, else <$XDG_CACHE_HOME or ~/.cache>/nablamimic/jax.
"""

import os
from pathlib import Path

import jax

CACHE_DIR = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR")
                 or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nablamimic" / "jax")


def enable(path=CACHE_DIR):
    Path(path).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
