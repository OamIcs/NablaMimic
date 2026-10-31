"""NablaMimic: whole-body motion tracking for the Unitree G1 trained by backpropagation through a differentiable
simulator.

Importing the package loads nothing heavy: MuJoCo Warp reads its backward switches at import, so it is imported only
after mjw_options.configure() (sim.make_warp, evaluate.py, bones/benchmark.py).
"""

__version__ = "0.1.0"
