import numpy as np
from dynamics import Lorenz
from controls import RandomSinusControl, GaussianProcessControl
from generator import TrajectoryGenerator
from utils import *

dt = 0.01
length = 5.0
n_traj = 200

dyn_name = 'lorenz'
dyn = Lorenz(gain=(1.0, 1.0, 1.0))
ctrl_name = 'gaussian_process'
ctrl = GaussianProcessControl(dim=3, n_features=100, length_scale=0.5, seed=0)
gen  = TrajectoryGenerator(dyn, ctrl, dt=dt, seed=0)

mode = 'long'

run(gen, mode, n_traj, length, dyn_name, ctrl_name)