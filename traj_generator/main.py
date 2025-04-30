import numpy as np
from dynamics import Lorenz
from controls import RandomSinusControl
from generator import TrajectoryGenerator
from utils import *

dt = 0.01
length = 3.0
n_traj = 20

dyn_name = 'lorenz'
dyn = Lorenz(gain=(1.0, 1.0, 1.0))
ctrl_name = 'rdsinus'
ctrl = RandomSinusControl(dim=3, n_freq=4, seed=0)
gen  = TrajectoryGenerator(dyn, ctrl, dt=dt, seed=0)

mode = 'long'

run(gen, mode, n_traj, length, dyn_name, ctrl_name)