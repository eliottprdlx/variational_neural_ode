import numpy as np
from dynamics import Lorenz, DoublePendulum
from controls import RandomSinusControl, GaussianProcessControl
from generator import TrajectoryGenerator
from utils import *

dt = 0.02
length = 5.0
n_traj = 200
length_scale = 0.05

dyn_name = 'lorenz'
dyn = Lorenz(gain=(10.0, 10.0, 10.0))
state_dim = dyn.state_dim
# x0 = np.array([np.pi / 4, 0.0, np.pi / 4, 0.0], dtype=np.float32)
ctrl_name = 'gaussian'
ctrl = GaussianProcessControl(dim=3, n_features=100, length_scale=length_scale, seed=0)
plot_control(ctrl, dt, length)
gen  = TrajectoryGenerator(dyn, ctrl, dt=dt, seed=0)

mode = 'long'

run(gen, mode, n_traj, length, dyn_name, ctrl_name, length_scale, state_dim, x0=None)