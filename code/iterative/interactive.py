# %%
import numpy as np
from main import OpenAnnulus, Simulator

oa = OpenAnnulus([0,0,0], 0, 1)
p = 2
max_t = 5
dt = 0.01
dx = 0.3
n = 1000
epsilon = 1e-2

sim = Simulator(oa, p, max_t, dt, dx, n)
u = sim.run(epsilon)

grid = oa.generate_grid(dx)
print(u(grid))
# %%
