import numpy as np
import cProfile
from time import perf_counter
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

from cardillo.solver import Solution
from cardillo.visualization import Renderer
from tdcrobots.math import quat2axis_angle

from rod_ivp import dydt, event, system, split_index

##########################
ti = system.t0
tf = 10

t_span = (ti, tf)
t_eval = np.linspace(ti, tf, 1000)
y0 = np.concatenate([system.q0, system.u0, system.la_c0], dtype=np.float64)

# warm up numba
dydt(ti, y0)

############################
print("calculating........")
prof = cProfile.Profile()
prof.enable()

t_sim0 = perf_counter()

# Solve system
sol = solve_ivp(
    dydt,
    t_span,
    y0.copy(),
    t_eval=t_eval,
    method="RK45",
    dense_output=True,
    # events=[event],
    rtol=1e-3,
    atol=1e-6,
)
print(f"{tf - ti} seconds simulation takes {perf_counter() - t_sim0:.2f} seconds!")
print("end")

prof.disable()
prof.dump_stats("prof.prof")

#########################
q, u, la_c = np.split(sol.y.T, split_index, axis=1)
# plot result
plt.subplot(3,1,1)
plt.plot(t_eval, q[:, -7])
plt.grid()
plt.subplot(3,1,2)
plt.plot(t_eval, q[:, -6])
plt.grid()
plt.subplot(3,1,3)
plt.plot(t_eval, [quat2axis_angle(np.array(qi))[2] for qi in q[:, -4:]])
plt.grid()
plt.show(block=False)

# render solution
sol = Solution(system, t_eval, q.T, u.T)
ren = Renderer(system)
ren.render_solution(sol, repeat=True)
