import numpy as np
import cProfile
from time import perf_counter
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

from cardillo.solver import Solution
from cardillo.visualization import Renderer

from rod_ivp import dydt, event, system, y0, split_index

t0 = system.t0
t1 = 10

t_span = (t0, t1)
t_eval = np.linspace(t0, t1, 1000)


dydt(0, y0) # warm up

print("calculating........")
prof = cProfile.Profile()
prof.enable()

t = perf_counter()
sol = solve_ivp(
    dydt,
    t_span,
    y0.copy(),
    t_eval=t_eval,
    method="RK45",
    dense_output=True,
    events=[event],
    rtol=1e-3,
    atol=1e-6,
)
print(f"{t1 - t0} seconds simulation takes {perf_counter() - t:.2f} seconds!")
print("end")
assert sol.success

prof.disable()
prof.dump_stats("prof.prof")
# print(sol)

# exit()
q, u, la_c = np.split(sol.y, split_index)

plt.subplot(2,1,1)
plt.plot(t_eval, q[-7, :])
plt.grid()
plt.subplot(2,1,2)
plt.plot(t_eval, q[-6, :])
plt.grid()
plt.show(block=False)

sol = Solution(system, t_eval, q.T, u.T)
ren = Renderer(system)
ren.render_solution(sol, repeat=True)
