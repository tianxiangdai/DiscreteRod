import numpy as np
import cProfile
from time import perf_counter

from matplotlib import pyplot as plt

from cardillo.solver import Solution
from cardillo.visualization import Renderer
from tdcrobots.math import quat2axis_angle

from rod_ivp import dydt, event, system, split_index

##########################
ti = system.t0
tf = 10
h = 0.0008

t_span = (ti, tf)
y0 = np.concatenate([system.q0, system.u0, system.la_c0], dtype=np.float64)

# warm up numba
dydt(ti, y0)


def runge_kutta_4_vector(dydt, y0, t0, tf, h):
    n = int((tf - t0) / h)
    
    t = np.zeros(n+1)
    y = np.zeros((n+1, len(y0)))

    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + 0.5*h, y[i] + 0.5*k1)
        k3 = h * dydt(t[i] + 0.5*h, y[i] + 0.5*k2)
        k4 = h * dydt(t[i] + h, y[i] + k3)

        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t[i+1] = t[i] + h

    return t, y


############################
print("calculating........")
prof = cProfile.Profile()
prof.enable()

t_sim0 = perf_counter()

# Solve system
t_eval, y = runge_kutta_4_vector(dydt, y0, ti, tf, h)

print(f"{tf - ti} seconds simulation takes {perf_counter() - t_sim0:.2f} seconds!")
print("end")

prof.disable()
prof.dump_stats("prof.prof")
#########################
q, u, la_c = np.split(y, split_index, axis=1)
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
step = int(len(q)//1000)
sol = Solution(system, t_eval[::step], q[::step], u[::step])
ren = Renderer(system)
ren.render_solution(sol, repeat=True)


