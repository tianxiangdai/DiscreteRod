import numpy as np
import cProfile
from time import perf_counter

from matplotlib import pyplot as plt

from cardillo.solver import Solution, Newton
from cardillo.visualization import Renderer
from cardillo.math_numba import quat2axis_angle

from rod_ode import (
    _dydt,
    _dydt_rateform,
    _normalize_quat,
    system,
    rod,
    force,
    system_statics,
)

rateform = True

ti = system.t0
tf = 10
h = 0.0009
t_span = (ti, tf)

###############
# rate form ODE
###############
split_index = np.array([system.nq, system.nq + system.nu])


def normalize_quat(t, y):
    q_end, u_end = split_index
    q, u = y[:q_end], y[q_end:u_end]
    # q, u = system.step_callback(t, q, u)
    _normalize_quat(q)
    # set true forces
    # la_c = y[u_end :]
    # la_c_true = rod.la_c(t, q, u)
    # y[u_end :] = la_c_true
    return 1


###############
# rate form ODE
###############
def dydt_rateform(t, y):
    q = y[: split_index[0]]

    W_c = rod.W_c(t, q).asformat("array")
    h_part = force.h(t, q[-7:], q[-6:])

    dydt = _dydt_rateform(t, y, split_index, h_part, W_c, rod.nnode)
    return dydt


###############
# rate form ODE
###############
def dydt(t, y):
    q_end, u_end = split_index
    q, u = y[:q_end], y[q_end:u_end]

    W_c = rod.W_c(t, q).asformat("csr")
    la_c = rod.la_c(t, q, u)
    h = W_c @ la_c
    # h = rod.Wla_c(t, q)
    h[-6:] += force.h(t, q[-7:], q[-6:])

    dydt = _dydt(t, y, split_index, h, rod.nnode)
    return dydt


###################
# initial condition
###################
solver = Newton(system_statics)
sol = solver.solve()
if rateform:
    dydt = dydt_rateform
    y0 = np.concatenate([sol.q[-1], sol.u[-1], sol.la_c[-1]], dtype=np.float64)
else:
    dydt = dydt
    y0 = np.concatenate([sol.q[-1], sol.u[-1]], dtype=np.float64)
##########################


def runge_kutta_4_vector(dydt, y0, t0, tf, h):
    n = int((tf - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k1)
        k3 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k2)
        k4 = h * dydt(t[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        normalize_quat(t, y[i + 1])
        t[i + 1] = t[i] + h

    return t, y


############################
dydt(0.0, y0)  # warm up
print("calculating........")
# prof = cProfile.Profile()
# prof.enable()

t_sim = perf_counter()

# Solve system
t_eval, y = runge_kutta_4_vector(dydt, y0, ti, tf, h)

print(f"{tf - ti} seconds simulation takes {perf_counter() - t_sim:.2f} seconds!")
print("end")

# prof.disable()
# prof.dump_stats("prof.prof")
#########################
q, u, la_c = np.split(y, split_index, axis=1)
# plot result
plt.subplot(3, 1, 1)
plt.plot(t_eval, q[:, -7])
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t_eval, q[:, -6])
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t_eval, [quat2axis_angle(np.array(qi))[2] for qi in q[:, -4:]])
plt.grid()
plt.show(block=True)

# render solution
step = int(len(q) // 1000)
sol = Solution(system, t_eval[::step], q[::step], u[::step])
ren = Renderer(system)
ren.render_solution(sol, repeat=True)
