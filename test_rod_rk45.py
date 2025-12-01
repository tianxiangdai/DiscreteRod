import numpy as np
import cProfile
from time import perf_counter
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

from cardillo.solver import Solution, Newton
from cardillo.visualization import Renderer
from tdcrobots.math import quat2axis_angle

from rod_ode import _dydt, _dydt_rateform, _normalize_quat, system, rod, force, system_statics

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
    q = y[:split_index[0]]
    
    W_c = rod.W_c(t, q)    
    h_part = force.h(t, q[-7:], q[-6:])
    
    dydt = _dydt_rateform(t, y, split_index, h_part, W_c, rod.nnode)
    return dydt

###############
# rate form ODE
###############
def dydt(t, y):
    q_end, u_end = split_index
    q, u = y[:q_end], y[q_end:u_end]
    
    W_c = rod.W_c(t, q)    
    la_c = rod.la_c(t, q, u)
    h = W_c @ la_c
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


print("calculating........")
# prof = cProfile.Profile()
# prof.enable()

t_sim = perf_counter()

# Solve system
t_eval = np.linspace(ti, tf, 1000)
sol = solve_ivp(
    dydt,
    t_span,
    y0.copy(),
    t_eval=t_eval,
    method="RK45",
    dense_output=True,
    events=[normalize_quat],
    rtol=1e-3,
    atol=1e-6,
)
print(f"{tf - ti} seconds simulation takes {perf_counter() - t_sim:.2f} seconds!")
print("end")

# prof.disable()
# prof.dump_stats("prof.prof")

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
plt.show(block=True)

# render solution
sol = Solution(system, t_eval, q, u)
ren = Renderer(system)
ren.render_solution(sol, repeat=True)
