import numpy as np
from matplotlib import pyplot as plt

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.constraints import RigidConnection

from cardillo import System
from cardillo.solver import Newton, ScipyDAE, DualStormerVerlet, ScipyIVP
from cardillo.forces import B_Moment, Force

from discreterod.rod import DiscreteRod
from discreterod.math import norm, T_SO3_inv_quat

nelement = 20
radius = 0.03
L = 1
density = 0.4 / (L * np.pi * radius**2)
cross_section = CircularCrossSection(radius)
cross_section_inertias = CrossSectionInertias(
    density=density, cross_section=cross_section
)

E, G = 7e5, 2e5
EA = E * cross_section.area
EI = E * cross_section.second_moment[1, 1]
GA = G * cross_section.area
GJ = G * cross_section.second_moment[0, 0]
material_model = Simo1986(
    np.array([EA, GA, GA]),
    np.array([GJ, EI, EI]),
)

# # ##############
# # # discrete rod
# # ##############
# Q = DiscreteRod.straight_configuration(nelement, L)
# rod = DiscreteRod(
#     cross_section,
#     material_model,
#     nelement,
#     Q,
#     cross_section_inertias=cross_section_inertias,
# )
# nodes = rod.nodes

# system = System()
# force = B_Moment(lambda t: t * np.array([0, 0, EI / L * 2 * np.pi]), nodes[-1])
# rc = RigidConnection(nodes[0], system.origin)

# system.add(*nodes)
# system.add(rod)
# system.add(force)
# system.add(rc)
# system.assemble()

# t1 = 0.2
# # solver = ScipyDAE(system, t1, t1/1000)
# solver = Newton(system, n_load_steps=10)
# sol = solver.solve()

# t, q = sol.t, sol.q
# r_OC1s = np.array([q[-1, n.qDOF[:3]] for n in nodes])


# # ###########
# # # mixed rod
# # ###########
# Rod = make_CosseratRod(polynomial_degree=1)
# Q = Rod.straight_configuration(nelement, L)
# rod = Rod(cross_section, material_model, nelement, Q=Q)

# system = System()
# force = B_Moment(lambda t: t * np.array([0, 0, EI / L * 2 * np.pi]), rod, xi=1)
# rc = RigidConnection(rod, system.origin, xi1=0)

# system.add(rod)
# system.add(force)
# system.add(rc)
# system.assemble()

# t1 = 0.2
# # solver = ScipyDAE(system, t1, t1/1000)
# solver = Newton(system, n_load_steps=10)
# sol = solver.solve()

# t, q = sol.t, sol.q
# r_OC2s = q[-1, rod.qDOF][rod.nodalDOF_r]

# print(np.allclose(r_OC1s, r_OC2s))
# plt.plot(r_OC1s[:, 0], r_OC1s[:, 1], "-xr")
# plt.plot(r_OC2s[:, 0], r_OC2s[:, 1], "-b.")
# plt.axis("equal")
# plt.show(block=True)

###########
# scipy ivp
###########
import cProfile
from time import perf_counter
from scipy.integrate import solve_ivp

from cardillo.solver import Solution
from cardillo.visualization import Renderer

from rod_ivp import dydt, event, system, y0, split_index

t0 = system.t0
t1 = 1

t_span = (t0, t1)
t_eval = np.linspace(t0, t1, 100)


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
sol = Solution(system, t_eval, q.T, u.T)
ren = Renderer(system)
ren.render_solution(sol, repeat=True)
