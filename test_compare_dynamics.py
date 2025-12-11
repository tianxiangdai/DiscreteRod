import numpy as np
from matplotlib import pyplot as plt

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.constraints import RigidConnection

from cardillo.solver import *
from cardillo.forces import *

from cardillo.system import System

from discreterod.jax_rod import DiscreteRod

nelement = 10
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

# ##############
# # discrete rod
# ##############
Q = DiscreteRod.straight_configuration(nelement, L)
rod = DiscreteRod(
    cross_section,
    material_model,
    nelement,
    Q,
    cross_section_inertias=cross_section_inertias,
)
nodes = rod.nodes

system1 = System()
force = Force(lambda t: t * np.array([0, -0.5, 0]) * (t < 1), nodes[-1], xi=1)
rc = RigidConnection(system1.origin, nodes[0])

system1.add(*nodes)
system1.add(rod)
system1.add(force)
system1.add(rc)
system1.assemble()

t1 = 5
solver = DualStormerVerlet(system1, t1, t1 / 100)
# solver = Newton(system1, n_load_steps=10)

sol1 = solver.solve()


###########
# mixed rod
###########
Rod = make_CosseratRod(polynomial_degree=1)
Q = Rod.straight_configuration(nelement, L)
rod = Rod(
    cross_section,
    material_model,
    nelement,
    Q=Q,
    cross_section_inertias=cross_section_inertias,
)

system2 = System()

force = Force(lambda t: t * np.array([0, -0.5, 0]) * (t < 1), rod, xi=1)
# force = B_Moment(lambda t: t * np.array([0, 0, EI / L * 2 * np.pi]) * (t<0.1), rod, xi=1)
rc = RigidConnection(rod, system2.origin, xi1=0)

system2.add(rod)
system2.add(force)
system2.add(rc)
system2.assemble()

solver = DualStormerVerlet(system2, t1, t1 / 100)
# system1.M(0, q).toarray().diagonal()
# np.sum(system2.M(0, q).toarray(), axis=0)

# solver = Newton(system, n_load_steps=10)
# from cardillo.visualization import Renderer
# ren = Renderer(system2, [rod])
# ren.start_step_render()

sol2 = solver.solve()

# print(np.allclose(r_OC1s, r_OC2s))


# r_OC1s = sol1.q[-1, rod.qDOF][rod.nodalDOF_r]
# r_OC2s = sol2.q[-1, rod.qDOF][rod.nodalDOF_r]

# plt.plot(r_OC1s[:, 0], r_OC1s[:, 1], "-xr")
# plt.plot(r_OC2s[:, 0], r_OC2s[:, 1], "-b.")
# plt.axis("equal")
# plt.show(block=True)

# plot result
plt.subplot(2, 1, 1)
plt.plot(sol1.t, sol1.q[:, nodes[-1].qDOF[0]])
plt.plot(sol2.t, sol2.q[:, rod.qDOF][:, rod.nodalDOF_r[-1][0]])
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(sol1.t, sol1.q[:, nodes[-1].qDOF[1]])
plt.plot(sol2.t, sol2.q[:, rod.qDOF][:, rod.nodalDOF_r[-1][1]])
plt.grid()
plt.show(block=True)

# # render solution
# step = int(len(q)//1000)
# sol = Solution(system, t_eval[::step], q[::step], u[::step])
# ren = Renderer(system)
# ren.render_solution(sol, repeat=True)
