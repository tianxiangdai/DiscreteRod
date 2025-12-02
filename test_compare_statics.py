import numpy as np
from matplotlib import pyplot as plt

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.constraints import RigidConnection

from cardillo.solver import Newton, ScipyDAE, DualStormerVerlet
from cardillo.forces import B_Moment

# from tdcrobots.rods.discrete_rod import DiscreteRod
from tdcrobots.rods import RodDamping
from cardillo.system import System
from tdcrobots.rods.force_line_distributed import Force_line_distributed

from discreterod.rod import DiscreteRod

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
force = B_Moment(lambda t:  t * np.array([0, 0, EI / L * 2 * np.pi]), nodes[-1])
rc = RigidConnection(system1.origin, nodes[0])

system1.add(*nodes)
system1.add(rod)
system1.add(force)
system1.add(rc)
system1.assemble()

t1 = 1
solver = Newton(system1, n_load_steps=10)
sol = solver.solve()

t, q = sol.t, sol.q
r_OC1s = np.array([q[-1, n.qDOF[:3]] for n in nodes])


###########
# mixed rod
###########
Rod = make_CosseratRod(polynomial_degree=1)
Q = Rod.straight_configuration(nelement, L)
rod = Rod(cross_section, material_model, nelement, Q=Q, cross_section_inertias=cross_section_inertias)

system2 = System()
force = B_Moment(lambda t: t * np.array([0, 0, EI / L * 2 * np.pi]), rod, xi=1)
rc = RigidConnection(rod, system2.origin, xi1=0)

system2.add(rod)
system2.add(force)
system2.add(rc)
system2.assemble()

solver = Newton(system2, n_load_steps=10)
sol = solver.solve()

t, q = sol.t, sol.q
r_OC2s = q[-1, rod.qDOF][rod.nodalDOF_r]

print(np.allclose(r_OC1s, r_OC2s))
plt.plot(r_OC1s[:, 0], r_OC1s[:, 1], "-xr")
plt.plot(r_OC2s[:, 0], r_OC2s[:, 1], "-b.")
plt.axis("equal")
plt.show(block=True)
