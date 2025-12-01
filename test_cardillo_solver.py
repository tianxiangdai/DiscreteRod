import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986

from cardillo import System
from cardillo.forces import B_Moment, Force
from cardillo.constraints import RigidConnection

from tdcrobots.math import norm, T_SO3_inv_quat

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

Q = DiscreteRod.straight_configuration(nelement, L)
rod = DiscreteRod(
    cross_section,
    material_model,
    nelement,
    Q,
    cross_section_inertias=cross_section_inertias,
)
nodes = rod.nodes

system = System()

@njit(cache=True)
def f_fun(t):
    return (0.5 - np.abs(t - 0.5)) * np.array([0, -1, 0]) * (t<1)
    
force = Force(f_fun, nodes[-1])
rc = RigidConnection(nodes[0], system.origin)

system.add(*nodes)
system.add(rod)
system.add(rc)
system.add(force)
system.assemble()


from cardillo.solver import ScipyDAE, Newton, DualStormerVerlet
tf = 10
solver = DualStormerVerlet(system, tf, dt=1e-2)
sol = solver.solve()

#########################
from matplotlib import pyplot as plt
from tdcrobots.math import quat2axis_angle
t_eval, q = sol.t, sol.q
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
from cardillo.visualization import Renderer
ren = Renderer(system)
ren.render_solution(sol, repeat=True)