import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986

from cardillo import System
from cardillo.forces import B_Moment, Force

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
system.add(*nodes)
system.add(rod)
system.add(force)
system.assemble()

M_inv = np.linalg.inv(system.M(0, system.q0).toarray())
C_inv = 1 / system.c_la_c().toarray().diagonal()

split_index = np.array([system.nq, system.nq + system.nu])

@njit(cache=True)
def _normalize_quat(q):
    # normalize quaternion
    for i in range(len(q)//7):
        d1 = 7*i+3
        d2 = 7*i+7
        q[d1:d2] /= norm(q[d1:d2])
    

def event(t, y):
    q_end, u_end = split_index
    q, u = y[:q_end], y[q_end:u_end]
    # q, u = system.step_callback(t, q, u)
    _normalize_quat(q)
    # set true forces
    # la_c = y[u_end :]
    la_c_true = rod.la_c(t, q, u)
    y[u_end :] = la_c_true
    return 1

# equation of motion with jit
@njit(cache=True)
def _dydt(y, split_index, h_part, W_c, nnode):
    q_end, u_end = split_index
    q, u, la_c = y[:q_end], y[q_end:u_end], y[u_end:]

    # allocate memory
    _dydt = np.zeros_like(y)
    q_dot, u_dot, la_c_dot = _dydt[:q_end], _dydt[q_end:u_end], _dydt[u_end:]
    
    # q_dot
    for i in range(nnode):
        q_dot[7*i:7*i+3] = u[6*i:6*i+3]
        q_dot[7*i+3:7*i+7] = T_SO3_inv_quat(q[7*i+3:7*i+7], normalize=False) @ u[6*i+3:6*i+6]
    
    h = W_c @ la_c
    h[-6:] += h_part
    u_dot[:] = M_inv @ h
    
    la_c_dot[:] = -(W_c.T @ u) 
    la_c_dot *= C_inv

    # fix the first rod node
    q_dot[:7] = 0
    u_dot[:7] = 0

    return _dydt


def dydt(t, y):
    q = y[:split_index[0]]
    
    W_c = rod.W_c(t, q)    
    h_part = force.h(t, q[-7:], q[-6:])
    
    dydt = _dydt(y, split_index, h_part, W_c, rod.nnode)
    return dydt

