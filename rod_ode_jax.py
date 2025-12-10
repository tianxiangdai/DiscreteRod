import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986

from cardillo import System
from cardillo.forces import B_Moment, Force
from cardillo.constraints import RigidConnection

from tdcrobots.math import norm, T_SO3_inv_quat

from discreterod.jax_rod import DiscreteRod

nelement = 40
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

###################################################
Q = DiscreteRod.straight_configuration(nelement, L)
rod = DiscreteRod(
    cross_section,
    material_model,
    nelement,
    Q,
    cross_section_inertias=cross_section_inertias,
)
nodes = rod.nodes

system_statics = System()

f_fun_statics = np.array([0, -0.5, 0])
    
force = Force(lambda t: t*f_fun_statics, nodes[-1])
rb = RigidConnection(system_statics.origin, nodes[0])
system_statics.add(rb)
system_statics.add(*nodes)
system_statics.add(rod)
system_statics.add(force)
system_statics.assemble()

###################################################
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
    return np.zeros(3, np.float64)
    # return np.array([0, -0.5, 0])
    # return (0.5 - np.abs(t - 0.5)) * np.array([0, -1, 0]) * (t<=1)
    
force = Force(f_fun, nodes[-1])
rb = RigidConnection(system.origin, nodes[0])
system.add(rb)
system.add(*nodes)
system.add(rod)
system.add(force)
system.assemble()
###################################################

M_inv = np.linalg.inv(system.M(0, system.q0).toarray())
C_inv = 1 / system.c_la_c().toarray().diagonal()


@njit(cache=True)
def _normalize_quat(q):
    # normalize quaternion
    for i in range(len(q)//7):
        d1 = 7*i+3
        d2 = 7*i+7
        q[d1:d2] /= norm(q[d1:d2])
    


# equation of motion with jit
@njit(cache=True)
def _dydt_rateform(t, y, split_index, h_part, W_c, nnode):
    q_end, u_end = split_index
    q, u, la_c = y[:q_end], y[q_end:u_end], y[u_end:]

    # allocate memory
    y_dot = np.zeros_like(y)
    q_dot, u_dot, la_c_dot = y_dot[:q_end], y_dot[q_end:u_end], y_dot[u_end:]
    
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

    return y_dot


# equation of motion with jit
@njit(cache=True)
def _dydt(t, y, split_index, h, nnode):
    q_end, u_end = split_index
    q, u = y[:q_end], y[q_end:u_end]

    # allocate memory
    y_dot = np.zeros_like(y)
    q_dot, u_dot = y_dot[:q_end], y_dot[q_end:]
    
    # q_dot
    for i in range(nnode):
        q_dot[7*i:7*i+3] = u[6*i:6*i+3]
        q_dot[7*i+3:7*i+7] = T_SO3_inv_quat(q[7*i+3:7*i+7], normalize=False) @ u[6*i+3:6*i+6]
    
    u_dot[:] = M_inv @ h

    # fix the first rod node
    q_dot[:7] = 0
    u_dot[:7] = 0

    return y_dot