from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np
from numba import njit
from array import array

from jax import vmap, jit, jacfwd
from jax import numpy as jnp
from cardillo.math_numba import (
    norm,
    ax2skew,
    Log_SO3_quat,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_quat,
    T_SO3_quat_P,
)

from cardillo.rods._base import RodExportBase
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.rods import CrossSectionInertias

from tdcrobots.discrete import RigidBody


from cardillo import math_jax
from cardillo.utility.coo_matrix import CooMatrix

eye3 = jnp.eye(3)


class DiscreteRod(RodExportBase):
    def __init__(
        self,
        cross_section,
        material_model,
        nelement,
        Q,
        q0=None,
        u0=None,
        cross_section_inertias=CrossSectionInertias(),
        name="discrete_node",
    ):
        super().__init__(cross_section)
        self.material_model = material_model
        self.nodes = self.get_nodes(
            nelement, Q, q0=q0, u0=u0, cross_section_inertias=cross_section_inertias
        )
        self.name = name

        self.nnode = len(self.nodes)
        self.nelement = self.nnode - 1
        self.nla_c = self.nelement * 6
        self.xis = np.linspace(0, 1, self.nnode)
        for xi, node in zip(self.xis, self.nodes):
            node.xi = xi
        self.C_n_inv = material_model.C_n_inv
        self.C_m_inv = material_model.C_m_inv
        self.elDOF = np.array(
            [np.arange(7 * el, 7 * (el + 2)) for el in range(self.nelement)]
        )
        self.elDOF_u = np.array(
            [np.arange(6 * el, 6 * (el + 2)) for el in range(self.nelement)]
        )
        self.elDOF_la_c = np.array(
            [np.arange(6 * el, 6 * (el + 1)) for el in range(self.nelement)]
        )
        for i, node in enumerate(self.nodes):
            nodalDOF = np.arange(7 * i, 7 * (i + 1))
            nodalDOF_u = np.arange(6 * i, 6 * (i + 1))
            node.nodalDOF = nodalDOF
            node.nodalDOF_r = nodalDOF[:3]
            node.nodalDOF_p = nodalDOF[3:]
            node.nodalDOF_r_u = nodalDOF_u[:3]
            node.nodalDOF_p_u = nodalDOF_u[3:]

        # allocate memery
        self._eval_cache = LRUCache(maxsize=2)
        self._deval_cache = LRUCache(maxsize=2)
        # self.__W_c_el = np.zeros((12, 6))

        self.set_reference_strains(Q)

    def set_reference_strains(self, Q):
        self.L = np.array(
            [
                norm(Q[7 * (i + 1) : 7 * (i + 1) + 3] - Q[7 * i : 7 * i + 3])
                for i in range(self.nelement)
            ]
        )
        self.B_Gamma0 = []
        self.B_Kappa0 = []
        A_IB0, B_Gamma0, B_Kappa0 = _eval_batch(Q[self.elDOF], self.L)
        A_IB0 = np.asarray(A_IB0)
        self.B_Gamma0 = np.asarray(B_Gamma0)
        self.B_Kappa0 = np.asarray(B_Kappa0)

    def local_qDOF_P(self, xi):
        n0, n1 = self.get_element_nodes(xi)
        return np.concatenate((n0.nodalDOF, n1.nodalDOF))

    @staticmethod
    def straight_configuration(
        nelement, L, r_OP0=np.zeros(3, dtype=float), A_IB0=np.eye(3, dtype=float)
    ):
        nnodes = nelement + 1
        x0 = np.linspace(0, L, num=nnodes)
        y0 = np.zeros(nnodes)
        z0 = np.zeros(nnodes)
        r_OC = np.vstack((x0, y0, z0))
        r_OC = r_OP0 + (A_IB0 @ r_OC).T
        P = np.repeat(Log_SO3_quat(A_IB0)[None, :], nnodes, axis=0)
        return np.hstack((r_OC, P)).flatten()

    @staticmethod
    def get_nodes(
        nelement, Q, q0=None, u0=None, cross_section_inertias=CrossSectionInertias()
    ):
        nnodes = nelement + 1
        q0 = Q.copy() if q0 is None else q0
        u0 = np.zeros(6 * nnodes, dtype=float) if u0 is None else u0
        Q = Q.reshape((-1, 7))
        q0 = q0.reshape((-1, 7))
        u0 = u0.reshape((-1, 6))
        nodes = []
        for i in range(nnodes):
            if i == 0:
                w = norm(Q[i + 1] - Q[i]) / 2
            elif i == nnodes - 1:
                w = norm(Q[i] - Q[i - 1]) / 2
            else:
                w = (norm(Q[i + 1] - Q[i]) + norm(Q[i] - Q[i - 1])) / 2
            mass = cross_section_inertias.A_rho0 * w
            B_Theta_C = cross_section_inertias.B_I_rho0 * w
            node = RigidBody(
                mass=mass,
                B_Theta_C=B_Theta_C,
                q0=q0[i],
                u0=u0[i],
                name="node_{}".format(i),
            )
            nodes.append(node)
        return nodes

    def assembler_callback(self):
        self.t0 = self.nodes[0].t0
        self.q0 = np.concatenate([n.q0 for n in self.nodes])
        self.qDOF = np.concatenate([n.qDOF for n in self.nodes])
        self.u0 = np.concatenate([n.u0 for n in self.nodes])
        self.uDOF = np.concatenate([n.uDOF for n in self.nodes])
        self.nq_element = len(self.nodes[0].q0) * 2
        self.nu_element = len(self.nodes[0].u0) * 2
        self._nu = len(self.u0)
        self._nq = len(self.q0)
        self._c_la_c_coo()
        # allocate memery
        self._c_q_coo = CooMatrix((self.nla_c, self._nq))
        self._W_c_coo = CooMatrix((self._nu, self.nla_c))       
        self._Wla_c_q_coo = CooMatrix((self._nu, self._nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            #
            self._c_q_coo.allocate(elDOF_la_c, elDOF)
            self._W_c_coo.allocate(elDOF_u, elDOF_la_c)
            self._Wla_c_q_coo.allocate(elDOF_u, elDOF)
        self._c_q_coo.fix_size()
        self._W_c_coo.fix_size()
        self._Wla_c_q_coo.fix_size()

    def get_element_nodes(self, xi):
        idx = np.searchsorted(self.xis, xi, side="right") - 1
        if idx == self.nnode - 1:
            idx -= 1
        return (self.nodes[idx], self.nodes[idx + 1])

    # ############
    # # compliance
    # ############
    def la_c(self, t, q, u):
        return _la_c(
            q,
            self.elDOF,
            self.elDOF_la_c,
            self.L,
            self.B_Gamma0,
            self.B_Kappa0,
            self.__c_la_c_el_inv,
            self.nla_c,
        )

    def c(self, t, q, u, la_c):
        c = np.empty(self.nla_c, dtype=np.float32)
        A_IB, B_Gamma, B_Kappa = self._eval(q)
        for el in range(self.nelement):
            elDOF_la_c = self.elDOF_la_c[el]
            c[elDOF_la_c] = _c_el(
                la_c[elDOF_la_c],
                self.L[el],
                self.B_Gamma0[el],
                self.B_Kappa0[el],
                self.C_n_inv,
                self.C_m_inv,
                B_Gamma[el], B_Kappa[el]
            )
        return c

    def c_la_c(self):
        return self.__c_la_c

    def _c_la_c_coo(self):
        self.__c_la_c = CooMatrix((self.nla_c, self.nla_c))
        self.__c_la_c_el_inv = []
        for el in range(self.nelement):
            elDOF_la_c = self.elDOF_la_c[el]
            c_la_c_el = (
                np.vstack(
                    (
                        np.hstack([self.C_n_inv, np.zeros((3, 3))]),
                        np.hstack([np.zeros((3, 3)), self.C_m_inv]),
                    )
                )
                * self.L[el]
            )
            self.__c_la_c[elDOF_la_c, elDOF_la_c] = c_la_c_el
            self.__c_la_c_el_inv.append(np.linalg.inv(c_la_c_el))
        self.__c_la_c_el_inv = np.array(self.__c_la_c_el_inv)

    def c_q(self, t, q, u, la_c):
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval(q)
        for el in range(self.nelement):
            value = _c_el_qe(self.L[el], B_Gamma_qe[el], B_Kappa_qe[el])
            self._c_q_coo.set_allocated(el, value)
        
        return self._c_q_coo

    def W_c(self, t, q):
        A_IB, B_Gamma, B_Kappa = self._eval(q)
        _W_c_els = _W_c_el_batch(self.L, A_IB, B_Gamma, B_Kappa)
        self._W_c_coo.data[:] = np.asarray(_W_c_els).ravel()
        return self._W_c_coo
    
    def W_c_dense(self, t, q):
        A_IB, B_Gamma, B_Kappa = self._eval(q)
        W_c_els = np.asarray(_W_c_el_batch(self.L, A_IB, B_Gamma, B_Kappa))
        return _W_c_dense(
            W_c_els, self.elDOF, self.elDOF_u, self.elDOF_la_c, self._nu, self.nla_c
        )
        
    def Wla_c_q(self, t, q, la_c):
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval(q)
        W = _Wla_c_el_qe_batch(la_c[self.elDOF_la_c], self.L, A_IB_qe, B_Gamma_qe, B_Kappa_qe)
        self._Wla_c_q_coo.data[:] = np.asarray(W).ravel()
        return self._Wla_c_q_coo
    
    
    # @cachedmethod(
    #     lambda self: self._eval_cache,
    #     key=lambda self, q: hashkey(*q),
    # )
    def _eval(self, q):
        A_IB, B_Gamma, B_Kappa = _eval_batch(q[self.elDOF], self.L)
        A_IB = np.asarray(A_IB)
        B_Gamma = np.asarray(B_Gamma)
        B_Kappa = np.asarray(B_Kappa)
        return A_IB, B_Gamma, B_Kappa
    
    @cachedmethod(
        lambda self: self._deval_cache,
        key=lambda self, q: hashkey(*q),
    )
    def _deval(self, q):
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = _deval_batch(q[self.elDOF], self.L)
        A_IB_qe = np.asarray(A_IB_qe)
        B_Gamma_qe = np.asarray(B_Gamma_qe)
        B_Kappa_qe = np.asarray(B_Kappa_qe)
        return A_IB_qe, B_Gamma_qe, B_Kappa_qe
    

    def r_OP(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        n0, n1 = self.get_element_nodes(xi)
        alpha = (xi - n0.xi) / (n1.xi - n0.xi)
        r_OC0 = qe[:3]
        r_OC1 = qe[7:10]
        P0 = qe[3:7]
        P1 = qe[10:]
        P = (1 - alpha) * P0 + alpha * P1
        r_OC = (1 - alpha) * r_OC0 + alpha * r_OC1
        A_IB = Exp_SO3_quat(P, normalize=True)
        return r_OC + A_IB @ B_r_CP

    def A_IB(self, t, qe, xi):
        n0, n1 = self.get_element_nodes(xi)
        alpha = (xi - n0.xi) / (n1.xi - n0.xi)
        P0 = qe[3:7]
        P1 = qe[10:]
        alpha = (xi - n0.xi) / (n1.xi - n0.xi)
        P = (1 - alpha) * P0 + alpha * P1
        return Exp_SO3_quat(P, normalize=True)


@njit(cache=True)
def _W_c_dense(W_c_els, elDOF, elDOF_u, elDOF_la_c, nu, nla_c):
    W_c = np.zeros((nu, nla_c))
    nelement = len(elDOF)
    for el in range(nelement):
        dof_u = elDOF_u[el]
        dof_la_c = elDOF_la_c[el]


        u0 = dof_u[0]
        u1 = dof_u[-1] + 1
        l0 = dof_la_c[0]
        l1 = dof_la_c[-1] + 1
        

        W_c[u0:u1, l0:l1] += W_c_els[el]

    return W_c

# @njit(cache=True)
def _la_c(
    q,
    elDOF,
    elDOF_la_c,
    L,
    B_Gamma0,
    B_Kappa0,
    c_la_c_el_inv,
    nla_c,
):
    nelement = len(elDOF)
    la_c = np.empty(nla_c, dtype=np.float32)
    A_IB, B_Gamma, B_Kappa = _eval_batch(q[elDOF], L)
    B_Gamma = np.asarray(B_Gamma)
    B_Kappa = np.asarray(B_Kappa)


    for el in range(nelement):
        qe = q[elDOF[el]]
        Le = L[el]
        
        c_el = np.empty(6, dtype=np.float32)

        c_el[:3] = - (B_Gamma[el] - B_Gamma0[el]) * Le
        c_el[3:] = - (B_Kappa[el] - B_Kappa0[el]) * Le
        la_loc = -c_la_c_el_inv[el] @ c_el

        la_c[elDOF_la_c[el]] = la_loc

    return la_c


@jit
def _Wla_c_el_qe_jax(la_c, Le, A_IB_qe, B_Gamma_qe, B_Kappa_qe):
    B_n = la_c[:3]
    B_m = la_c[3:]

    W0 = (B_n[None, :] @ A_IB_qe).squeeze(axis=1)

    common = (
        -0.5 * (math_jax.ax2skew(B_n) * Le @ B_Gamma_qe)
        -0.5 * (math_jax.ax2skew(B_m) * Le @ B_Kappa_qe)
    )

    W = jnp.vstack([
        W0,
        common,
        -W0,
        common
    ])
    return W

_Wla_c_el_qe_batch = jit(vmap(_Wla_c_el_qe_jax))
    
    
@njit(cache=True)
def _c_el(la_c, L, B_Gamma0, B_Kappa0, C_n_inv, C_m_inv, B_Gamma, B_Kappa):
    c_el = np.empty(6, dtype=np.float32)
    #
    B_n = la_c[:3]
    B_m = la_c[3:]

    c_el[:3] = (C_n_inv @ B_n - (B_Gamma - B_Gamma0)) * L
    c_el[3:] = (C_m_inv @ B_m - (B_Kappa - B_Kappa0)) * L
    return c_el

@njit(cache=True)
def _c_el_qe(L, B_Gamma_qe, B_Kappa_qe):
    c_el_qe = np.empty((6, 14), dtype=np.float32)
    c_el_qe[:3] = -B_Gamma_qe * L
    c_el_qe[3:] = -B_Kappa_qe * L
    return c_el_qe

@jit
def _W_c_el_jax(L, A_IB, B_Gamma, B_Kappa):    
    s1 = 0.5 * math_jax.ax2skew(B_Gamma) * L
    s2 = 0.5 * math_jax.ax2skew(B_Kappa) * L
    
    W_c_el = jnp.zeros((12, 6))
    #
    W_c_el = W_c_el.at[:3, :3].set(A_IB)
    W_c_el = W_c_el.at[3:6, :3].set(s1)
    W_c_el = W_c_el.at[3:6, 3:].set(eye3 + s2)
    W_c_el = W_c_el.at[6:9, :3].set(-A_IB)
    W_c_el = W_c_el.at[9:, :3].set(s1)
    W_c_el = W_c_el.at[9:, 3:].set(-eye3 + s2)
    return W_c_el

_W_c_el_batch = jit(vmap(_W_c_el_jax))
  

@jit
def _eval_jax(qe, Le):
    r_OC0 = qe[:3]
    P0 = qe[3:7]

    r_OC1 = qe[7:10]
    P1 = qe[10:]

    r_OC_s = (r_OC1 - r_OC0) / Le

    P = (P0 + P1) / 2
    P_s = (P1 - P0) / Le

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    B_Gamma = A_IB.T @ r_OC_s

    B_Kappa = T @ P_s
    return A_IB, B_Gamma, B_Kappa

_eval_batch = jit(vmap(_eval_jax))
_deval_jax = jit(jacfwd(_eval_jax, argnums=0))
_deval_batch = jit(vmap(jacfwd(_eval_jax, argnums=0)))
