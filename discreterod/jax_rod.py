from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np
from numba import njit
from array import array

from tdcrobots.math import (
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


from discreterod import jax_math
from jax import vmap, jit, jacfwd
from jax import numpy as jnp

eye3 = np.eye(3)


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
        self.J = np.array(
            [
                norm(Q[7 * (i + 1) : 7 * (i + 1) + 3] - Q[7 * i : 7 * i + 3])
                for i in range(self.nelement)
            ]
        )
        self.B_Gamma0 = []
        self.B_Kappa0 = []
        A_IB0, B_Gamma0, B_Kappa0 = _eval_batch(Q[self.elDOF], self.J)
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
        coo_c_q = CooMatrix((self.nla_c, self._nq))
        coo_W_c = CooMatrix((self._nu, self.nla_c))       
        coo_Wla_c_q = CooMatrix((self._nu, self._nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            #
            rows, cols = elDOF_la_c, elDOF
            coo_c_q.row.extend(np.repeat(rows, len(cols)))
            coo_c_q.col.extend(np.tile(cols, len(rows)))
            #
            rows, cols = elDOF_u, elDOF_la_c
            coo_W_c.row.extend(np.repeat(rows, len(cols)))
            coo_W_c.col.extend(np.tile(cols, len(rows)))
            #
            rows, cols = elDOF_u, elDOF
            coo_Wla_c_q.row.extend(np.repeat(rows, len(cols)))
            coo_Wla_c_q.col.extend(np.tile(cols, len(rows)))
        self.coo_c_q = coo_c_q
        self.coo_W_c = coo_W_c
        self.coo_Wla_c_q = coo_Wla_c_q

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
            self.J,
            self.B_Gamma0,
            self.B_Kappa0,
            self.__c_la_c_el_inv,
            self.nla_c,
        )

    def c(self, t, q, u, la_c):
        c = np.empty(self.nla_c, dtype=np.float32)
        A_IB, B_Gamma, B_Kappa = self._eval(q)
        A_IB = np.asarray(A_IB)
        B_Gamma = np.asarray(B_Gamma)
        B_Kappa = np.asarray(B_Kappa)
        for el in range(self.nelement):
            elDOF_la_c = self.elDOF_la_c[el]
            c[elDOF_la_c] = _c_el(
                la_c[elDOF_la_c],
                self.J[el],
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
                * self.J[el]
            )
            self.__c_la_c[elDOF_la_c, elDOF_la_c] = c_la_c_el
            self.__c_la_c_el_inv.append(np.linalg.inv(c_la_c_el))
        self.__c_la_c_el_inv = np.array(self.__c_la_c_el_inv)

    def c_q(self, t, q, u, la_c):
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval(q)
        B_Gamma_qe = np.asarray(B_Gamma_qe)
        B_Kappa_qe = np.asarray(B_Kappa_qe)
        n = 0
        for el in range(self.nelement):
            value = _c_el_qe(self.J[el], B_Gamma_qe[el], B_Kappa_qe[el])
            n2 = n + value.size
            self.coo_c_q.data[n:n2] = array("d", value.ravel(order="C"))
            n = n2
        
        return self.coo_c_q

    def W_c(self, t, q):
        A_IB, B_Gamma, B_Kappa = self._eval(q)
        A_IB = np.asarray(A_IB)
        B_Gamma = np.asarray(B_Gamma)
        B_Kappa = np.asarray(B_Kappa)
        n = 0
        for el in range(self.nelement):
            value = _W_c_el(self.J[el], A_IB[el], B_Gamma[el], B_Kappa[el])
            n2 = n + value.size
            self.coo_W_c.data[n:n2] = array("d", value.ravel(order="C"))
            n = n2
        return self.coo_W_c
        
    def Wla_c_q(self, t, q, la_c):
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval(q)
        A_IB_qe = np.asarray(A_IB_qe)
        B_Gamma_qe = np.asarray(B_Gamma_qe)
        B_Kappa_qe = np.asarray(B_Kappa_qe)
        n = 0
        for el in range(self.nelement):
            elDOF_la_c = self.elDOF_la_c[el]
            value = _Wla_c_el_qe(la_c[elDOF_la_c], self.J[el], A_IB_qe[el], B_Gamma_qe[el], B_Kappa_qe[el])
            n2 = n + value.size
            self.coo_Wla_c_q.data[n:n2] = array("d", value.ravel(order="C"))
            n = n2
        return self.coo_Wla_c_q
    
    
    @cachedmethod(
        lambda self: self._eval_cache,
        key=lambda self, q: hashkey(*q),
    )
    def _eval(self, q):
        return _eval_batch(q[self.elDOF], self.J)
    
    @cachedmethod(
        lambda self: self._deval_cache,
        key=lambda self, q: hashkey(*q),
    )
    def _deval(self, q):
        return _deval_batch(q[self.elDOF], self.J)
    

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


# @njit(cache=True)
def _la_c(
    q,
    elDOF,
    elDOF_la_c,
    J,
    B_Gamma0,
    B_Kappa0,
    c_la_c_el_inv,
    nla_c,
):
    nelement = len(elDOF)
    la_c = np.empty(nla_c, dtype=np.float32)
    A_IB, B_Gamma, B_Kappa = _eval_batch(q[elDOF], J)
    B_Gamma = np.asarray(B_Gamma)
    B_Kappa = np.asarray(B_Kappa)


    for el in range(nelement):
        qe = q[elDOF[el]]
        Je = J[el]
        
        c_el = np.empty(6, dtype=np.float32)

        c_el[:3] = - (B_Gamma[el] - B_Gamma0[el]) * Je
        c_el[3:] = - (B_Kappa[el] - B_Kappa0[el]) * Je
        la_loc = -c_la_c_el_inv[el] @ c_el

        la_c[elDOF_la_c[el]] = la_loc

    return la_c


# @njit(cache=True)
def _Wla_c_el_qe(la_c, Je, A_IB_qe, B_Gamma_qe, B_Kappa_qe):
    Wla_c_el_qe = np.empty((12, 14), dtype=np.float32)
    B_n = la_c[:3]
    B_m = la_c[3:]
    # Wla_c_el_qe[:3] = np.einsum("ijk,j", A_IB_qe, B_n)
    Wla_c_el_qe[:3] = (B_n[None,:]@A_IB_qe).squeeze(axis=1)
    Wla_c_el_qe[3:6] = Wla_c_el_qe[9:] = (
        -0.5 * ax2skew(B_n) * Je @ B_Gamma_qe - 0.5 * ax2skew(B_m) * Je @ B_Kappa_qe
    )
    Wla_c_el_qe[6:9] = -Wla_c_el_qe[:3]
    return Wla_c_el_qe

@njit(cache=True)
def _c_el(la_ce, Je, B_Gamma0, B_Kappa0, C_n_inv, C_m_inv, B_Gamma, B_Kappa):
    c_el = np.empty(6, dtype=np.float32)
    #
    B_n = la_ce[:3]
    B_m = la_ce[3:]

    c_el[:3] = (C_n_inv @ B_n - (B_Gamma - B_Gamma0)) * Je
    c_el[3:] = (C_m_inv @ B_m - (B_Kappa - B_Kappa0)) * Je
    return c_el

@njit(cache=True)
def _c_el_qe(Je, B_Gamma_qe, B_Kappa_qe):
    c_el_qe = np.empty((6, 14), dtype=np.float32)
    c_el_qe[:3] = -B_Gamma_qe * Je
    c_el_qe[3:] = -B_Kappa_qe * Je
    return c_el_qe

@njit(cache=True)
def _W_c_el(Je, A_IB, B_Gamma, B_Kappa):
    W_c_el = np.zeros((12, 6))
    #
    s1 = 0.5 * ax2skew(B_Gamma) * Je
    s2 = 0.5 * ax2skew(B_Kappa) * Je
    W_c_el[:3, :3] = A_IB
    W_c_el[3:6, :3] = s1
    W_c_el[3:6, 3:] = eye3 + s2
    W_c_el[6:9, :3] = -A_IB
    W_c_el[9:, :3] = s1
    W_c_el[9:, 3:] = -eye3 + s2
    return W_c_el
    

@jit
def _eval(qe, Je):
    r_OC0 = qe[:3]
    P0 = qe[3:7]

    r_OC1 = qe[7:10]
    P1 = qe[10:]

    r_OC_s = (r_OC1 - r_OC0) / Je

    P = (P0 + P1) / 2
    P_s = (P1 - P0) / Je

    A_IB = jax_math.Exp_SO3_quat(P, normalize=True)
    #
    T = jax_math.T_SO3_quat(P, normalize=True)
    B_Gamma = A_IB.T @ r_OC_s

    B_Kappa = T @ P_s
    return A_IB, B_Gamma, B_Kappa

_eval_batch = jit(vmap(_eval))
_deval = jit(jacfwd(_eval, argnums=0))
_deval_batch = jit(vmap(jacfwd(_eval, argnums=0)))
