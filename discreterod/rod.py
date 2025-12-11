from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np
from numba import njit

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

from cardillo.discrete import RigidBody

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
        _eval_cache = LRUCache(maxsize=self.nnode + 20)
        self._deval_cache = LRUCache(maxsize=self.nnode + 20)

        self.__dr_OC_qe = np.hstack(
            (-np.eye(3), np.zeros((3, 4)), np.eye(3), np.zeros((3, 4)))
        )
        self.__P0_qe = np.hstack((np.zeros((4, 3)), np.eye(4), np.zeros((4, 7))))
        self.__P1_qe = np.hstack((np.zeros((4, 10)), np.eye(4)))
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
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            A_IB, B_Gamma0, B_Kappa0 = _eval(Q[elDOF], self.J[el])
            self.B_Gamma0.append(B_Gamma0)
            self.B_Kappa0.append(B_Kappa0)
        self.B_Gamma0 = np.array(self.B_Gamma0)
        self.B_Kappa0 = np.array(self.B_Kappa0)

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
        c = np.empty(self.nla_c, dtype=np.float64)
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            elDOF_la_c = self.elDOF_la_c[el]
            c[elDOF_la_c] = _c_el(
                qe,
                la_c[elDOF_la_c],
                self.J[el],
                self.B_Gamma0[el],
                self.B_Kappa0[el],
                self.C_n_inv,
                self.C_m_inv,
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
        coo = CooMatrix((self.nla_c, self._nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_la_c, elDOF] = self.c_el_qe(q[elDOF], la_c[elDOF_la_c], el)
        return coo

    def c_el_qe(self, qe, la_ce, el):
        c_el_qe = np.empty((6, 14), dtype=np.float64)
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval(qe, self.J[el])

        c_el_qe[:3] = -B_Gamma_qe * self.J[el]
        c_el_qe[3:] = -B_Kappa_qe * self.J[el]

        return c_el_qe

    def W_c(self, t, q):
        return _W_c(
            self.elDOF, self.elDOF_u, self.elDOF_la_c, self.J, q, self._nu, self.nla_c
        )

    def W_la_c(self, t, q):
        return _W_la_c(
            self.elDOF,
            self.elDOF_u,
            self.elDOF_la_c,
            self.J,
            q,
            self._nu,
            self.B_Gamma0,
            self.B_Kappa0,
            self.__c_la_c_el_inv,
        )

    def Wla_c_q(self, t, q, la_c):
        coo = CooMatrix((self._nu, self._nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_u, elDOF] = self.Wla_c_el_qe(q[elDOF], la_c[elDOF_la_c], el)
        return coo

    def Wla_c_el_qe(self, qe, la_ce, el):
        Wla_c_el_qe = np.empty((12, 14), dtype=np.float64)
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval(qe, self.J[el])
        Je = self.J[el]
        B_n = la_ce[:3]
        B_m = la_ce[3:]
        Wla_c_el_qe[:3] = np.einsum("ijk,j", A_IB_qe, B_n)
        Wla_c_el_qe[3:6] = Wla_c_el_qe[9:] = (
            -0.5 * ax2skew(B_n) * Je @ B_Gamma_qe - 0.5 * ax2skew(B_m) * Je @ B_Kappa_qe
        )
        Wla_c_el_qe[6:9] = -Wla_c_el_qe[:3]
        return Wla_c_el_qe

    # @cachedmethod(
    #     lambda self: self._deval_cache,
    #     key=lambda self, qe, Je: hashkey(*qe, Je),
    # )
    def _deval(self, qe, Je):
        r_OC0 = qe[:3]
        P0 = qe[3:7]

        r_OC1 = qe[7:10]
        P1 = qe[10:]

        r_OC_s = (r_OC1 - r_OC0) / Je
        r_OC_s_qe = self.__dr_OC_qe / Je

        P = (P0 + P1) / 2
        P_s = (P1 - P0) / Je
        P_qe = (self.__P0_qe + self.__P1_qe) / 2
        P_s_qe = (self.__P1_qe - self.__P0_qe) / Je

        A_IB = Exp_SO3_quat(P, normalize=True)
        A_IB_qe = Exp_SO3_quat_p(P, normalize=True) @ P_qe
        #
        T = T_SO3_quat(P, normalize=True)
        # B_Gamma = A_IB.T @ r_OC_s
        B_Gamma_qe = np.einsum("k,kij", r_OC_s, A_IB_qe) + A_IB.T @ r_OC_s_qe

        # B_Kappa = T @ P_s
        B_Kappa_qe = (
            np.einsum(
                "ijk,j->ik",
                T_SO3_quat_P(P, normalize=True),
                P_s,
            )
            @ P_qe
            + T @ P_s_qe
        )
        # return A_IB, B_Gamma, B_Kappa, r_OC_s_qe, A_IB_qe, B_Gamma_qe, B_Kappa_qe
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
    la_c = np.empty(nla_c, dtype=np.float64)

    for el in range(nelement):
        qe = q[elDOF[el]]
        Je = J[el]

        c_el = np.empty(6, dtype=np.float64)
        A_IB, B_Gamma, B_Kappa = _eval(qe, Je)

        c_el[:3] = -(B_Gamma - B_Gamma0[el]) * Je
        c_el[3:] = -(B_Kappa - B_Kappa0[el]) * Je
        la_loc = -c_la_c_el_inv[el] @ c_el

        la_c[elDOF_la_c[el]] = la_loc

    return la_c


@njit(cache=True)
def _c_el(qe, la_ce, Je, B_Gamma0, B_Kappa0, C_n_inv, C_m_inv):
    c_el = np.empty(6, dtype=np.float64)
    A_IB, B_Gamma, B_Kappa = _eval(qe, Je)
    #
    B_n = la_ce[:3]
    B_m = la_ce[3:]

    c_el[:3] = (C_n_inv @ B_n - (B_Gamma - B_Gamma0)) * Je
    c_el[3:] = (C_m_inv @ B_m - (B_Kappa - B_Kappa0)) * Je
    return c_el


@njit(cache=True)
def _W_c(elDOF, elDOF_u, elDOF_la_c, J, q, nu, nla_c):
    W_c = np.zeros((nu, nla_c))
    nelement = len(elDOF)
    for el in range(nelement):
        dof = elDOF[el]
        dof_u = elDOF_u[el]
        dof_la_c = elDOF_la_c[el]

        qe = q[dof]
        Je = J[el]

        u0 = dof_u[0]
        u1 = dof_u[-1] + 1
        l0 = dof_la_c[0]
        l1 = dof_la_c[-1] + 1

        A_IB, B_Gamma, B_Kappa = _eval(qe, Je)

        W_c[u0:u1, l0:l1] += _W_c_el(Je, A_IB, B_Gamma, B_Kappa)

    return W_c


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


@njit(cache=True)
def _W_la_c(elDOF, elDOF_u, elDOF_la_c, J, q, nu, B_Gamma0, B_Kappa0, c_la_c_el_inv):
    h = np.zeros(nu)
    nelement = len(elDOF)
    for el in range(nelement):
        dof_u = elDOF_u[el]

        qe = q[elDOF[el]]
        Je = J[el]

        u0 = dof_u[0]
        u1 = dof_u[-1] + 1

        c_el = np.empty(6, dtype=np.float64)
        A_IB, B_Gamma, B_Kappa = _eval(qe, Je)

        c_el[:3] = -(B_Gamma - B_Gamma0[el]) * Je
        c_el[3:] = -(B_Kappa - B_Kappa0[el]) * Je
        la_c_el = -c_la_c_el_inv[el] @ c_el

        h[u0:u1] += _W_c_el(Je, A_IB, B_Gamma, B_Kappa) @ la_c_el

    return h


@njit(cache=True)
def _eval(qe, Je):
    r_OC0 = qe[:3]
    P0 = qe[3:7]

    r_OC1 = qe[7:10]
    P1 = qe[10:]

    r_OC_s = (r_OC1 - r_OC0) / Je

    P = (P0 + P1) / 2
    P_s = (P1 - P0) / Je

    A_IB = Exp_SO3_quat(P, normalize=True)
    #
    T = T_SO3_quat(P, normalize=True)
    B_Gamma = A_IB.T @ r_OC_s

    B_Kappa = T @ P_s
    return A_IB, B_Gamma, B_Kappa
