import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap

from .algebra import norm, ax2skew, ax2skew_a, ax2skew_squared

# for small angles we use first order approximations of the equations since
# most of the SO(3) and SE(3) equations get singular for psi -> 0.
# angle_singular = 1.0e-6
angle_singular = 0.0

eye3 = jnp.eye(3, dtype=jnp.float32)


def Spurrier(R: np.ndarray) -> np.ndarray:
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    decision = np.zeros(4, dtype=jnp.float32)
    decision[:3] = np.diag(R)
    decision[3] = np.trace(R)
    i = np.argmax(decision)

    quat = np.zeros(4, dtype=jnp.float32)
    if i != 3:
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i + 1] = np.sqrt(0.5 * R[i, i] + 0.25 * (1 - decision[3]))
        quat[0] = (R[k, j] - R[j, k]) / (4 * quat[i + 1])
        quat[j + 1] = (R[j, i] + R[i, j]) / (4 * quat[i + 1])
        quat[k + 1] = (R[k, i] + R[i, k]) / (4 * quat[i + 1])

    else:
        quat[0] = 0.5 * np.sqrt(1 + decision[3])
        quat[1] = (R[2, 1] - R[1, 2]) / (4 * quat[0])
        quat[2] = (R[0, 2] - R[2, 0]) / (4 * quat[0])
        quat[3] = (R[1, 0] - R[0, 1]) / (4 * quat[0])

    return quat


def quat2axis_angle(Q: np.ndarray) -> np.ndarray:
    """Extract the rotation vector psi for a given quaterion Q = [q0, q] in
    accordance with Wiki2021.

    References
    ----------
    Wiki2021: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation
    """
    q0, vq = Q[0], Q[1:]
    q = norm(vq)
    if q > 0:
        axis = vq / q
        angle = 2 * np.arctan2(q, q0)
        return angle * axis
    else:
        return np.zeros(3)


@jit
def Exp_SO3_quat(P, normalize: bool = True):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.163), Nuetzi2016 (3.31) and Rucker2018 (13).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0], P[1:]
    P2 = P @ P

    return jnp.where(
        normalize,
        eye3 + (2.0 / P2) * (p0 * ax2skew(p) + ax2skew_squared(p)),
        (p0**2 - p @ p) * eye3 + jnp.outer(p, 2.0 * p) + 2.0 * p0 * ax2skew(p),
    )


Exp_SO3_quat_batch = jit(vmap(Exp_SO3_quat))

Exp_SO3_quat_p = jit(jacfwd(Exp_SO3_quat, argnums=0))
Exp_SO3_quat_p_batch = jit(vmap(jacfwd(Exp_SO3_quat, argnums=0)))


Log_SO3_quat = Spurrier


@jit
def T_SO3_quat(P, normalize=True):
    """Tangent map for unit quaternion. See Egeland2002 (6.327).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = P[0], P[1:]

    return jnp.where(
        normalize,
        (2 / (P @ P)) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p))),
        2 * (P @ P) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p))),
    )


T_SO3_quat_batch = jit(vmap(T_SO3_quat))

T_SO3_quat_P = jit(jacfwd(T_SO3_quat, argnums=0))
T_SO3_quat_P_batch = jit(vmap(jacfwd(T_SO3_quat, argnums=0)))


@jit
def T_SO3_inv_quat(P, normalize=True):
    """Inverse tangent map for unit quaternion. See Egeland2002 (6.329) and
    (6.330), Nuetzi2016 (3.11) and (4.19) as well as Rucker2018 (21) 
    and (22).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0], P[1:]
    return jnp.where(
        normalize,
        0.5 * jnp.vstack((-p, p0 * eye3 + ax2skew(p))),
        1 / (2 * (P @ P) ** 2) * jnp.vstack((-p, p0 * eye3 + ax2skew(p))),
    )


T_SO3_inv_quat_batch = jit(vmap(T_SO3_inv_quat))

T_SO3_inv_quat_P = jit(jacfwd(T_SO3_inv_quat, argnums=0))
T_SO3_inv_quat_P_batch = jit(vmap(jacfwd(T_SO3_inv_quat, argnums=0)))
