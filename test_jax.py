import numpy as np
from time import perf_counter
from jax import jit, vmap
from jax import numpy as jnp


def benchmark(inputs, fun_cardillo, fun_numba, fun_jax):
    # cardillo
    t0 = perf_counter()
    for el in inputs:
        fun_cardillo(el)
    t_cardillo = perf_counter() - t0
    print(f"time cardillo:\t{t_cardillo:.3e}")

    # numba
    fun_numba(inputs[0])  # warm up
    t0 = perf_counter()
    for el in inputs:
        fun_numba(el)
    t_numba = perf_counter() - t0
    print(f"time numba:\t{t_numba:.3e},\t{t_cardillo/t_numba:.2f}")

    # jax batch
    inputs = jnp.array(inputs)
    batched_fun = jit(vmap(fun_jax))
    # warm up
    batched_fun(inputs).block_until_ready()
    t0 = perf_counter()
    batched_fun(inputs).block_until_ready()
    t_jax = perf_counter() - t0
    print(f"time Jax:\t{t_jax:.3e},\t{t_cardillo/t_jax:.2f}")


from cardillo import math as math_cardillo
from cardillo import math_numba
from cardillo import math_jax


P = np.random.random((100, 4))
print("Test A")
benchmark(P, math_cardillo.Exp_SO3_quat, math_numba.Exp_SO3_quat, math_jax.Exp_SO3_quat)
print("\nTest A_P")
benchmark(P, math_cardillo.Exp_SO3_quat, math_numba.Exp_SO3_quat, math_jax.Exp_SO3_quat)
print("Test T")
benchmark(P, math_cardillo.T_SO3_quat, math_numba.T_SO3_quat, math_jax.T_SO3_quat)
print("\nTest T_P")
benchmark(P, math_cardillo.T_SO3_quat_P, math_numba.T_SO3_quat_P, math_jax.T_SO3_quat_P)
print("\nTest T_inv")
benchmark(
    P, math_cardillo.T_SO3_inv_quat, math_numba.T_SO3_inv_quat, math_jax.T_SO3_inv_quat
)
