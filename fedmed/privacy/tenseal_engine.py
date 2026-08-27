"""TenSEAL runtime checks and homomorphic-encryption scaffolding."""

import tenseal as ts


def check_tenseal() -> bool:
    """Verify that TenSEAL can create a CKKS context."""
    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.global_scale = 2**40
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if check_tenseal():
        print("TenSEAL runtime check passed.")
    else:
        print("TenSEAL runtime check failed.")