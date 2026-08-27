"""TenSEAL utilities and homomorphic-encryption engine for FedMed."""

from typing import Optional

import tenseal as ts
import torch


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


class TenSEALEngine:
    """Manages CKKS encryption contexts and vector operations."""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: Optional[list] = None,
        global_scale: float = 2**40,
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.global_scale = global_scale
        self.context = self._create_ckks_context()

    def _create_ckks_context(self) -> ts.Context:
        """Validate parameters and build a TenSEAL CKKS context."""
        if self.poly_modulus_degree not in [4096, 8192, 16384]:
            raise ValueError(
                f"poly_modulus_degree must be 4096, 8192, or 16384. "
                f"Got {self.poly_modulus_degree}"
            )

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        context.global_scale = self.global_scale
        context.generate_galois_keys()
        return context

    def get_public_context(self) -> bytes:
        """Serialize context without secret keys."""
        ctx_copy = self.context.copy()
        ctx_copy.make_context_public()
        return ctx_copy.serialize()

    def encrypt_tensor(self, tensor: torch.Tensor):
        """Encrypt a PyTorch float tensor using the local secret context."""
        flat_data = tensor.detach().cpu().numpy().flatten().tolist()
        return ts.ckks_vector(self.context, flat_data)