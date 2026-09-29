"""TenSEAL utilities and homomorphic-encryption engine for FedMed."""

from typing import List, Optional, Tuple
import torch

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None
    TENSEAL_AVAILABLE = False


def check_tenseal() -> bool:
    """Verify that TenSEAL can create a CKKS context."""
    if not TENSEAL_AVAILABLE:
        return False
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
        if not TENSEAL_AVAILABLE:
            raise RuntimeError("TenSEAL library is not installed.")
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.global_scale = global_scale
        self.context = self._create_ckks_context()

    def _create_ckks_context(self):
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

    def flatten_tensor(
        self, tensor: torch.Tensor
    ) -> Tuple[List[float], torch.Size]:
        """Flattens a multi-dimensional PyTorch tensor and saves its original shape."""
        shape = tensor.shape
        flat_list = tensor.detach().cpu().flatten().tolist()
        return flat_list, shape

    def encrypt_flat_tensor(
        self, flat_data: List[float]
    ):
        """Encrypts flattened float values using the local secret context."""
        return ts.ckks_vector(self.context, flat_data)

    def select_parameters(
        self,
        state_dict: dict,
        encrypted_parameters: List[str],
    ) -> Tuple[dict, dict]:
        """Separate model parameters into encrypted and plaintext groups."""
        encrypted = {}
        plaintext = {}

        for name, tensor in state_dict.items():
            if name in encrypted_parameters:
                encrypted[name] = tensor
            else:
                plaintext[name] = tensor

        return encrypted, plaintext

    def prepare_model_parameters(
        self,
        state_dict: dict,
        encrypted_parameters: List[str],
    ) -> dict:
        """Prepare selected model parameters for encrypted transmission."""
        encrypted, plaintext = self.select_parameters(
            state_dict,
            encrypted_parameters,
        )

        prepared = {
            "encrypted": {},
            "plaintext": plaintext,
        }

        for name, tensor in encrypted.items():
            flat_data, shape = self.flatten_tensor(tensor)
            ciphertext = self.encrypt_flat_tensor(flat_data)

            prepared["encrypted"][name] = {
                "ciphertext": self.serialize_ciphertext(ciphertext),
                "shape": tuple(shape),
            }

        return prepared

    def serialize_ciphertext(
        self, enc_vector
    ) -> bytes:
        """Converts an encrypted CKKS vector into byte payload."""
        return enc_vector.serialize()

    def deserialize_ciphertext(
        self,
        enc_bytes: bytes,
        context=None,
    ):
        """Reconstructs CKKSVector from received byte payload."""
        ctx = context or self.context
        return ts.ckks_vector_from(ctx, enc_bytes)

    def decrypt_vector(
        self,
        enc_vector,
        original_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """Decrypts CKKS vector and reshapes it to the original shape."""
        decrypted_list = enc_vector.decrypt()
        tensor = torch.tensor(decrypted_list, dtype=torch.float32)

        if original_shape is not None:
            tensor = tensor.view(original_shape)

        return tensor