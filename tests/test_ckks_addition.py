import torch

from fedmed.privacy.tenseal_engine import TenSEALEngine


def test_ckks_encrypted_vector_addition():
    """Verify CKKS encrypted addition matches plaintext addition."""

    # Create validated CKKS context
    engine = TenSEALEngine()

    # Plaintext vectors
    vector_a = torch.tensor([1.0, 2.0, 3.0])
    vector_b = torch.tensor([4.0, 5.0, 6.0])

    # Expected plaintext result
    expected_result = vector_a + vector_b

    # Encrypt both vectors
    encrypted_a = engine.encrypt_tensor(vector_a)
    encrypted_b = engine.encrypt_tensor(vector_b)

    # Perform addition while data remains encrypted
    encrypted_result = encrypted_a + encrypted_b

    # Decrypt the result
    decrypted_result = engine.decrypt_vector(encrypted_result)

    # CKKS uses approximate arithmetic, so compare with tolerance
    assert torch.allclose(
        decrypted_result,
        expected_result,
        atol=1e-3,
        rtol=1e-3,
    ), (
        f"Encrypted addition failed. "
        f"Expected {expected_result.tolist()}, "
        f"but got {decrypted_result.tolist()}"
    )