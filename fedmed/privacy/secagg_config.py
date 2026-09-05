"""fedmed/privacy/secagg_config.py

Configuration schemas, security thresholds, and parameter validation
for Flower SecAgg+.
"""

from typing import Dict

from pydantic import BaseModel, Field, model_validator


class SecAggPlusConfig(BaseModel):
    """Configuration parameters for the SecAgg+ protocol."""

    num_clients: int = Field(
        default=3,
        description="Total number of hospital nodes in federation.",
    )

    threshold: int = Field(
        default=2,
        description=(
            "Minimum surviving nodes required to reconstruct "
            "aggregate (k-of-n)."
        ),
    )

    modulus_range: int = Field(
        default=2**31,
        description="Quantization modulus range for secure masking.",
    )

    clipping_bound: float = Field(
        default=10.0,
        description=(
            "L2-norm vector clipping bound prior to integer "
            "quantization."
        ),
    )

    quantization_bits: int = Field(
        default=16,
        description="Bit resolution for fixed-point parameter quantization.",
    )

    enable_dropouts: bool = Field(
        default=True,
        description=(
            "Enables Shamir secret sharing reconstruction "
            "on client dropouts."
        ),
    )

    @model_validator(mode="after")
    def validate_threshold_parameters(self) -> "SecAggPlusConfig":
        """Ensure cryptographic threshold safety constraints."""

        if self.threshold < 2:
            raise ValueError(
                "SecAgg+ threshold must be at least 2 to prevent "
                "individual mask recovery."
            )

        if self.threshold > self.num_clients:
            raise ValueError(
                "Threshold cannot exceed total number of "
                "participating clients."
            )

        if self.quantization_bits not in [8, 16, 32]:
            raise ValueError(
                "Quantization bits must be one of 8, 16, or 32."
            )

        return self

    def to_flower_secagg_dict(self) -> Dict[str, object]:
        """Convert configuration into a Flower-compatible dictionary."""

        return {
            "num_shares": self.num_clients,
            "reconstruction_threshold": self.threshold,
            "modulus_range": self.modulus_range,
            "clipping_bound": self.clipping_bound,
            "quantization_bits": self.quantization_bits,
        }


if __name__ == "__main__":
    # Validate default security scheme
    sec_cfg = SecAggPlusConfig(num_clients=3, threshold=2)

    print("✓ SecAgg+ configuration validated successfully:")
    print(sec_cfg.model_dump_json(indent=2))