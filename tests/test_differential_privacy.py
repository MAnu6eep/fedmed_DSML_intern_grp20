import pytest

from fedmed.privacy.differential_privacy import (
    DifferentialPrivacyConfig,
    apply_dp,
)


def test_default_configuration():
    config = DifferentialPrivacyConfig()

    assert config.enabled is False
    assert config.max_grad_norm == 1.0
    assert config.noise_multiplier == 1.0
    assert config.epsilon is None
    assert config.delta is None


def test_custom_configuration():
    config = DifferentialPrivacyConfig(
        enabled=True,
        max_grad_norm=2.0,
        noise_multiplier=0.5,
        epsilon=8.0,
        delta=1e-5,
    )

    assert config.enabled is True
    assert config.max_grad_norm == 2.0
    assert config.noise_multiplier == 0.5
    assert config.epsilon == 8.0
    assert config.delta == 1e-5


def test_valid_configuration():
    config = DifferentialPrivacyConfig(
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        epsilon=5.0,
        delta=1e-5,
    )

    config.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_grad_norm", 0.0),
        ("max_grad_norm", -1.0),
        ("noise_multiplier", -1.0),
        ("epsilon", 0.0),
        ("epsilon", -1.0),
    ],
)
def test_invalid_positive_parameters(field, value):
    config = DifferentialPrivacyConfig(**{field: value})

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.1])
def test_invalid_delta(delta):
    config = DifferentialPrivacyConfig(delta=delta)

    with pytest.raises(ValueError):
        config.validate()


def test_apply_dp_when_disabled():
    gradients = [1.0, 2.0, 3.0]
    config = DifferentialPrivacyConfig(enabled=False)

    result = apply_dp(config, gradients)

    assert result == gradients


def test_apply_dp_when_enabled_not_implemented():
    gradients = [1.0, 2.0, 3.0]
    config = DifferentialPrivacyConfig(enabled=True)

    with pytest.raises(NotImplementedError):
        apply_dp(config, gradients)