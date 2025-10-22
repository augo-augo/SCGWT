import torch

from scgwt.motivation import estimate_observation_entropy


def test_estimate_observation_entropy_shape() -> None:
    observation = torch.randn(4, 3, 16, 16)
    entropy = estimate_observation_entropy(observation)
    assert entropy.shape == (4,)


def test_entropy_increases_with_variance() -> None:
    low_var = torch.zeros(4, 3, 16, 16)
    high_var = torch.randn(4, 3, 16, 16)
    entropy_low = estimate_observation_entropy(low_var)
    entropy_high = estimate_observation_entropy(high_var)
    assert torch.all(entropy_high >= entropy_low - 1e-6)
