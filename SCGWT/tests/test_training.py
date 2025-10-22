import torch

from scgwt.config import load_training_config

from scgwt.training import RolloutBuffer, StepResult, TrainingLoop


def test_rollout_buffer_sample_shapes() -> None:
    buffer = RolloutBuffer(capacity=8)
    observation = torch.zeros(3, 2)
    action = torch.ones(2)
    next_observation = torch.ones(3, 2)
    self_state = torch.zeros(2)
    for _ in range(5):
        buffer.push(observation, action, next_observation, self_state)
    sampled_obs, sampled_actions, sampled_next, sampled_states = buffer.sample(batch_size=4)
    assert sampled_obs.shape == (4, 3, 2)
    assert sampled_actions.shape == (4, 2)
    assert sampled_next.shape == (4, 3, 2)
    assert sampled_states is not None
    assert sampled_states.shape == (4, 2)


def test_training_loop_samples_action_without_input() -> None:
    config = load_training_config("configs/testing.yaml")
    loop = TrainingLoop(config)
    observation = torch.zeros(1, *config.encoder.observation_shape)
    self_state = torch.zeros(1, config.self_state_dim)
    result = loop.step(observation, action=None, self_state=self_state, train=False)
    assert isinstance(result, StepResult)
    assert result.action.shape == (1, config.dynamics.action_dim)
    assert result.reward_components is not None
    for key in ("competence", "empowerment", "safety"):
        assert key in result.reward_components
        assert result.reward_components[key].shape[0] == observation.shape[0]
    assert result.training_metrics is None
    assert result.training_loss is None


def test_training_loop_step_returns_result(tmp_path) -> None:
    config = load_training_config("configs/testing.yaml")
    loop = TrainingLoop(config)
    observation = torch.zeros(2, *config.encoder.observation_shape)
    action = torch.zeros(2, config.dynamics.action_dim)
    next_observation = torch.zeros(2, *config.encoder.observation_shape)
    self_state = torch.zeros(2, config.self_state_dim)
    result = loop.step(
        observation,
        action,
        next_observation=next_observation,
        self_state=self_state,
        train=True,
    )
    assert isinstance(result, StepResult)
    assert result.action.shape == (observation.shape[0], config.dynamics.action_dim)
    assert result.intrinsic_reward.shape[0] == observation.shape[0]
    assert result.observation_entropy.shape == (observation.shape[0],)
    assert result.novelty.shape == (observation.shape[0], config.encoder.num_slots)
    assert result.slot_scores.shape == (observation.shape[0], config.encoder.num_slots)
    assert result.reward_components is not None
    assert set(result.reward_components.keys()) == {"competence", "empowerment", "safety"}
    assert result.training_loss is None
    assert result.training_metrics is None
