from typing import Callable

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys "reward", "format_reward", and
        "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.

    Returns:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages: shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
        raw_rewards: shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        metadata: your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    rollout_batch_size = len(rollout_responses)
    # Compute raw rewards
    raw_rewards = torch.zeros(rollout_batch_size)
    for i in range(rollout_batch_size):
        rewards = reward_fn(rollout_responses[i], repeated_ground_truths[i])
        raw_rewards[i] = rewards["reward"]
    # Compute group-normalized rewards
    advantages = torch.zeros_like(raw_rewards)
    n_groups = rollout_batch_size // group_size
    for g in range(n_groups):
        start_idx = g * group_size
        end_idx = start_idx + group_size
        group_rewards = raw_rewards[start_idx:end_idx]
        group_mean = group_rewards.mean()
        if normalize_by_std:
            group_std = group_rewards.std()
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / (
                group_std + advantage_eps
            )
        else:
            advantages[start_idx:end_idx] = group_rewards - group_mean
    return advantages, raw_rewards, {}
