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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
    raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
        reward/advantage for each rollout response.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.

    Returns:
    torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
    """
    return -policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
    advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
    old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
    cliprange: float Clip parameter ϵ (e.g. 0.2).

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss: torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
        loss.
    metadata: dict containing whatever you want to log. We suggest logging whether each
        token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
        the min was lower than the LHS.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    return -torch.min(ratio * advantages, clip_ratio * advantages), {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Select and compute the desired policy-gradient loss.
    Args:
    policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
    loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
    raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages: Required for "reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
    old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).
    cliprange: Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss: (batch_size, sequence_length), per-token loss.
    metadata: dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards must be provided for no_baseline loss.")
        return (
            compute_naive_policy_gradient_loss(
                raw_rewards,
                policy_log_probs,
            ),
            {},
        )
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                "advantages must be provided for reinforce_with_baseline loss."
            )
        return (
            compute_naive_policy_gradient_loss(
                advantages,
                policy_log_probs,
            ),
            {},
        )
    elif loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages must be provided for grpo_clip loss.")
        if old_log_probs is None:
            raise ValueError("old_log_probs must be provided for grpo_clip loss.")
        if cliprange is None:
            raise ValueError("cliprange must be provided for grpo_clip loss.")
        return compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
    tensor: torch.Tensor The data to be averaged.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.

    Returns:
    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_tensor = tensor * mask
    sum_masked = masked_tensor.sum(dim=dim)
    count_masked = mask.sum(dim=dim)
    return sum_masked / count_masked


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
    response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
    gradient_accumulation_steps: Number of microbatches per optimizer step.
    loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
    raw_rewards: Needed when loss_type == "no_baseline"; shape (batch_size, 1).
    advantages: Needed when loss_type != "no_baseline"; shape (batch_size, 1).
    old_log_probs: Required for GRPO-Clip; shape (batch_size, sequence_length).
    cliprange: Clip parameter ϵ for GRPO-Clip.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
    metadata: Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    masked_loss = masked_mean(loss, response_mask)
    adjusted_loss = masked_loss / gradient_accumulation_steps
    adjusted_loss.backward()
    return adjusted_loss, metadata
