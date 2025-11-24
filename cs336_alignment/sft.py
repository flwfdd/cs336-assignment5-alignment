import json
import re

import torch
from drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from transformers import PreTrainedModel  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore
from transformers.generation.utils import GenerateDecoderOnlyOutput


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that
    is 1 for the response tokens and 0 for other tokens (prompt or padding).
    Then the returned dictionary should have the following keys:
    - input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        the tokenized prompt and output strings, with the final token sliced off.
    - labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        shifted input ids, i.e., the input ids without the first token.
    - response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        a boolean mask that is True for all tokens in the response,
        and False for all question and padding tokens.
    """
    prompt_tokenized = tokenizer(prompt_strs)
    output_tokenized = tokenizer(output_strs)

    max_len = (
        max(
            [
                len(prompt_ids) + len(output_ids)
                for prompt_ids, output_ids in zip(
                    prompt_tokenized.input_ids, output_tokenized.input_ids
                )
            ]
        )
        - 1
    )  # max(prompt_and_output_lens) - 1
    batch_size = len(prompt_strs)
    pad_token_id: int = tokenizer.pad_token_id  # type: ignore
    input_ids = torch.full(
        (batch_size, max_len), fill_value=pad_token_id, dtype=torch.long
    )
    labels = input_ids.clone()
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for i, (
        prompt_ids,
        output_ids,
    ) in enumerate(
        zip(
            prompt_tokenized.input_ids,
            output_tokenized.input_ids,
        )
    ):
        prompt_len = len(prompt_ids)
        output_len = len(output_ids)
        total_len = prompt_len + output_len - 1
        input_ids[i, :total_len] = torch.tensor(
            prompt_ids + output_ids[:-1], dtype=torch.long
        )
        if total_len < max_len:  # to fix test case but maybe does not matter
            input_ids[i, total_len] = output_ids[-1]
        labels[i, :total_len] = torch.tensor(
            prompt_ids[1:] + output_ids, dtype=torch.long
        )
        response_mask[i, prompt_len - 1 : total_len] = True
    print("Input IDs:", input_ids)
    print("Labels:", labels)
    print("Response Mask:", response_mask)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
    Returns:
        entropy: torch.Tensor of shape (batch_size, seq_len)
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    p_log_p = probs * log_probs
    p_log_p = torch.nan_to_num(p_log_p, nan=0.0)  # handle nan when all logits are -inf
    entropy = -torch.sum(p_log_p, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get the log probabilities of the response tokens given the input ids.
    Args:
        model: PreTrainedModel placed on the correct device and in inference mode if gradients should not be computed
        input_ids: torch.Tensor of shape (batch_size, seq_len) prompt + response tokens
        labels: torch.Tensor of shape (batch_size, seq_len)
        return_token_entropy: bool, whether to return the token entropy
    Returns:
        A dictionary with the following keys:
        - log_probs: torch.Tensor of shape (batch_size, seq_len) conditional log probabilities
        - token_entropy: torch.Tensor of shape (batch_size, seq_len) (if return_token_entropy is True)
    """
    logits = model(input_ids).logits  # (batch_size, seq_len, vocab_size)
    all_log_probs = torch.nn.functional.log_softmax(
        logits, dim=-1
    )  # (batch_size, seq_len, vocab_size)
    log_probs = torch.gather(all_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )  # (batch_size, seq_len)
    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Normalize the tensor along the specified dimension using the mask and normalize_constant.
    Args:
        tensor: torch.Tensor to be normalized
        mask: torch.Tensor of the same shape as tensor, with 1s for valid positions and 0s for masked positions
        normalize_constant: float, the constant to normalize by
        dim: int or None, the dimension to normalize along
    Returns:
        normalized_tensor:the normalized sum, where masked elements (mask == 0) don’t contribute to the sum.
    """
    masked_tensor = tensor * mask
    sum_masked_tensor = torch.sum(masked_tensor, dim=dim)
    normalized_tensor = sum_masked_tensor / normalize_constant
    return normalized_tensor


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
            metadata: Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """
    loss = (
        masked_normalize(-policy_log_probs, response_mask, normalize_constant)
        / policy_log_probs.size(0)
        / gradient_accumulation_steps
    )
    loss.backward()
    metadata = {}
    return loss, metadata


def log_generations(
    prompt_strs: list[str],
    answer_strs: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_strings: list[str] | None = None,
) -> None:
    """
    Log the generations of the model given the prompts.
    Args:
        prompts_strs: list of prompt strings
        outputs_strs: list of output strings
        model: PreTrainedModel placed on the correct device and in inference mode
        tokenizer: PreTrainedTokenizerBase
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            prompt_strs, return_tensors="pt", padding=True, padding_side="left"
        ).to(model.device)
        generated: GenerateDecoderOnlyOutput = model.generate(  # type: ignore
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            stop_strings=stop_strings,
        )
        response_ids = generated.sequences[:, inputs["input_ids"].shape[1] :]  # type: ignore
        response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        response_mask = response_ids.ne(
            tokenizer.pad_token_id  # type: ignore
        )  # (batch_size, response_len)
        logits = torch.stack(generated.scores).permute(  # type: ignore
            1, 0, 2
        )  # (batch_size, seq_len, vocab_size)
        entropy = compute_entropy(logits)  # (batch_size, seq_len)
        entropy = masked_normalize(
            entropy,
            response_mask,
            normalize_constant=response_mask.sum(dim=1),
            dim=1,
        )  # (batch_size,)
        logs = []
        correct_lens = []
        incorrect_lens = []
        for i in range(len(prompt_strs)):
            log = {
                "prompt": prompt_strs[i],
                "answer": answer_strs[i],
                "generation": response_strs[i],
                "reward": r1_zero_reward_fn(response_strs[i], answer_strs[i]),
                "entropy": entropy[i].item(),
                "length": response_mask[i].sum().item(),
            }
            logs.append(log)
            if log["reward"]["reward"] > 0:
                correct_lens.append(log["length"])
            else:
                incorrect_lens.append(log["length"])
        print(
            json.dumps(
                {
                    "logs": logs,
                    "average_entropy": sum([log["entropy"] for log in logs])
                    / len(logs),
                    "average_correct_length": (
                        sum(correct_lens) / len(correct_lens)
                        if len(correct_lens) > 0
                        else 0
                    ),
                    "average_incorrect_length": (
                        sum(incorrect_lens) / len(incorrect_lens)
                        if len(incorrect_lens) > 0
                        else 0
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-Math-1.5B")
    prompt_strs = [
        "I will repeat: <think> </think> <answer>2</answer> <think> </think> <answer>2</answer> <think> </think> <answer>2</answer> <think>",
    ]
    answer_strs = [
        " 2",
    ]
    log_generations(
        prompt_strs, answer_strs, model, tokenizer, 16, stop_strings=["</answer>"]
    )
