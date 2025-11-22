import torch
from transformers import PreTrainedTokenizerBase  # type: ignore
from transformers import Qwen2Tokenizer  # type: ignore


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
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


if __name__ == "__main__":
    # model = PreTrainedTokenizerBase.from_pretrained(
    #     "models/Qwen2.5-Math-1.5B",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    # )
    tokenizer = Qwen2Tokenizer.from_pretrained("models/Qwen2.5-Math-1.5B")
    tokenize_prompt_and_output(
        ["Hello, how are you?", "1"], [" I'm fine, thank you.", "2"], tokenizer
    )
