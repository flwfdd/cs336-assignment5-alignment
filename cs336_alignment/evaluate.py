import asyncio
import json
import os
import random
import re
from typing import Callable, Dict, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for index, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        if re.search(r"</think>\s*<answer>", generated_text):
            generated_text = re.sub(
                r"</think>\s*<answer>", "</think> <answer>", generated_text
            )
        results.append(
            {
                "prompt": output.prompt,
                "answer": answers[index],
                "generated_text": generated_text,
                "metrics": reward_fn(generated_text, answers[index]),
            }
        )
    with open("evaluation_results.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def evaluate_openai_api(
    api_key: str,
    base_url: str,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    answers: List[str],
    params: Dict,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    async def _run_async_evaluation():
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        sem = asyncio.Semaphore(100)

        async def process_single(prompt, answer):
            max_retries = 5
            async with sem:
                for attempt in range(max_retries):
                    try:
                        response = await client.chat.completions.create(
                            model=params.get("model", "deepseek-chat"),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=params.get("temperature", 1.0),
                            top_p=params.get("top_p", 1.0),
                            max_tokens=params.get("max_tokens", 1024),
                        )
                        text = response.choices[0].message.content or ""
                        if re.search(r"</think>\s*<answer>", text):
                            text = re.sub(
                                r"</think>\s*<answer>", "</think> <answer>", text
                            )
                        metrics = reward_fn(text, answer)
                        return {
                            "prompt": prompt,
                            "answer": answer,
                            "generated_text": text,
                            "metrics": metrics,
                        }
                    except Exception as e:
                        print(
                            f"Error on attempt {attempt + 1} for prompt: {prompt[:30]}...: {e}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(random.uniform(0, 2**attempt))
                        else:
                            print(
                                f"Failed to process prompt after {max_retries} attempts: {prompt[:30]}..."
                            )
                            return {
                                "prompt": prompt,
                                "answer": answer,
                                "generated_text": "",
                                "metrics": {
                                    "format_reward": 0.0,
                                    "answer_reward": 0.0,
                                    "reward": 0.0,
                                },
                            }

        tasks = [process_single(p, a) for p, a in zip(prompts, answers)]
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
        await client.close()

        return results

    results = asyncio.run(_run_async_evaluation())
    with open("evaluation_results.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import json

    import pandas as pd
    from drgrpo_grader import r1_zero_reward_fn

    with open("./prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    data = []
    with open("../data/math/validation.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(
                {
                    "prompt": prompt_template.replace("{question}", item["problem"]),
                    "answer": item["solution"],
                }
            )

    # use openai or vllm
    if False:
        evaluate_openai_api(
            api_key=os.environ.get("DEEPSEEK_API_KEY") or "",
            base_url="https://api.deepseek.com",
            reward_fn=r1_zero_reward_fn,
            prompts=[item["prompt"] for item in data],
            answers=[item["answer"] for item in data],
            params={
                "model": "deepseek-chat",
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 1024,
            },
        )
    else:
        llm = LLM(model="/root/models/Qwen2.5-Math-1.5B")
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        evaluate_vllm(
            vllm_model=llm,
            reward_fn=r1_zero_reward_fn,
            prompts=[item["prompt"] for item in data],
            answers=[item["answer"] for item in data],
            eval_sampling_params=sampling_params,
        )
    # Print results
    with open("evaluation_results.jsonl", "r") as f:
        results = [json.loads(line) for line in f]
    metrics = {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}
    for result in results:
        # print("Prompt:", result["prompt"])
        # print("Generated Text:", result["generated_text"])
        # print("Answer:", result["metrics"])
        # print("Metrics:", result["metrics"])
        # print("=" * 50)

        # Aggregate metrics
        for key in metrics.keys():
            metrics[key] += result["metrics"][key]
    print("=" * 50)
    print("Aggregated Results:", metrics)
    for key in metrics.keys():
        metrics[key] /= len(results)
    print("Average Metrics:", metrics)
