import json

with open("deepseek_v3_train_evaluation.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

out = []
for i in data:
    if i["metrics"]["reward"] == 1:
        generated_text = i["generated_text"]
        while (
            generated_text.startswith("<think>")
            or generated_text.startswith(" ")
            or generated_text.startswith("\n")
        ):
            if generated_text.startswith("<think>"):
                generated_text = generated_text[len("<think>") :]
            if generated_text.startswith(" ") or generated_text.startswith("\n"):
                generated_text = generated_text[1:]
        out.append({"prompt": i["prompt"], "output": generated_text})
print(f"Filtered {len(out)} / {len(data)} examples.")

with open("sft.jsonl", "w") as f:
    for i in out:
        f.write(json.dumps(i, ensure_ascii=False) + "\n")
