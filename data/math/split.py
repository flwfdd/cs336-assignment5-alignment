from pathlib import Path

import pandas as pd

# Read the parquet file and split into train and validation sets
df = pd.read_parquet("math.parquet")

split_idx = int(len(df) * 0.6)

train_df = df[:split_idx]
val_df = df[split_idx:]

train_df.to_json("train.jsonl", orient="records", lines=True, force_ascii=False)
val_df.to_json("validation.jsonl", orient="records", lines=True, force_ascii=False)

print(f"Total samples: {len(df)}")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
