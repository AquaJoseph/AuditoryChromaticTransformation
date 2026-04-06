import pandas as pd
from src.muse.dataset import MuSeDataset


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

df = pd.read_csv("datasets\\muse\\raw\\muse_v3.csv")
audio_root = "datasets\\muse\\audio"


dataset = MuSeDataset(df=df, audio_root=audio_root)

print("Dataset length:", len(dataset))

for i in range(10):
    sample = dataset[i]

    if sample is None:
        print(f"[{i}] -> None")
        continue

    print(f"[{i}] -> OK")
    print("tokens shape:", sample["tokens"].shape)
    print("vad shape:", sample["vad"].shape)
    print("global_features shape:", sample["global_features"].shape)
    print("num_tokens:", sample["num_tokens"])
    print("-" * 40)