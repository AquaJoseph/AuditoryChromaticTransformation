import torch
import pandas as pd
import configparser
import warnings

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.muse.model import BertVADRegressor
from src.muse.dataset import load_dataframe

from .infer_dataset import build_dataset, build_loader

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


config = configparser.ConfigParser()
config.read("src\\fma\\config.cfg")

fma_df = load_dataframe("datasets/fma/fma_metadata/modified/full_fma_vad_predictions_group.parquet")


tokeniser = AutoTokenizer.from_pretrained(config["model"]["name"])

inference_dataset = build_dataset(fma_df, tokenizer=tokeniser)
inference_loader = build_loader(inference_dataset, batch_size=config.getint("infer", "batch_size"), shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertVADRegressor(
    model_name=config["model"]["name"]
)

model.load_state_dict(
    torch.load("checkpoints\\muse\\bert-large-uncased-saves\\final_bert_vad.pt", map_location=device)
)

preds = []
model = model.to(device)

model.eval()
with torch.no_grad():
    for batch in tqdm(inference_loader, desc="Running Inference on FMA Dataset"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pred = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds.extend(pred.cpu().tolist())

fma_df["vad_pred"] = preds

fma_df[["valence_pred", "arousal_pred", "dominance_pred"]] = pd.DataFrame(
    preds,
    index=fma_df.index
)

fma_df["vad_pred"] = fma_df[
    ["valence_pred", "arousal_pred", "dominance_pred"]
].values.tolist()


fma_df.to_csv(
    "datasets/fma/fma_metadata/modified/full_fma_vad_predictions_group.csv",
    index=False
)

fma_df.to_parquet(
    "datasets/fma/fma_metadata/modified/full_fma_vad_predictions_group.parquet",
    index=False
)