import torch
import torch.nn as nn

from transformers import BertModel

class BertVADRegressor(nn.Module):
    def __init__(
            self,
            model_name: str = "bert-large-uncased",
            dropout: float = 0.3,
            output_dim: int = 3,
            hidden_dim: int = 512) -> None:
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),

        )
    
    def forward(
        self,
        input_ids : torch.Tensor,
        atten_mask : torch.Tensor)-> torch.Tensor:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=atten_mask
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        return self.regressor(pooled_output)