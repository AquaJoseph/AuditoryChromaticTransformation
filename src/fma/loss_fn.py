import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mean_pool(
        x: torch.Tensor,
        attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts

class BatchDiversityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        similarity = embeddings @ embeddings.T

        batch_size = similarity.size(0)
        eye = torch.eye(
            batch_size,
            device=similarity.device,
            dtype=torch.bool
        )

        off_diag = similarity[~eye]
        return off_diag.pow(2).mean()
    
class AffinityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)

        pred_affinity = embeddings @ embeddings.T   # [N, N]
        true_affinity = targets @ targets.T         # [N, N]

        return torch.norm(pred_affinity - true_affinity, p="fro") ** 2