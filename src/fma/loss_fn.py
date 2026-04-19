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
        targets = F.normalize(targets, dim=-1)

        pred_affinity = embeddings @ embeddings.T
        true_affinity = targets @ targets.T

        return (pred_affinity - true_affinity).pow(2).mean()
    
class VADLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        vad_pred: torch.Tensor,
        vad_target: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(vad_pred.float(), vad_target.float())
    
class HybridLoss(nn.Module):
    def __init__(
        self,
        diversity_weight: float = 1.0,
        affinity_weight: float = 0.0,
        vad_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.diversity_weight = diversity_weight
        self.affinity_weight = affinity_weight
        self.vad_weight = vad_weight

        self.diversity_loss = BatchDiversityLoss()
        self.affinity_loss = AffinityLoss()
        self.vad_loss = VADLoss()

    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        vad_pred: torch.Tensor,
        vad_targets: torch.Tensor,
        affinity_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = masked_mean_pool(sequence_embeddings, attention_mask)

        total_loss = torch.tensor(
            0.0,
            device=sequence_embeddings.device,
            dtype=sequence_embeddings.dtype
        )

        if self.diversity_weight > 0:
            total_loss = total_loss + (
                self.diversity_weight * self.diversity_loss(pooled)
            )

        if self.affinity_weight > 0:
            if affinity_targets is None:
                raise ValueError(
                    "affinity_targets must be provided when affinity_weight > 0."
                )
            total_loss = total_loss + (
                self.affinity_weight * self.affinity_loss(pooled, affinity_targets)
            )

        if self.vad_weight > 0:
            total_loss = total_loss + (
                self.vad_weight * self.vad_loss(vad_pred, vad_targets)
            )

        return total_loss