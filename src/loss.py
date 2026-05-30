import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        # image_embeds, text_embeds: [B, embed_dim], already L2-normed
        # similarity matrix: [B, B]
        logits = (image_embeds @ text_embeds.T) / self.temperature

        # diagonal = correct pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # symmetric: image→text + text→image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
