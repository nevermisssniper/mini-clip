import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds, image_ids=None):
        """Symmetric loss: all captions of the same image are positives.

        Without IDs, preserve the original one-positive-per-row interface.
        With IDs, use uniform probability targets over matching images so
        no valid caption (or augmented view) is trained as a negative.
        """
        logits = (image_embeds @ text_embeds.T) / self.temperature
        if image_ids is None:
            labels = torch.arange(logits.size(0), device=logits.device)
            return (F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.T, labels)) / 2
        ids = image_ids.to(logits.device)
        if ids.ndim != 1 or len(ids) != logits.size(0) or logits.size(0) != logits.size(1):
            raise ValueError("image_ids must contain one ID per aligned image-caption pair")
        positives = (ids[:, None] == ids[None, :]).to(logits.dtype)
        image_targets = positives / positives.sum(dim=1, keepdim=True)
        text_targets = positives.T / positives.T.sum(dim=1, keepdim=True)
        loss_i2t = -(image_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss_t2i = -(text_targets * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
        return (loss_i2t + loss_t2i) / 2
