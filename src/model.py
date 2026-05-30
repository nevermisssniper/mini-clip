import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import DistilBertModel


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, freeze_blocks: int = 8):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()  # bypass classifier head, expose 768-dim CLS token
        self.vit = vit

        # freeze: patch embed, pos embed, CLS token, blocks 0–(freeze_blocks-1)
        for param in self.vit.conv_proj.parameters():
            param.requires_grad = False
        self.vit.class_token.requires_grad = False
        self.vit.encoder.pos_embedding.requires_grad = False

        for i, block in enumerate(self.vit.encoder.layers):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.proj = nn.Linear(768, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        features = self.vit(x)           # [B, 768]
        embeddings = self.proj(features)  # [B, embed_dim]
        return F.normalize(embeddings, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, model_name: str = "distilbert-base-uncased"):
        super().__init__()

        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids, attention_mask: [B, seq_len]
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B, 768] — CLS token
        embeddings = self.proj(cls)            # [B, embed_dim]
        return F.normalize(embeddings, dim=-1)


class MiniCLIP(nn.Module):
    def __init__(self, embed_dim: int = 512, freeze_blocks: int = 8):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, freeze_blocks=freeze_blocks)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeds = self.image_encoder(images)                            # [B, embed_dim]
        text_embeds = self.text_encoder(input_ids, attention_mask)           # [B, embed_dim]
        return image_embeds, text_embeds
