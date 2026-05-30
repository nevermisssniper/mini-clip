import os
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.config import Config
from src.data import build_dataloaders
from src.loss import InfoNCELoss
from src.model import MiniCLIP


def val_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            total += loss_fn(img_emb, txt_emb).item()
    return total / len(loader)


def train(cfg: Config):
    device = torch.device(cfg.device)

    train_loader, val_loader, _ = build_dataloaders(
        captions_csv=Path(cfg.captions_csv),
        images_dir=Path(cfg.data_dir) / "Images",
        splits_dir=Path(cfg.splits_dir),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_length=cfg.max_length,
        image_size=cfg.image_size,
    )

    model = MiniCLIP(embed_dim=cfg.embed_dim, freeze_blocks=cfg.freeze_blocks).to(device)
    loss_fn = InfoNCELoss(temperature=cfg.temperature)

    # separate LRs: projection heads (newly init'd) vs. pretrained backbones
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'proj' in n]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and 'proj' not in n]

    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.lr_backbone},
        {'params': head_params,     'lr': cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[cfg.lr_backbone, cfg.lr_head],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
    )

    wandb.init(project=cfg.wandb_project, config=vars(cfg))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            loss = loss_fn(img_emb, txt_emb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = val_epoch(model, val_loader, loss_fn, device)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch + 1})
        print(f"Epoch {epoch+1:>2}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.checkpoint_dir}/best.pt")
            print(f"  → checkpoint saved (val={val_loss:.4f})")

    wandb.finish()


if __name__ == "__main__":
    train(Config())
