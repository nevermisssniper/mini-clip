from dataclasses import dataclass


@dataclass
class Config:
    # data
    data_dir: str = "data/flickr8k"
    captions_csv: str = "data/flickr8k/captions_clean.csv"
    splits_dir: str = "data/splits"
    image_size: int = 224
    max_length: int = 32

    # model
    embed_dim: int = 512
    freeze_blocks: int = 8

    # training
    batch_size: int = 32
    epochs: int = 15
    temperature: float = 0.07

    # optimizer — separate LRs for backbone vs. projection
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01

    # schedule
    warmup_epochs: int = 1

    # misc
    device: str = "mps"
    num_workers: int = 0       # 0 on MPS to avoid fork issues
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "mini-clip"
