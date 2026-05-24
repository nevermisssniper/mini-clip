# src/data.py

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer


class Flickr8kDataset(Dataset):
    """
    Returns (image, input_ids, attention_mask) for a given index.
    Each sample is one (image, caption) pair. With ~5 captions per image,
    the same image appears ~5 times in this dataset with different captions.
    """

    def __init__(
        self,
        captions_csv: Path,
        images_dir: Path,
        split_file: Path,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 32,
        image_size: int = 224,
        is_train: bool = True,
    ):
        df = pd.read_csv(captions_csv)
        split_images = set(Path(split_file).read_text().splitlines())
        self.df = df[df['image'].isin(split_images)].reset_index(drop=True)

        self.images_dir = Path(images_dir)
        self.max_length = max_length

        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(self.images_dir / row['image']).convert("RGB")
        image = self.transform(image)

        encoding = self.tokenizer(
            row['caption'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


def build_dataloaders(
    captions_csv: Path,
    images_dir: Path,
    splits_dir: Path,
    batch_size: int = 256,
    num_workers: int = 4,
    max_length: int = 32,
    image_size: int = 224,
):
    """Build train/val/test DataLoaders with sensible defaults."""

    train_ds = Flickr8kDataset(
        captions_csv=captions_csv,
        images_dir=images_dir,
        split_file=splits_dir / "train.txt",
        max_length=max_length,
        image_size=image_size,
        is_train=True,
    )
    val_ds = Flickr8kDataset(
        captions_csv=captions_csv,
        images_dir=images_dir,
        split_file=splits_dir / "val.txt",
        max_length=max_length,
        image_size=image_size,
        is_train=False,
    )
    test_ds = Flickr8kDataset(
        captions_csv=captions_csv,
        images_dir=images_dir,
        split_file=splits_dir / "test.txt",
        max_length=max_length,
        image_size=image_size,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader
