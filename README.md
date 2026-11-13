# mini-clip

A from-scratch PyTorch implementation of [CLIP](https://arxiv.org/abs/2103.00020) (Radford et al., OpenAI 2021) trained on Flickr8k. Built to understand multimodal contrastive learning end-to-end — encoder design, loss formulation, embedding geometry, and training tradeoffs at small scale.

---

## Architecture

| Component | Detail |
|---|---|
| Image encoder | ViT-B/16, blocks 1–8 frozen, 9–12 fine-tuned |
| Text encoder | DistilBERT, all layers trainable |
| Projection | Linear(768→512) + L2 norm (both towers) |
| Loss | Symmetric InfoNCE, τ = 0.07 (fixed) |
| Optimizer | AdamW — backbone 1e-5, heads 1e-4 |
| Dataset | Flickr8k (6k train / 1k val / 1.09k test images, ~5 captions each) |
| Training | 15 epochs, batch size 256, A100 (~32 min) |

Both encoders project their 768-dim outputs into a shared 512-dim unit hypersphere. The InfoNCE loss maximizes cosine similarity between matched (image, caption) pairs while pushing in-batch examples belonging to other images apart — symmetrically across both retrieval directions.

**Why freeze early ViT blocks?** Early transformer blocks encode low-level features (edges, textures) that transfer cleanly from ImageNet pretraining. Gradient updates are concentrated in the top 4 blocks where task-specific semantic abstractions live.

**Why fix τ = 0.07?** On a 6k-image dataset, learnable τ tends to collapse toward extreme values and destabilize training. Fixed τ at the original CLIP value trades a marginal performance ceiling for stability.

---

## Results

Test set: 1,091 images × 5,455 captions. Best checkpoint: epoch 5 (val loss = 2.4716).

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Image → Text | 21.9% | 51.2% | 64.3% |
| Text → Image | 16.7% | 35.4% | 43.2% |

The i→t / t→i asymmetry is expected — each image has 5 ground-truth captions, making image-to-text retrieval intrinsically more permissive. Val loss diverged from train loss after epoch 5, a known consequence of Flickr8k's scale (30k pairs insufficient to prevent in-batch memorization).

---

## Repo Structure

```
mini-clip/
├── src/
│   ├── config.py       # hyperparameters
│   ├── data.py         # Flickr8k dataset & dataloaders
│   ├── model.py        # ImageEncoder, TextEncoder, MiniCLIP
│   ├── loss.py         # symmetric InfoNCE
│   ├── train.py        # training loop
│   └── eval.py         # Recall@K evaluation
├── train_colab.ipynb   # end-to-end notebook (A100 run)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/mini-clip.git
cd mini-clip
pip install -r requirements.txt
```

Flickr8k images need to be downloaded separately (license-restricted). Place them under `data/flickr8k/`.

**Local (MPS/CPU):**
```bash
python src/train.py
```

**Colab (A100):**
Open `train_colab.ipynb` and run all cells.

---

## Key Design Decisions

- **Shared embedding space** — no predefined geometry; the unit hypersphere structure emerges from jointly optimizing both projection heads against the InfoNCE loss
- **512-dim projection** — bottleneck forces each encoder to distill only cross-modal transferable features; retaining 768-dim allows modality-specific variance to persist into the shared space
- **DistilBERT fully trainable** — larger domain gap from Wikipedia/BookCorpus pretraining to Flickr captions relative to ViT's ImageNet→Flickr gap; all layers benefit from adaptation

---

## Next Steps

- Hard negative mining (semantically similar but non-matching pairs for sharper gradient signal)
- Scale to Flickr30k (150k pairs) or MS-COCO (600k pairs)
- Learnable τ with clamped range — viable at larger data scale without collapse risk
- Cosine LR decay, stronger augmentations (RandAugment, SimCLR-style), deeper ViT unfreezing

---

## References

- Radford et al. (2021) — [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Dosovitskiy et al. (2020) — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Sanh et al. (2019) — [DistilBERT](https://arxiv.org/abs/1910.01108)


## Training correctness updates

The training and validation losses now treat all captions with the same image ID as positives. Only the two new embedding heads use the head learning rate; pretrained attention projections use the backbone rate. The results above are historical results from before these corrections and have not been re-measured.

Run CPU regression tests without downloading encoder weights: `python -m pytest -q tests` (requires PyTorch and pytest).


## Interactive workbench (2026-11-13)

Run `python serve_demo.py`, then open <http://127.0.0.1:8000>. See [QUICKSTART.md](QUICKSTART.md) for core commands, tests, provenance and remaining requirements.
