import torch
from tqdm import tqdm


def encode_all(model, loader, device):
    """Run full dataset through model, return all image + text embeddings."""
    model.eval()
    all_img, all_txt = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="encoding"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            all_img.append(img_emb.cpu())
            all_txt.append(txt_emb.cpu())
    return torch.cat(all_img), torch.cat(all_txt)


def recall_at_k(img_embs, txt_embs, ks=(1, 5, 10), captions_per_image=5):
    """
    Standard Flickr8k retrieval eval.

    img_embs: [N*C, D] — N images, each repeated C times (one per caption)
    txt_embs: [N*C, D] — N*C captions, ordered to match img_embs

    i2t: for each unique image, rank all captions — hit if any correct caption in top K
    t2i: for each caption, rank all unique images — hit if correct image in top K
    """
    n_pairs = img_embs.shape[0]
    n_images = n_pairs // captions_per_image

    # deduplicate images — val/test loaders are unshuffled, rows 0,C,2C,... are unique
    unique_img = img_embs[::captions_per_image]           # [N, D]
    sim_i2t = unique_img @ txt_embs.T                     # [N, N*C]
    sim_t2i = txt_embs @ unique_img.T                     # [N*C, N]

    results = {}

    # i2t
    i2t = {}
    for k in ks:
        top_k = sim_i2t.topk(k, dim=1).indices            # [N, k]
        correct = 0
        for i in range(n_images):
            gt = set(range(i * captions_per_image, (i + 1) * captions_per_image))
            if any(idx.item() in gt for idx in top_k[i]):
                correct += 1
        i2t[k] = correct / n_images
    results['i2t'] = i2t

    # t2i (vectorized)
    t2i = {}
    gt_images = (torch.arange(n_pairs) // captions_per_image).unsqueeze(1)  # [N*C, 1]
    for k in ks:
        top_k = sim_t2i.topk(k, dim=1).indices            # [N*C, k]
        t2i[k] = (top_k == gt_images).any(dim=1).float().mean().item()
    results['t2i'] = t2i

    return results


def print_recalls(results):
    ks = sorted(results['i2t'].keys())
    print(f"{'':>6}", end="")
    for k in ks:
        print(f"  R@{k:>2}", end="")
    print()
    for direction in ('i2t', 't2i'):
        label = "i→t" if direction == 'i2t' else "t→i"
        print(f"{label:>6}", end="")
        for k in ks:
            print(f"  {results[direction][k]*100:>5.1f}", end="")
        print()
