"""CPU-only tests; no encoder weights or dataset downloads required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'src' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projection_groups_exclude_backbone_attention():
    model = nn.Module()
    for name in ('image_encoder', 'text_encoder'):
        tower = nn.Module()
        tower.proj = nn.Linear(3, 2)
        tower.attention = nn.MultiheadAttention(4, 1)
        model.add_module(name, tower)
    model.image_encoder.attention.in_proj_weight.requires_grad = False
    backbone, heads = load('optim').parameter_groups(model, 1e-5, 1e-4)
    head_ids = {id(p) for p in heads['params']}
    assert id(model.image_encoder.proj.weight) in head_ids
    assert id(model.image_encoder.attention.out_proj.weight) not in head_ids
    all_params = backbone['params'] + heads['params']
    assert len(all_params) == len({id(p) for p in all_params})
    assert {id(p) for p in all_params} == {id(p) for p in model.parameters() if p.requires_grad}


def test_unique_ids_preserve_original_loss():
    loss = load('loss').InfoNCELoss()
    torch.manual_seed(3)
    images, texts = torch.randn(4, 3), torch.randn(4, 3)
    torch.testing.assert_close(loss(images, texts), loss(images, texts, torch.arange(4)))


def test_same_image_caption_permutation_is_invariant():
    loss = load('loss').InfoNCELoss(temperature=1)
    images = torch.tensor([[1., 0.], [.8, .2], [0., 1.]], requires_grad=True)
    texts = torch.tensor([[.9, .1], [.7, .3], [.1, .9]], requires_grad=True)
    ids = torch.tensor([10, 10, 20])
    value = loss(images, texts, ids)
    torch.testing.assert_close(value, loss(images, texts[[1, 0, 2]], ids))
    value.backward()
    assert torch.isfinite(images.grad).all() and torch.isfinite(texts.grad).all()
