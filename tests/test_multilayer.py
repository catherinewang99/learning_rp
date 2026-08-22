"""Multi-layer probing, pooling, and the layer-supervision mapping."""

import torch

from src.losses.layerwise import layer_supervision
from src.models import ProbedModel, build_model
from src.models.probed import pool_features


def test_conv_probing_by_type_execution_order():
    backbone, probe_kwargs = build_model("tiny_cnn", widths=(4, 8, 16), out_dim=5)
    probed = ProbedModel(backbone, **probe_kwargs)  # Conv2d, drop_last=True
    feats = probed.features(probed.params(), torch.randn(3, 3, 16, 16))
    assert list(feats) == ["conv1", "conv2"]  # 3 convs, last dropped
    assert feats["conv1"].shape == (3, 4)  # spatial-pooled to (B, C)
    assert probed.layer_order == ["conv1", "conv2"]


def test_masked_token_pooling():
    h = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = pool_features(h, mask)
    assert pooled.shape == (2, 4)
    assert torch.allclose(pooled[0], h[0, :2].mean(dim=0))
    assert torch.allclose(pooled[1], h[1, 0])


def test_layer_supervision_even_spread():
    guide = [f"g{i}" for i in range(4)]
    target = [f"t{i}" for i in range(8)]
    mapping = layer_supervision(guide, target)
    assert mapping == {"g0": "t0", "g1": "t2", "g2": "t5", "g3": "t7"}

    upper = layer_supervision(guide, target, upper_half=True)
    assert set(upper.values()) <= {"t4", "t5", "t6", "t7"}


def test_gpt2_layernorm_probing_and_masked_pool():
    torch.manual_seed(0)
    backbone, probe_kwargs = build_model(
        "gpt2", vocab_size=64, n_positions=16, n_layer=2, n_embd=32, n_head=2
    )
    probed = ProbedModel(backbone, **probe_kwargs)
    x = {
        "input_ids": torch.randint(0, 64, (3, 10)),
        "attention_mask": torch.ones(3, 10, dtype=torch.long),
    }
    feats = probed.features(probed.params(), x)
    # 2 LNs per block * 2 blocks + ln_f = 5 captured, last dropped -> 4
    assert len(feats) == 4
    for h in feats.values():
        assert h.shape == (3, 32)
