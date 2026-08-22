"""Properties of the kernel math (pure functions, tiny tensors)."""

import torch

from src.kernels import cka, center_gram, linear_gram, plasticity_kernel


def test_gram_shape_and_symmetry():
    h = torch.randn(7, 5)
    k = linear_gram(h)
    assert k.shape == (7, 7)
    assert torch.allclose(k, k.T, atol=1e-6)


def test_center_gram_zero_mean():
    k = linear_gram(torch.randn(9, 4))
    kc = center_gram(k)
    assert torch.allclose(kc.sum(dim=0), torch.zeros(9), atol=1e-4)


def test_cka_self_is_one_and_scale_invariant():
    k1 = linear_gram(torch.randn(8, 6))
    k2 = linear_gram(torch.randn(8, 3))
    assert torch.allclose(cka(k1, k1), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(cka(k1, k2), cka(1000.0 * k1, k2), atol=1e-5)
    assert cka(k1, k2) < 1.0


def test_plasticity_kernel_shape_psd():
    v = torch.randn(5, 6, 6)
    pi = plasticity_kernel(v)
    assert pi.shape == (5, 5)
    eigenvalues = torch.linalg.eigvalsh(pi)
    assert (eigenvalues > -1e-4).all()
