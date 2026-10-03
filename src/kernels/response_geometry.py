"""FORM VIGHNESH YAY: Numerically stable comparisons of kernel RESPONSES (FUCKING FINALLY), not hidden coordinates.

R[i] = vec(V_i). For Pi-CKA, center R over experiences before forming R R^T.
This is the same centered linear-kernel comparison in exact arithmetic.
Direct matching compares each signed row direction without experience centering.
"""

from __future__ import annotations

import math
import torch


def precise_kernel(kernel_fn):
    def run(features):
        return kernel_fn(features.double())
    return run


def _safe_rows(v: torch.Tensor):
    if v.ndim != 3 or v.shape[-1] != v.shape[-2] or v.shape[0] == 0:
        raise ValueError(f'Expected nonempty V with shape (m,n,n), got {tuple(v.shape)}')
    #TECHNICALLY this error shouldn't happen above. But idk what the fuck is going on.
    rows = v.flatten(1).double()
    finite = torch.isfinite(rows).all(dim=1)
    clean = torch.where(torch.isfinite(rows), rows, torch.zeros_like(rows))
    return clean, finite


def response_geometry(v: torch.Tensor, min_response_norm: float = 1e-10,
                      min_relative_spread: float = 1e-5):
    """Return raw row directions, centered Pi, and explicit validity flags.

    Pi is valid only when ALL rows are finite/nonzero and between-experience
    variation exceeds min_relative_spread. We do not score a constant response
    bank as either a perfect match or a valid zero-similarity measurement.
    Thresholds apply to V = delta-K / reference-lr and are configurable.
    """
    rows, finite = _safe_rows(v)
    norms = torch.linalg.vector_norm(rows, dim=1)
    row_valid = finite & (norms > min_response_norm)
    directions = rows / norms.clamp_min(min_response_norm).unsqueeze(1)

    # Rescale to prevent overflow, then center before forming the Gram.
    scale = rows.detach().abs().amax().clamp_min(torch.finfo(rows.dtype).tiny)
    scaled = rows / scale
    centered = scaled - scaled.mean(dim=0, keepdim=True)
    raw_norm = torch.linalg.vector_norm(scaled)
    centered_norm = torch.linalg.vector_norm(centered)
    spread = centered_norm / raw_norm.clamp_min(torch.finfo(rows.dtype).tiny)
    valid = (row_valid.all() & (rows.shape[0] >= 3)
             & (spread >= min_relative_spread) & (centered_norm > 0))

    # A branch avoids the derivative of normalization at a zero-variance bank.
    if bool(valid.detach()):
        centered = centered / centered_norm
        gram = centered @ centered.T
        gram = gram / torch.linalg.vector_norm(gram)
    else:
        gram = rows.new_zeros((rows.shape[0], rows.shape[0]))
    return {'rows': rows, 'directions': directions, 'row_valid': row_valid,
            'norms': norms, 'spread': spread, 'gram': gram, 'valid': valid}


def response_scores(target_v: torch.Tensor, guide_v: torch.Tensor,
                    min_response_norm: float = 1e-10,
                    min_relative_spread: float = 1e-5):
    """Scores share experience indices; V matching additionally shares probes.

    Direct V matching is architecture-independent but requires the same probe
    identities/order on the two sides. Pi-CKA permits different response widths.
    """
    if target_v.shape[0] != guide_v.shape[0]:
        raise ValueError('Plasticity comparison requires matched experience counts.')
    target = response_geometry(target_v, min_response_norm, min_relative_spread)
    guide = response_geometry(guide_v, min_response_norm, min_relative_spread)
    out = {'target': target, 'guide': guide,
           'pi_valid': bool((target['valid'] & guide['valid']).detach()),
           'v_valid_fraction': 0.0}
    if out['pi_valid']:
        # Frobenius cosine of centered PSD Grams. Clamp only roundoff at bounds.
        out['pi_similarity'] = (target['gram'] * guide['gram']).sum().clamp(0, 1)
    if target_v.shape[1:] == guide_v.shape[1:]:
        valid = target['row_valid'] & guide['row_valid']
        out['v_valid_fraction'] = valid.double().mean().item()
        if bool(valid.any()):
            dots = (target['directions'][valid] * guide['directions'][valid]).sum(1)
            out['v_cosine'] = dots.clamp(-1, 1).mean()
    return out


def gram_cka(k1: torch.Tensor, k2: torch.Tensor):
    """Stable ordinary CKA for representation kernels; None when undefined."""
    if k1.shape != k2.shape or k1.ndim != 2 or k1.shape[0] != k1.shape[1]:
        raise ValueError('K-CKA needs square kernels on the same probe set.')
    centered = []
    for k in (k1, k2):
        k = k.double()
        if not bool(torch.isfinite(k).all()):
            return None
        scale = k.detach().abs().amax()
        if float(scale) == 0:
            return None
        k = k / scale
        k = k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()
        norm = torch.linalg.vector_norm(k)
        if float(norm.detach()) < 1e-12:
            return None
        centered.append(k / norm)
    return (centered[0] * centered[1]).sum().clamp(0, 1)


@torch.no_grad()
def geometry_diagnostics(v: torch.Tensor, min_response_norm: float = 1e-10,
                         min_relative_spread: float = 1e-5):
    geometry = response_geometry(v, min_response_norm, min_relative_spread)
    norms = geometry['norms']
    out = {'response_norm_mean': norms.mean().item(),
           'response_norm_min': norms.min().item(),
           'response_spread': geometry['spread'].item(),
           'response_valid_fraction': geometry['row_valid'].double().mean().item(),
           'pi_valid': float(geometry['valid'])}
    if bool(geometry['valid']):
        eig = torch.linalg.eigvalsh(geometry['gram']).clamp_min(0)
        probabilities = eig / eig.sum()
        entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
        out['response_effective_rank'] = entropy.exp().item()
        out['pi_self_similarity'] = geometry['gram'].square().sum().item()
    valid = geometry['row_valid']
    if int(valid.sum()) >= 2:
        directions = geometry['directions'][valid]
        cosine = directions @ directions.T
        m = cosine.shape[0]
        out['within_response_cosine'] = ((cosine.sum() - cosine.diag().sum())
                                        / (m * (m - 1))).item()
    return out
