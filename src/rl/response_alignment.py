from __future__ import annotations

import math

import torch

from ..kernels.response_geometry import geometry_diagnostics, gram_cka, response_scores
from ..losses.layerwise import layer_supervision


class ResponseAlignmentLoss:
    needs = {'K', 'V', 'Pi'}

    def __init__(self, objective: str = 'pi', min_response_norm: float = 1e-10,
                 min_relative_spread: float = 1e-5, pi_weight: float = 1.0,
                 v_weight: float = 0.1, upper_half: bool = False):
        if objective not in {'pi', 'v', 'pi_v'}:
            raise ValueError('response_objective must be pi, v, or pi_v.')
        if (not math.isfinite(min_response_norm) or min_response_norm <= 0
                or not 0 < min_relative_spread < 1):
            raise ValueError('Positive norm threshold and relative spread in (0,1) required.')
        if not all(math.isfinite(w) and w > 0 for w in (pi_weight, v_weight)):
            raise ValueError('Response objective weights must be positive and finite.')
        self.objective, self.upper_half = objective, upper_half
        self.min_response_norm, self.min_relative_spread = min_response_norm, min_relative_spread
        self.pi_weight, self.v_weight = pi_weight, v_weight

    def __call__(self, target_summaries, guide_summaries):
        mapping = layer_supervision(list(guide_summaries), list(target_summaries), self.upper_half)
        pi_losses, v_losses, parts = [], [], {}
        for g, t in mapping.items():
            scores = response_scores(target_summaries[t]['V'], guide_summaries[g]['V'],
                                     self.min_response_norm, self.min_relative_spread)
            prefix = f'{g}->{t}'
            parts[f'{prefix}/pi_valid'] = float(scores['pi_valid'])
            parts[f'{prefix}/v_valid_fraction'] = scores['v_valid_fraction']
            parts[f'{prefix}/target_response_spread'] = scores['target']['spread'].detach()
            parts[f'{prefix}/guide_response_spread'] = scores['guide']['spread'].detach()
            if scores['pi_valid']:
                loss = 1 - scores['pi_similarity']
                pi_losses.append(loss)
                parts[f'{prefix}/pi_cka_similarity'] = scores['pi_similarity'].detach()
                parts[f'{prefix}/pi_cka_loss'] = loss.detach()
            if 'v_cosine' in scores:
                loss = 1 - scores['v_cosine']  # Half the squared distance of unit rows.
                v_losses.append(loss)
                parts[f'{prefix}/v_cosine'] = scores['v_cosine'].detach()
                parts[f'{prefix}/v_match_loss'] = loss.detach()
        parts['pi_valid_pair_fraction'] = len(pi_losses) / len(mapping)
        parts['v_valid_pair_fraction'] = len(v_losses) / len(mapping)
        parts['n_pairs'] = len(mapping)
        total, active = None, 0
        if self.objective in {'pi', 'pi_v'} and pi_losses:
            total = self.pi_weight * torch.stack(pi_losses).mean()
            active += 1
        if self.objective in {'v', 'pi_v'} and v_losses:
            weight = 1.0 if self.objective == 'v' else self.v_weight
            component = weight * torch.stack(v_losses).mean()
            total = component if total is None else total + component
            active += 1
        parts['active_terms'] = active
        # No valid objective is not a successful zero loss. Trainer skips alignment.
        return total, parts

    @torch.no_grad()
    def evaluation(self, guide_summaries, target_summaries, guide_name='vision',
                   target_name='audio'):
        mapping = layer_supervision(list(guide_summaries), list(target_summaries), self.upper_half)
        values = {'k_cka': [], 'pi_cka': [], 'v_cosine': []}
        out = {}
        for g, t in mapping.items():
            pair = f'{g}->{t}'
            guide, target = guide_summaries[g], target_summaries[t]
            score_k = gram_cka(guide['K'], target['K'])
            if score_k is not None:
                out[f'k_cka/{pair}'] = score_k.item()
                values['k_cka'].append(score_k.item())
            scores = response_scores(target['V'], guide['V'],
                                     self.min_response_norm, self.min_relative_spread)
            out[f'pi_valid/{pair}'] = float(scores['pi_valid'])
            out[f'v_valid_fraction/{pair}'] = scores['v_valid_fraction']
            if scores['pi_valid']:
                sim = scores['pi_similarity'].item()
                out[f'pi_cka/{pair}'] = sim
                values['pi_cka'].append(sim)
                # Fixed circular permutation: diagnostic without consuming RNG.
                gt = scores['guide']['gram'].roll(1, 0).roll(1, 1)
                shuffled = (scores['target']['gram'] * gt).sum().clamp(0, 1).item()
                out[f'pi_cka_shuffled/{pair}'] = shuffled
                out[f'pi_cka_gap/{pair}'] = sim - shuffled
            if 'v_cosine' in scores:
                sim = scores['v_cosine'].item()
                out[f'v_cosine/{pair}'] = sim
                values['v_cosine'].append(sim)
        for metric, numbers in values.items():
            out[f'{metric}/valid_pair_fraction'] = len(numbers) / len(mapping)
            if numbers:
                out[f'{metric}/mean'] = sum(numbers) / len(numbers)
        for side, summaries in ((guide_name, guide_summaries), (target_name, target_summaries)):
            for layer, summary in summaries.items():
                diagnostics = geometry_diagnostics(summary['V'], self.min_response_norm,
                                                   self.min_relative_spread)
                out.update({f'{side}/response/{layer}/{key}': value
                            for key, value in diagnostics.items()})
        return out
