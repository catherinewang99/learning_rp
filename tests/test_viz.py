"""Kernel visualization: sorting correctness and that the plots render."""

import matplotlib

matplotlib.use("Agg")
import torch

from src.data.experiences import Experience
from src.kernels.plasticity import plasticity_summary
from src.viz import plot_delta, plot_kernel, plot_plasticity, shared_vmax, sorted_bundle
from tests.test_av_smoke import make_sides


def make_bundle():
    torch.manual_seed(0)
    side = make_sides()["audio"]
    labels = torch.tensor([2, 0, 1, 0, 2, 1])  # deliberately unsorted
    xs = torch.randn(6, 1, 16, 20)
    exps = [Experience(x=xs[i : i + 1], y=labels[i : i + 1]) for i in range(6)]
    sums = plasticity_summary(side.probed, side.detached_params(), side.rule,
                              exps, xs, side.buffers)
    layer = list(sums)[0]
    return sums, labels, layer, sorted_bundle(sums, layer, labels, labels)


def test_sorted_bundle_orders_and_shapes():
    sums, labels, layer, b = make_bundle()
    assert b["probe_labels"].tolist() == sorted(labels.tolist())
    assert b["K"].shape == (6, 6) and b["Pi"].shape == (6, 6) and b["V"].shape == (6, 6, 6)
    # sorting is a symmetric permutation: diagonal multiset preserved
    assert torch.allclose(b["K"].diag().sort().values, sums[layer]["K"].diag().sort().values)
    order = torch.argsort(labels, stable=True)
    assert torch.allclose(b["Pi"], sums[layer]["Pi"][order][:, order])
    # V rows follow experience order, both probe axes follow probe order
    assert torch.allclose(b["V"][0], sums[layer]["V"][order[0]][order][:, order])


def test_plots_render():
    _, _, _, b = make_bundle()
    for fn, kwargs in ((plot_kernel, {}), (plot_plasticity, {}),
                       (plot_delta, {"experience": 0}), (plot_delta, {"experience": "mean"}),
                       (plot_delta, {"experience": "absmean"})):
        ax = fn(b, **kwargs)
        assert ax.images, "no heatmap drawn"
    assert shared_vmax([b, b], "Pi") > 0
