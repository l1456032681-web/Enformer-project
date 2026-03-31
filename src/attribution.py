"""Attribution utilities (gradient x input)."""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch

from src.enformer import tokens_to_enformer_onehot


def select_track_bin(preds: torch.Tensor, track_index: int, bin_index: int) -> torch.Tensor:
    """Select a scalar prediction for attribution."""
    return preds[0, bin_index, track_index]


def gradient_x_input(
    inputs: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute gradient x input for a scalar forward function.

    forward_fn must return a scalar tensor.
    """
    inputs = inputs.clone().detach().requires_grad_(True)
    output = forward_fn(inputs)
    if output.numel() != 1:
        raise ValueError("forward_fn must return a scalar for attribution")
    output.backward()
    grads = inputs.grad
    if grads is None:
        raise RuntimeError("No gradients computed; check that inputs require gradients")
    return grads * inputs


def gradient_x_onehot(
    model,
    onehot: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Convenience wrapper when the model accepts one-hot inputs.

    If you are using enformer-pytorch with token inputs, adapt this to
    attribute through the embedding layer.
    """

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        preds = model(x)
        return target_fn(preds)

    return gradient_x_input(onehot, forward_fn)


def gradient_x_input_for_track(
    model,
    tokens: torch.Tensor,
    track_index: int,
    bin_index: int,
    head: str = "human",
) -> torch.Tensor:
    """Gradient x input for a single track/bin using token indices."""
    onehot = tokens_to_enformer_onehot(tokens).detach().clone().requires_grad_(True)
    preds = model(onehot, head=head)
    if preds.ndim != 3:
        raise ValueError(f"Expected predictions with shape (batch, bins, tracks), got {preds.shape}")
    target = preds[0, bin_index, track_index]
    target.backward()
    grads = onehot.grad
    if grads is None:
        raise RuntimeError("No gradients computed; check that inputs require gradients")
    return grads * onehot


def reduce_attribution(attribution: torch.Tensor, method: str = "sum") -> torch.Tensor:
    """Reduce attribution across channels to per-base scores."""
    if attribution.ndim < 2:
        return attribution
    if method == "sum":
        return attribution.sum(dim=-1)
    if method == "mean":
        return attribution.mean(dim=-1)
    raise ValueError(f"Unknown reduction method: {method}")


def call_peaks(
    scores: torch.Tensor,
    threshold_quantile: float = 0.995,
    min_width: int = 20,
) -> List[Tuple[int, int]]:
    """Call peaks from a 1D score array using a quantile threshold."""
    import numpy as np

    if isinstance(scores, torch.Tensor):
        values = scores.detach().cpu().numpy()
    else:
        values = np.asarray(scores)

    if values.ndim != 1:
        values = values.reshape(-1)

    cutoff = np.quantile(values, threshold_quantile)
    mask = values >= cutoff

    peaks: List[Tuple[int, int]] = []
    start = None
    for idx, is_on in enumerate(mask):
        if is_on and start is None:
            start = idx
        elif not is_on and start is not None:
            end = idx
            if end - start >= min_width:
                peaks.append((start, end))
            start = None
    if start is not None:
        end = len(values)
        if end - start >= min_width:
            peaks.append((start, end))

    return peaks
