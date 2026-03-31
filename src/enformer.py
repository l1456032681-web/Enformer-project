"""Enformer model utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

BASE_TO_INDEX: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}


def load_enformer(device: str = "cuda"):
    """Load the pretrained Enformer model."""
    try:
        from enformer_pytorch import from_pretrained
    except ImportError as exc:
        raise ImportError(
            "enformer-pytorch is required. Install with 'pip install enformer-pytorch'."
        ) from exc
    try:
        model = from_pretrained("EleutherAI/enformer-official-rough")
    except AttributeError as exc:
        if "all_tied_weights_keys" in str(exc):
            raise RuntimeError(
                "Enformer loading failed due to an incompatible transformers version. "
                "Install `transformers<5` in the runtime, restart, and try again."
            ) from exc
        raise
    model = model.to(device)
    model.eval()
    return model


def dna_to_tokens(seq: str) -> torch.Tensor:
    """Convert a DNA string to token indices (A,C,G,T,N -> 0..4)."""
    seq = seq.upper()
    tokens = [BASE_TO_INDEX.get(base, 4) for base in seq]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def tokens_to_onehot(tokens: torch.Tensor, num_tokens: int = 5) -> torch.Tensor:
    """One-hot encode token indices (batch, length)."""
    if tokens.dtype != torch.long:
        tokens = tokens.long()
    return torch.nn.functional.one_hot(tokens, num_classes=num_tokens).float()


def tokens_to_enformer_onehot(tokens: torch.Tensor) -> torch.Tensor:
    """One-hot encoding compatible with Enformer (4 channels, Ns as uniform)."""
    if tokens.dtype != torch.long:
        tokens = tokens.long()
    tokens = tokens.clamp(min=0)
    onehot = torch.nn.functional.one_hot(tokens, num_classes=5).float()
    onehot = onehot[..., :4]
    n_mask = tokens == 4
    if n_mask.any():
        onehot = onehot.clone()
        onehot[n_mask] = 0.25
    return onehot


def sequence_to_enformer_onehot(seq: str, device: str = "cuda") -> torch.Tensor:
    """Convert a sequence string to Enformer-compatible one-hot encoding."""
    tokens = dna_to_tokens(seq).to(device)
    return tokens_to_enformer_onehot(tokens)


def prepare_sequence_tensor(seq: str, device: str = "cuda") -> torch.Tensor:
    tokens = dna_to_tokens(seq)
    return tokens.to(device)


def predict_tracks(model, tokens: torch.Tensor) -> torch.Tensor:
    """Run Enformer and return predictions."""
    with torch.no_grad():
        preds = model(tokens)
    return preds
