"""Small demo workloads (random weights, no downloads) used to try TraceLens end to end.

Two of them are deliberately built to contain a known bottleneck, which makes them handy for validating the detectors on
real hardware:

``tiny_ops``    hundreds of tiny elementwise operations: launch / dispatch overhead dominates
``sync_heavy``  calls ``.item()`` between operations: forces host-device synchronization, so a GPU sits idle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Workload:
    name: str
    description: str
    fn: Callable[[], object]


class _TinyGPTBlock(nn.Module):
    """Pre-norm transformer block with attention written out as separate ops (so it shows up as many operators)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.d = heads, d
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / (d // self.heads) ** 0.5
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        x = x + self.proj((probs @ v).transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 2000, d: int = 256, heads: int = 4, layers: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(_TinyGPTBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


class _SyncHeavy(nn.Module):
    """Linear layers with a ``.item()`` between each one, which blocks the host until the GPU catches up."""

    def __init__(self, d: int = 512, layers: int = 12):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
            _ = x.mean().item()  # host-device synchronization
        return x


class _TinyOps(nn.Module):
    """A long chain of tiny elementwise operations on a small tensor."""

    def __init__(self, n: int = 150):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n):
            x = (x * 1.0001 + 0.001) if i % 2 == 0 else torch.tanh(x)
        return x


DEMO_MODELS = {
    "mlp": "Two-layer MLP (1024-4096-1024): a few large matmuls plus GELU",
    "cnn": "Small convolutional network on 64x64 images",
    "tiny_gpt": "4-layer decoder-only transformer with explicit attention ops",
    "transformer_encoder": "torch.nn.TransformerEncoder, 4 layers, d_model 256",
    "tiny_ops": "Seeded bottleneck: ~150 tiny elementwise ops (launch/dispatch overhead)",
    "sync_heavy": "Seeded bottleneck: .item() between layers (host-device synchronization)",
}


def build_workload(name: str, batch_size: int = 8, device: str = "cpu") -> Workload:
    """Build one of :data:`DEMO_MODELS` with deterministic random weights and inputs on ``device``."""
    if name not in DEMO_MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(DEMO_MODELS)}")
    torch.manual_seed(0)
    dev = torch.device(device)

    if name == "mlp":
        model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
        x = torch.randn(batch_size * 16, 1024, device=dev)
    elif name == "cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
        )
        x = torch.randn(batch_size, 3, 64, 64, device=dev)
    elif name == "tiny_gpt":
        model = TinyGPT()
        x = torch.randint(0, 2000, (batch_size, 128), device=dev)
    elif name == "transformer_encoder":
        layer = nn.TransformerEncoderLayer(256, 4, dim_feedforward=1024, batch_first=True)
        model = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)
        x = torch.randn(batch_size, 128, 256, device=dev)
    elif name == "tiny_ops":
        model, x = _TinyOps(), torch.randn(64, 64, device=dev)
    else:  # sync_heavy
        model, x = _SyncHeavy(), torch.randn(batch_size, 512, device=dev)

    model = model.to(dev).eval()

    def run() -> object:
        with torch.no_grad():
            return model(x)

    return Workload(name=name, description=DEMO_MODELS[name], fn=run)
