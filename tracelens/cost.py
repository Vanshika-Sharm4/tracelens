"""Rough FLOP and memory-traffic estimates for PyTorch operators, plus hardware profiles.

These are *estimates* used to classify operators as memory-bound or compute-bound with a roofline argument:

    arithmetic intensity (AI) = FLOPs / bytes moved,   ridge point = peak FLOP/s / peak bytes/s

An operator whose AI is below the ridge point cannot reach peak FLOP/s no matter how well it is written. Bytes are
estimated from input shapes plus an assumed output size, ignoring caches, so the numbers are indicative, not exact.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

DTYPE_BYTES = {
    "float": 4,
    "double": 8,
    "c10::Half": 2,
    "c10::BFloat16": 2,
    "long": 8,
    "int": 4,
    "short": 2,
    "bool": 1,
    "signed char": 1,
    "unsigned char": 1,
    "c10::complex<float>": 8,
    "c10::complex<double>": 16,
}

# Leaf compute operators we can cost. Wrapper ops (aten::linear, aten::matmul) are skipped on purpose: their cost is
# already counted by the leaf op nested inside them, and counting both would double-count.
_MATMUL_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"}
_CONV_OPS = {"aten::conv1d", "aten::conv2d", "aten::conv3d"}

# Operators that are bandwidth-limited by nature (about one FLOP per element touched).
MEMORY_BOUND_OPS = (
    "aten::add", "aten::sub", "aten::mul", "aten::div", "aten::neg", "aten::exp", "aten::log", "aten::sqrt",
    "aten::rsqrt", "aten::pow", "aten::tanh", "aten::sigmoid", "aten::relu", "aten::gelu", "aten::silu",
    "aten::clamp", "aten::softmax", "aten::_softmax", "aten::log_softmax", "aten::layer_norm",
    "aten::native_layer_norm", "aten::batch_norm", "aten::native_batch_norm", "aten::dropout",
    "aten::native_dropout", "aten::copy_", "aten::clone", "aten::contiguous", "aten::to", "aten::_to_copy",
    "aten::cat", "aten::masked_fill", "aten::where", "aten::embedding", "aten::index_select", "aten::mean",
    "aten::sum", "aten::max_pool2d", "aten::adaptive_avg_pool2d", "aten::tril", "aten::triu",
)


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_tflops: float  # dense fp16/fp32 matmul peak the roofline should use
    peak_gbps: float

    @property
    def ridge(self) -> float:
        """FLOPs per byte at which the roofline turns from sloped (memory) to flat (compute)."""
        return self.peak_tflops * 1e12 / (self.peak_gbps * 1e9)


# Vendor datasheet figures (fp16 tensor-core peak, DRAM bandwidth). Override with a custom Hardware if yours differs.
HARDWARE = {
    "t4": Hardware("NVIDIA T4", 65.0, 320.0),
    "l4": Hardware("NVIDIA L4", 121.0, 300.0),
    "a100": Hardware("NVIDIA A100", 312.0, 1555.0),
    "h100": Hardware("NVIDIA H100", 989.0, 3350.0),
    "generic-gpu": Hardware("Generic GPU", 100.0, 1000.0),
    "generic-cpu": Hardware("Generic CPU", 1.0, 100.0),
}


def detect_hardware(device_name: str | None, has_gpu: bool) -> Hardware:
    """Pick a hardware profile from the trace's device name, falling back to a generic profile."""
    if device_name:
        upper = device_name.upper()
        for key in ("T4", "L4", "A100", "H100"):
            if key in upper:
                return HARDWARE[key.lower()]
    return HARDWARE["generic-gpu" if has_gpu else "generic-cpu"]


def numel(dims: list[int]) -> int:
    return int(math.prod(dims)) if dims else 0


def _elt_bytes(types: list | None, index: int) -> int:
    if types and index < len(types):
        return DTYPE_BYTES.get(types[index], 4)
    return 4


def _tensor_inputs(dims: list | None, types: list | None) -> list[tuple[list[int], int]]:
    """Non-empty tensor inputs as ``(shape, bytes_per_element)``."""
    out = []
    for i, d in enumerate(dims or []):
        if isinstance(d, list) and d and all(isinstance(x, int) for x in d):
            out.append((d, _elt_bytes(types, i)))
    return out


def _parse_int_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(x, int) for x in value):
        return list(value)
    return None


def estimate_cost(name: str, dims: list | None, types: list | None, concrete: list | None) -> tuple[float | None, float | None]:
    """Return ``(flops, bytes)`` for one operator call, or ``(None, None)`` when it cannot be estimated."""
    tensors = _tensor_inputs(dims, types)
    if not tensors:
        return None, None

    try:
        if name in _MATMUL_OPS:
            # mm: [M,K],[K,N]   addmm: [bias],[M,K],[K,N]   bmm: [B,M,K],[B,K,N]   baddbmm: [bias],[B,M,K],[B,K,N]
            mats = tensors[1:] if name in ("aten::addmm", "aten::baddbmm") else tensors
            (a, ea), (b, eb) = mats[0], mats[1]
            batch = numel(a[:-2])
            m, k, n = a[-2], a[-1], b[-1]
            flops = 2.0 * max(batch, 1) * m * n * k
            nbytes = max(batch, 1) * (m * k * ea + k * n * eb + m * n * ea)
            return flops, float(nbytes)

        if name in _CONV_OPS:
            x, w = tensors[0], tensors[1]
            stride = _parse_int_list((concrete or [None] * 4)[3] if concrete and len(concrete) > 3 else None)
            padding = _parse_int_list(concrete[4] if concrete and len(concrete) > 4 else None)
            dilation = _parse_int_list(concrete[5] if concrete and len(concrete) > 5 else None)
            groups = _parse_int_list(concrete[6] if concrete and len(concrete) > 6 else None) or [1]
            spatial = len(x[0]) - 2
            if stride is None or padding is None:
                return None, None
            stride = (stride * spatial)[:spatial] if len(stride) == 1 else stride
            padding = (padding * spatial)[:spatial] if len(padding) == 1 else padding
            dilation = ((dilation or [1]) * spatial)[:spatial] if not dilation or len(dilation) == 1 else dilation
            batch_n, c_in = x[0][0], x[0][1]
            c_out, kernel = w[0][0], w[0][2:]
            out_spatial = [
                (x[0][2 + i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1 for i in range(spatial)
            ]
            out_elems = batch_n * c_out * math.prod(out_spatial)
            flops = 2.0 * out_elems * (c_in // groups[0]) * math.prod(kernel)
            nbytes = (numel(x[0]) + numel(w[0]) + out_elems) * x[1]
            return flops, float(nbytes)
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return None, None

    if name in MEMORY_BOUND_OPS:
        # About one FLOP per element; traffic is every input read once plus an output the size of the largest input.
        biggest = max(numel(d) * e for d, e in tensors)
        nbytes = sum(numel(d) * e for d, e in tensors) + biggest
        flops = float(max(numel(d) for d, _ in tensors))
        return flops, float(nbytes)

    return None, None


def classify(name: str, flops: float | None, nbytes: float | None, hw: Hardware) -> str:
    """``compute-bound``, ``memory-bound`` or ``unknown`` for one operator."""
    if flops and nbytes:
        return "compute-bound" if flops / nbytes >= hw.ridge else "memory-bound"
    if name in MEMORY_BOUND_OPS:
        return "memory-bound"
    return "unknown"


def parse_hardware(spec: str | None) -> Hardware | None:
    """``None`` -> auto-detect; ``"t4"`` etc. -> a known profile; ``"custom:TFLOPS:GBPS"`` -> your own peaks."""
    if not spec or spec == "auto":
        return None
    key = spec.lower()
    if key in HARDWARE:
        return HARDWARE[key]
    if key.startswith("custom:"):
        try:
            _, tflops, gbps = key.split(":")
            return Hardware("Custom", float(tflops), float(gbps))
        except ValueError:
            pass
    raise ValueError(f"unknown hardware {spec!r}; use one of {', '.join(HARDWARE)} or custom:TFLOPS:GBPS")
