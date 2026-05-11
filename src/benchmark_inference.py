from __future__ import annotations

import argparse
import resource
import statistics
import struct
import sys
import time
import zlib
from pathlib import Path

import torch
try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generate import get_device, load_tokenizer
from src.model import MiniGPT


MB = 1024 * 1024
FONT_3X5: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000"),
    "%": ("101", "001", "010", "100", "101"),
    "+": ("000", "010", "111", "010", "000"),
    ",": ("000", "000", "000", "010", "100"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    "/": ("001", "001", "010", "100", "100"),
    ":": ("000", "010", "000", "010", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "001"),
    "R": ("111", "101", "111", "110", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


class Canvas:
    def rectangle(
        self,
        xy: tuple[int, int, int, int],
        fill: tuple[int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        raise NotImplementedError

    def line(
        self,
        xy: tuple[int, int, int, int] | list[tuple[int, int]],
        fill: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError


class PILCanvas(Canvas):
    def __init__(self, width: int, height: int) -> None:
        assert Image is not None
        assert ImageDraw is not None
        self.image = Image.new("RGB", (width, height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def rectangle(
        self,
        xy: tuple[int, int, int, int],
        fill: tuple[int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        self.draw.rectangle(xy, fill=fill, outline=outline, width=width)

    def line(
        self,
        xy: tuple[int, int, int, int] | list[tuple[int, int]],
        fill: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        self.draw.line(xy, fill=fill, width=width)

    def save(self, path: Path) -> None:
        self.image.save(path)


class SimpleCanvas(Canvas):
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray([255] * width * height * 3)

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        idx = (y * self.width + x) * 3
        self.pixels[idx : idx + 3] = bytes(color)

    def rectangle(
        self,
        xy: tuple[int, int, int, int],
        fill: tuple[int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        left, top, right, bottom = xy
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.width - 1, right), min(self.height - 1, bottom)
        if fill is not None:
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    self.set_pixel(x, y, fill)
        if outline is not None:
            for offset in range(width):
                self.line((left, top + offset, right, top + offset), outline, 1)
                self.line((left, bottom - offset, right, bottom - offset), outline, 1)
                self.line((left + offset, top, left + offset, bottom), outline, 1)
                self.line((right - offset, top, right - offset, bottom), outline, 1)

    def line(
        self,
        xy: tuple[int, int, int, int] | list[tuple[int, int]],
        fill: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        if isinstance(xy, list):
            for start, end in zip(xy, xy[1:]):
                self.line((start[0], start[1], end[0], end[1]), fill, width)
            return

        x0, y0, x1, y1 = xy
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            for wx in range(-(width // 2), width - width // 2):
                for wy in range(-(width // 2), width - width // 2):
                    self.set_pixel(x0 + wx, y0 + wy, fill)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def save(self, path: Path) -> None:
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * stride
            raw.extend(self.pixels[start : start + stride])
        compressed = zlib.compress(bytes(raw), level=9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(chunk(b"IHDR", ihdr))
        png.extend(chunk(b"IDAT", compressed))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(png)


def new_canvas(width: int, height: int) -> Canvas:
    if Image is not None and ImageDraw is not None:
        return PILCanvas(width, height)
    return SimpleCanvas(width, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MiniGPT prefill/decode latency and VRAM usage."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint.pt"),
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt used for the prefill pass. Ignored when --prompt-file is set.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("data/prompt.txt"),
        help="Text file used as the initial prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Number of tokens to generate during decode benchmarking.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup forward passes before timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of benchmark runs used for mean +/- std reporting.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("benchmark_reports"),
        help="Directory where PNG reports are written.",
    )
    return parser.parse_args()


def build_model(checkpoint: dict, device: torch.device) -> MiniGPT:
    config = checkpoint["config"]
    model = MiniGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        block_size=config["block_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / MB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sanitize_prompt(prompt: str, tokenizer) -> str:
    if prompt and any(ch not in tokenizer.stoi for ch in prompt):
        return tokenizer.itos[0]
    return prompt


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return " "


def memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_mb = props.total_memory / MB
        allocated_mb = torch.cuda.memory_allocated(device) / MB
        reserved_mb = torch.cuda.memory_reserved(device) / MB
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_mb = (total_bytes - free_bytes) / MB
        return {
            "total_mb": total_mb,
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "used_mb": used_mb,
            "allocated_pct": (allocated_mb / total_mb) * 100 if total_mb else 0.0,
            "reserved_pct": (reserved_mb / total_mb) * 100 if total_mb else 0.0,
            "used_pct": (used_mb / total_mb) * 100 if total_mb else 0.0,
        }

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "total_mb": rss_mb,
        "allocated_mb": rss_mb,
        "reserved_mb": rss_mb,
        "used_mb": rss_mb,
        "allocated_pct": 100.0,
        "reserved_pct": 100.0,
        "used_pct": 100.0,
    }


def record_sample(samples: list[dict[str, float | int | str]], phase: str, step: int, device: torch.device) -> None:
    snap = memory_snapshot(device)
    samples.append(
        {
            "phase": phase,
            "step": step,
            "allocated_mb": snap["allocated_mb"],
            "reserved_mb": snap["reserved_mb"],
            "used_mb": snap["used_mb"],
            "allocated_pct": snap["allocated_pct"],
            "reserved_pct": snap["reserved_pct"],
            "used_pct": snap["used_pct"],
        }
    )


def parameter_memory_bytes(model: MiniGPT) -> int:
    return sum(param.numel() * param.element_size() for param in model.parameters())


def estimate_prefill_flops(model: MiniGPT, seq_len: int) -> float:
    c = model.token_embedding.embedding_dim
    layers = len(model.blocks)
    per_layer = (24 * seq_len * c * c) + (4 * seq_len * seq_len * c)
    head_flops = 2 * seq_len * c * model.head.out_features
    return layers * per_layer + head_flops


def estimate_decode_flops_per_token(model: MiniGPT, context_len: int) -> float:
    c = model.token_embedding.embedding_dim
    layers = len(model.blocks)
    per_layer = (24 * c * c) + (4 * context_len * c)
    head_flops = 2 * c * model.head.out_features
    return layers * per_layer + head_flops


def estimate_prefill_activation_bytes(model: MiniGPT, seq_len: int, element_size: int) -> float:
    c = model.token_embedding.embedding_dim
    heads = model.blocks[0].attn.num_heads
    attn_matrix = heads * seq_len * seq_len
    activation_elems = (10 * seq_len * c) + (2 * attn_matrix)
    return activation_elems * element_size


def estimate_decode_memory_bytes_per_token(model: MiniGPT, context_len: int, element_size: int) -> float:
    c = model.token_embedding.embedding_dim
    layers = len(model.blocks)
    kv_bytes = layers * 2 * context_len * c * element_size
    activation_bytes = layers * 12 * c * element_size
    return kv_bytes + activation_bytes + parameter_memory_bytes(model)


def build_phase_proxy(model: MiniGPT, effective_prefill_tokens: int, generated_tokens: int) -> dict[str, float]:
    element_size = next(model.parameters()).element_size()
    context_for_decode = min(model.block_size, effective_prefill_tokens + max(generated_tokens // 2, 1))
    prefill_flops = estimate_prefill_flops(model, effective_prefill_tokens)
    decode_flops = estimate_decode_flops_per_token(model, context_for_decode)
    prefill_bytes = parameter_memory_bytes(model) + estimate_prefill_activation_bytes(
        model, effective_prefill_tokens, element_size
    )
    decode_bytes = estimate_decode_memory_bytes_per_token(model, context_for_decode, element_size)
    prefill_intensity = prefill_flops / prefill_bytes if prefill_bytes else 0.0
    decode_intensity = decode_flops / decode_bytes if decode_bytes else 0.0
    return {
        "prefill_flops_g": prefill_flops / 1e9,
        "decode_flops_g": decode_flops / 1e9,
        "prefill_intensity": prefill_intensity,
        "decode_intensity": decode_intensity,
        "decode_context_tokens": float(context_for_decode),
    }


def benchmark(
    model: MiniGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    warmup_iters: int,
    device: torch.device,
) -> dict[str, float | int | str | list[dict[str, float | int | str]]]:
    prompt = sanitize_prompt(prompt, tokenizer)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    samples: list[dict[str, float | int | str]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    record_sample(samples, "BASELINE", 0, device)

    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(input_ids[:, -model.block_size:])

        synchronize(device)
        record_sample(samples, "READY", 0, device)

        prefill_start = time.perf_counter()
        logits, _ = model(input_ids[:, -model.block_size:])
        synchronize(device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000
        record_sample(samples, "PREFILL", 0, device)

        generated = input_ids
        decode_step_ms: list[float] = []
        decode_start = time.perf_counter()
        for step in range(max_new_tokens):
            step_start = time.perf_counter()
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if step + 1 < max_new_tokens:
                logits, _ = model(generated[:, -model.block_size:])
            synchronize(device)
            decode_step_ms.append((time.perf_counter() - step_start) * 1000)
            record_sample(samples, "DECODE", step + 1, device)
        decode_total_ms = (time.perf_counter() - decode_start) * 1000

    decode_ms_per_token = decode_total_ms / max_new_tokens if max_new_tokens else 0.0
    tokens_per_sec = (max_new_tokens * 1000 / decode_total_ms) if decode_total_ms else 0.0
    effective_prefill_tokens = int(min(input_ids.shape[1], model.block_size))
    phase_proxy = build_phase_proxy(model, effective_prefill_tokens, max_new_tokens)

    return {
        "device": str(device),
        "prompt_tokens": int(input_ids.shape[1]),
        "effective_prefill_tokens": effective_prefill_tokens,
        "generated_tokens": max_new_tokens,
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_total_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "decode_step_ms_mean": statistics.mean(decode_step_ms) if decode_step_ms else 0.0,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_memory_mb(device),
        "memory_samples": samples,
        "prefill_flops_g": phase_proxy["prefill_flops_g"],
        "decode_flops_g": phase_proxy["decode_flops_g"],
        "prefill_intensity": phase_proxy["prefill_intensity"],
        "decode_intensity": phase_proxy["decode_intensity"],
        "decode_context_tokens": phase_proxy["decode_context_tokens"],
    }


def summarize_runs(runs: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    summary = dict(runs[0])
    metric_keys = (
        "prefill_ms",
        "decode_total_ms",
        "decode_ms_per_token",
        "decode_step_ms_mean",
        "tokens_per_sec",
        "peak_memory_mb",
        "prefill_flops_g",
        "decode_flops_g",
        "prefill_intensity",
        "decode_intensity",
    )

    for key in metric_keys:
        values = [float(run[key]) for run in runs]
        summary[f"{key}_mean"] = statistics.mean(values)
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

    summary["repeats"] = len(runs)
    return summary


def aggregate_memory_samples(
    runs: list[dict[str, float | int | str | list[dict[str, float | int | str]]]]
) -> list[dict[str, float | int | str]]:
    first = runs[0]["memory_samples"]
    assert isinstance(first, list)
    aggregated: list[dict[str, float | int | str]] = []
    for idx, base_sample in enumerate(first):
        phase = str(base_sample["phase"])
        step = int(base_sample["step"])
        aggregated.append(
            {
                "phase": phase,
                "step": step,
                "allocated_pct": statistics.mean(
                    float(run["memory_samples"][idx]["allocated_pct"]) for run in runs
                ),
                "reserved_pct": statistics.mean(
                    float(run["memory_samples"][idx]["reserved_pct"]) for run in runs
                ),
                "used_pct": statistics.mean(
                    float(run["memory_samples"][idx]["used_pct"]) for run in runs
                ),
                "allocated_mb": statistics.mean(
                    float(run["memory_samples"][idx]["allocated_mb"]) for run in runs
                ),
                "reserved_mb": statistics.mean(
                    float(run["memory_samples"][idx]["reserved_mb"]) for run in runs
                ),
                "used_mb": statistics.mean(
                    float(run["memory_samples"][idx]["used_mb"]) for run in runs
                ),
            }
        )
    return aggregated


def draw_text(draw: Canvas, x: int, y: int, text: str, fill: tuple[int, int, int], scale: int = 2) -> None:
    cursor_x = x
    for raw_char in text.upper():
        glyph = FONT_3X5.get(raw_char, FONT_3X5[" "])
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit == "1":
                    draw.rectangle(
                        (
                            cursor_x + col * scale,
                            y + row * scale,
                            cursor_x + (col + 1) * scale - 1,
                            y + (row + 1) * scale - 1,
                        ),
                        fill=fill,
                    )
        cursor_x += (len(glyph[0]) + 1) * scale


def plot_line(
    draw: Canvas,
    area: tuple[int, int, int, int],
    values_a: list[float],
    values_b: list[float],
    label: str,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    legend_a: str,
    legend_b: str,
) -> None:
    left, top, right, bottom = area
    draw.rectangle(area, outline=(210, 210, 210), width=1)
    draw.line((left, bottom, right, bottom), fill=(180, 180, 180), width=1)
    draw.line((left, top, left, bottom), fill=(180, 180, 180), width=1)
    draw_text(draw, left, top - 18, label, (40, 40, 40))

    all_values = values_a + values_b
    if not all_values:
        return
    y_max = max(max(all_values), 1.0)
    count = max(len(values_a), len(values_b))

    def to_xy(index: int, value: float) -> tuple[int, int]:
        x = left + int((right - left) * index / max(count - 1, 1))
        normalized = value / y_max
        y = bottom - int((bottom - top) * normalized)
        return x, y

    for tick in range(5):
        y = bottom - int((bottom - top) * tick / 4)
        draw.line((left, y, right, y), fill=(240, 240, 240), width=1)
        value = y_max * tick / 4
        draw_text(draw, max(0, left - 34), y - 5, f"{value:.0f}", (120, 120, 120))

    if len(values_a) > 1:
        draw.line([to_xy(i, value) for i, value in enumerate(values_a)], fill=color_a, width=3)
    if len(values_b) > 1:
        draw.line([to_xy(i, value) for i, value in enumerate(values_b)], fill=color_b, width=3)

    legend_y = top + 8
    draw.rectangle((right - 150, legend_y, right - 140, legend_y + 10), fill=color_a)
    draw_text(draw, right - 135, legend_y, legend_a, (40, 40, 40))
    draw.rectangle((right - 70, legend_y, right - 60, legend_y + 10), fill=color_b)
    draw_text(draw, right - 55, legend_y, legend_b, (40, 40, 40))


def plot_bars(
    draw: Canvas,
    area: tuple[int, int, int, int],
    values: list[float],
    labels: list[str],
    title: str,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = area
    draw.rectangle(area, outline=(210, 210, 210), width=1)
    draw_text(draw, left, top - 18, title, (40, 40, 40))
    y_max = max(max(values), 1.0)
    width = right - left
    bar_width = max(width // max(len(values) * 2, 2), 20)

    for tick in range(5):
        y = bottom - int((bottom - top) * tick / 4)
        draw.line((left, y, right, y), fill=(240, 240, 240), width=1)
        value = y_max * tick / 4
        draw_text(draw, max(0, left - 34), y - 5, f"{value:.1f}", (120, 120, 120))

    for idx, (value, label) in enumerate(zip(values, labels)):
        center_x = left + int((idx + 0.5) * width / len(values))
        bar_height = int((bottom - top) * (value / y_max))
        draw.rectangle(
            (center_x - bar_width // 2, bottom - bar_height, center_x + bar_width // 2, bottom),
            fill=color,
        )
        draw_text(draw, center_x - 18, bottom + 8, label, (40, 40, 40))


def save_vram_plot(
    output_path: Path,
    samples: list[dict[str, float | int | str]],
    summary: dict[str, float | int | str],
) -> None:
    draw = new_canvas(1280, 900)
    draw_text(draw, 40, 24, "PREFILL/DECODE VRAM PROFILE", (30, 30, 30), scale=3)
    draw_text(
        draw,
        40,
        60,
        f"DEVICE {summary['device']}  PROMPT {summary['effective_prefill_tokens']} TOKENS  DECODE {summary['generated_tokens']} TOKENS",
        (80, 80, 80),
    )

    decode_samples = [sample for sample in samples if sample["phase"] == "DECODE"]
    ready_samples = [sample for sample in samples if sample["phase"] in {"READY", "PREFILL"}]
    phase_values_alloc = [float(sample["allocated_pct"]) for sample in ready_samples]
    phase_values_used = [float(sample["used_pct"]) for sample in ready_samples]
    phase_labels = ["READY", "PREFILL"]

    plot_bars(
        draw,
        (80, 140, 560, 420),
        phase_values_alloc,
        phase_labels,
        "ALLOCATED VRAM % BY PHASE",
        (71, 133, 255),
    )
    plot_bars(
        draw,
        (700, 140, 1180, 420),
        phase_values_used,
        phase_labels,
        "USED VRAM % BY PHASE",
        (255, 140, 66),
    )

    plot_line(
        draw,
        (80, 520, 1180, 820),
        [float(sample["allocated_pct"]) for sample in decode_samples],
        [float(sample["reserved_pct"]) for sample in decode_samples],
        "DECODE STEP VRAM %",
        (71, 133, 255),
        (235, 87, 87),
        "ALLOC",
        "RESV",
    )
    draw_text(
        draw,
        80,
        840,
        f"PEAK_MB {float(summary['peak_memory_mb_mean']):.1f}  ALLOC LINE BLUE  RESERVED LINE RED",
        (80, 80, 80),
    )
    draw.save(output_path)


def save_bound_plot(output_path: Path, summary: dict[str, float | int | str]) -> None:
    draw = new_canvas(1280, 760)
    draw_text(draw, 40, 24, "PREFILL VS DECODE PROXY", (30, 30, 30), scale=3)

    plot_bars(
        draw,
        (80, 140, 560, 520),
        [
            float(summary["prefill_ms_mean"]),
            float(summary["decode_ms_per_token_mean"]),
        ],
        ["PREFILL", "DECODE"],
        "LATENCY MS",
        (88, 196, 120),
    )
    plot_bars(
        draw,
        (700, 140, 1180, 520),
        [
            float(summary["prefill_intensity_mean"]),
            float(summary["decode_intensity_mean"]),
        ],
        ["PREFILL", "DECODE"],
        "FLOP/BYTE PROXY",
        (155, 89, 182),
    )

    classification = "PREFILL COMPUTE-LEANING / DECODE MEMORY-LEANING"
    if float(summary["prefill_intensity_mean"]) <= float(summary["decode_intensity_mean"]):
        classification = "NO CLEAR SPLIT IN THIS RUN"

    draw_text(
        draw,
        80,
        590,
        f"PREFILL FLOPS {float(summary['prefill_flops_g_mean']):.2f}G",
        (60, 60, 60),
    )
    draw_text(
        draw,
        80,
        620,
        f"DECODE FLOPS/TOKEN {float(summary['decode_flops_g_mean']):.2f}G",
        (60, 60, 60),
    )
    draw_text(
        draw,
        80,
        650,
        f"CLASS {classification}",
        (60, 60, 60),
    )
    draw_text(
        draw,
        80,
        680,
        f"DECODE CONTEXT PROXY {float(summary['decode_context_tokens']):.0f} TOKENS",
        (60, 60, 60),
    )
    draw.save(output_path)


def write_reports(
    report_dir: Path,
    samples: list[dict[str, float | int | str]],
    summary: dict[str, float | int | str],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    vram_path = report_dir / "prefill_decode_vram.png"
    bound_path = report_dir / "prefill_decode_bound.png"
    save_vram_plot(vram_path, samples, summary)
    save_bound_plot(bound_path, summary)
    return vram_path, bound_path


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = load_tokenizer(checkpoint["tokenizer_state"])
    prompt = resolve_prompt(args)

    runs = []
    for _ in range(args.repeats):
        model = build_model(checkpoint, device)
        runs.append(
            benchmark(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                warmup_iters=args.warmup_iters,
                device=device,
            )
        )

    metrics = summarize_runs(runs)
    sample_summary = aggregate_memory_samples(runs)
    vram_path, bound_path = write_reports(args.report_dir, sample_summary, metrics)

    print(f"Device: {metrics['device']}")
    print(f"Prompt tokens: {metrics['prompt_tokens']}")
    print(f"Effective prefill tokens: {metrics['effective_prefill_tokens']}")
    print(f"Generated tokens: {metrics['generated_tokens']}")
    print(f"Repeats: {metrics['repeats']}")
    print(
        f"Prefill ms: {metrics['prefill_ms_mean']:.3f} +/- {metrics['prefill_ms_std']:.3f}"
    )
    print(
        "Decode total ms: "
        f"{metrics['decode_total_ms_mean']:.3f} +/- {metrics['decode_total_ms_std']:.3f}"
    )
    print(
        "Decode ms/token: "
        f"{metrics['decode_ms_per_token_mean']:.3f} +/- {metrics['decode_ms_per_token_std']:.3f}"
    )
    print(
        f"Tokens/sec: {metrics['tokens_per_sec_mean']:.3f} +/- {metrics['tokens_per_sec_std']:.3f}"
    )
    label = "Peak GPU MB" if device.type == "cuda" else "Peak CPU MB"
    print(
        f"{label}: {metrics['peak_memory_mb_mean']:.3f} +/- {metrics['peak_memory_mb_std']:.3f}"
    )
    print(
        "Intensity proxy (higher => more compute-leaning): "
        f"prefill={metrics['prefill_intensity_mean']:.3f}, "
        f"decode={metrics['decode_intensity_mean']:.3f}"
    )
    print(f"VRAM report: {vram_path}")
    print(f"Bound report: {bound_path}")


if __name__ == "__main__":
    main()
