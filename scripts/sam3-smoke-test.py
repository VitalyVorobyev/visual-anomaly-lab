#!/usr/bin/env -S uv run --project backend --extra dl python
"""Does SAM 3 run on this Mac's MPS, and is a second phrase on one image cheap?

ADR-0029's rule: a standalone check before wrapper code trusts a new library on the
accelerator. Explore's Text mode (`explore/text.py`) loads transformers' `Sam3Model` in the
resident worker and caches one image's vision features across phrases; this script checks
the three things that design leans on, cheapest first, on a synthetic image:

    1. is MPS built and available, with no CPU fallback enabled for missing kernels
    2. does a full forward pass run on MPS, and does it find the discs drawn in the image
    3. do the MPS masks match the CPU masks
    4. what does the first phrase cost, and what does a second phrase on the cached
       vision features cost

The weights are the catalogued, gated `facebook/sam3` revision. They are never downloaded
here: pass `--model-dir` (a directory holding the pinned files, such as the app's
`<data dir>/model-cache/assets/sam3`), or the script reads the pinned revision from the
default Hugging Face cache offline.

    ./scripts/sam3-smoke-test.py [--model-dir DIR]

Exit code 0 means MPS is usable for Text mode. A non-zero exit is a finding.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RULE = "-" * 72
REPOSITORY = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
PHRASE = "circle"
SECOND_PHRASE = "square"
DISCS = ((64, 64, 36), (190, 80, 28), (120, 180, 44))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _run(name: str, body: Callable[[], str]) -> Check:
    try:
        return Check(name=name, ok=True, detail=body())
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        traceback.print_exc(file=sys.stderr)
        return Check(name=name, ok=False, detail=f"{type(exc).__name__}: {first_line}")


def _image() -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), (228, 224, 214))
    draw = ImageDraw.Draw(image)
    for x, y, radius in DISCS:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(40, 70, 150))
    draw.rectangle((200, 190, 240, 230), fill=(160, 50, 40))
    return image


def _model_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            REPOSITORY,
            revision=REVISION,
            local_files_only=True,
            allow_patterns=["*.json", "*.txt", "model.safetensors"],
        )
    )


class Probe:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.image = _image()
        self.masks: dict[str, Any] = {}

    def load(self, device: str) -> tuple[Any, Any]:
        from transformers import Sam3Model, Sam3Processor

        processor = Sam3Processor.from_pretrained(self.directory, local_files_only=True)
        model = Sam3Model.from_pretrained(self.directory, local_files_only=True)
        return processor, model.to(device).eval()

    def run(self, device: str, phrase: str, processor: Any, model: Any) -> Any:
        import torch

        inputs = processor(images=self.image, text=phrase, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        return processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]


def check_availability() -> str:
    import torch

    if not torch.backends.mps.is_built():
        raise RuntimeError("this torch build has no MPS backend compiled in")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is built but not available on this machine")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK is set; unset it so a gap raises")
    return f"torch {torch.__version__}, macOS {platform.mac_ver()[0]}"


def check_forward(probe: Probe) -> str:
    processor, model = probe.load("mps")
    result = probe.run("mps", PHRASE, processor, model)
    probe.masks["mps"] = result["masks"].cpu()
    count = int(result["masks"].shape[0])
    if count != len(DISCS):
        raise RuntimeError(f"'{PHRASE}' found {count} instances; {len(DISCS)} discs are drawn")
    scores = ", ".join(f"{float(score):.2f}" for score in result["scores"])
    return f"'{PHRASE}' -> {count} instances ({scores})"


def check_parity(probe: Probe) -> str:
    processor, model = probe.load("cpu")
    result = probe.run("cpu", PHRASE, processor, model)
    cpu, mps = result["masks"].cpu(), probe.masks.get("mps")
    if mps is None or cpu.shape != mps.shape:
        raise RuntimeError(
            f"CPU masks {tuple(cpu.shape)} against MPS {getattr(mps, 'shape', None)}"
        )
    # Instances are matched by best IoU, since the two devices may rank near-ties apart.
    worst = 1.0
    for mask in mps.bool():
        ious = [
            float((mask & other).sum()) / max(float((mask | other).sum()), 1.0)
            for other in cpu.bool()
        ]
        worst = min(worst, max(ious))
    if worst < 0.99:
        raise RuntimeError(f"an MPS instance matches its CPU twin at IoU {worst:.3f}")
    return f"every instance matches its CPU twin at IoU >= {worst:.3f}"


def check_cached_phrase(probe: Probe) -> str:
    import torch

    processor, model = probe.load("mps")

    def timed(body: Callable[[], Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        value = body()
        torch.mps.synchronize()
        return value, (time.perf_counter() - started) * 1000.0

    with torch.inference_mode():
        image_inputs = processor(images=probe.image, return_tensors="pt").to("mps")
        vision, encode_ms = timed(
            lambda: model.get_vision_features(pixel_values=image_inputs["pixel_values"])
        )
        timings = []
        for phrase in (PHRASE, SECOND_PHRASE, PHRASE):
            text = processor(text=phrase, return_tensors="pt").to("mps")
            _, elapsed = timed(
                lambda text=text: model(
                    vision_embeds=vision,
                    input_ids=text["input_ids"],
                    attention_mask=text["attention_mask"],
                )
            )
            timings.append(elapsed)
    return f"vision encoder {encode_ms:.0f} ms once; then per phrase " + ", ".join(
        f"{value:.0f} ms" for value in timings
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model-dir", help="a directory holding the pinned SAM 3 files")
    args = parser.parse_args()

    print(RULE)
    print(f"SAM 3 smoke test · {REPOSITORY}@{REVISION[:12]}")
    print(RULE)
    availability = _run("MPS available", check_availability)
    checks = [availability]
    if availability.ok:
        try:
            probe = Probe(_model_dir(args.model_dir))
        except Exception as exc:
            checks.append(Check("weights", False, f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(_run("forward on MPS", lambda: check_forward(probe)))
            checks.append(_run("MPS matches CPU", lambda: check_parity(probe)))
            checks.append(_run("cached vision features", lambda: check_cached_phrase(probe)))
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL':>4}] {check.name}: {check.detail}")
    print(RULE)
    failed = [check for check in checks if not check.ok]
    print("MPS is usable for Text mode." if not failed else f"{len(failed)} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
