"""A command line for the jobs that have no button yet.

Teacher distillation produces an *asset*, not an experiment, and there is no screen for it —
so this is how it is started. It runs the same handler the queue would, with the same
`JobContext`, and writes the same JSON-lines event stream to stdout, which is what the job
system tees into a log file (**ADR-0009**). Redirecting stdout to a file therefore gives a
log in exactly the format the rest of the application produces.

Deliberately **not** a second way to do things that already have one. Importing, training and
evaluating are reachable from the application, and a CLI that duplicated them would be a
second surface to keep correct.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext


def _distill(args: argparse.Namespace) -> dict[str, Any]:
    from anomaly_lab.models.distill import run_distill_job

    config: dict[str, Any] = {
        "name": args.name,
        "model_size": args.model_size,
        "corpus": args.corpus,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "normalization_images": args.normalization_images,
        "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
    }
    if args.corpus_path:
        config["corpus_path"] = args.corpus_path

    return run_distill_job(
        JobContext(
            job_id=args.job_id,
            kind=JobKind.DISTILL,
            params={"config": config, "resume": not args.no_resume},
            settings=Settings(),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomaly-lab", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    distill = subparsers.add_parser(
        "distill",
        help="Distil a frozen source model into the compact PDN teacher.",
        description=(
            "Produces a teacher in the model cache that an experiment can then name with "
            "teacher_source='distilled'. Resumable: rerun the same --name to continue."
        ),
    )
    distill.add_argument("--name", required=True, help="Names the teacher and its directory.")
    distill.add_argument("--model-size", default="small", choices=["small", "medium"])
    distill.add_argument(
        "--corpus",
        default="imagenette",
        choices=["imagenette", "directory"],
        help=(
            "'imagenette' is the smoke corpus already on disk; 'directory' is how ImageNet is used."
        ),
    )
    distill.add_argument("--corpus-path", default="", help="Image tree when --corpus directory.")
    distill.add_argument("--steps", type=int, default=10_000)
    distill.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="The reference uses 16. This is the knob that fits the run into unified memory.",
    )
    distill.add_argument("--normalization-images", type=int, default=1024)
    distill.add_argument("--seed", type=int, default=0)
    distill.add_argument("--checkpoint-every", type=int, default=1000)
    distill.add_argument("--no-resume", action="store_true", help="Refuse an existing name.")
    distill.add_argument("--job-id", type=int, default=0, help="Only labels the event stream.")
    distill.set_defaults(handler=_distill)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    # The result goes to stderr, not stdout: stdout is the event stream, and a summary
    # printed into it would be a line no `parse_line` expects.
    print(json.dumps(result, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
