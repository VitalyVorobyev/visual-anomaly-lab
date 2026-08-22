"""Discovery and one-action registration of local public benchmark packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.config import Settings
from anomaly_lab.datasets.adapters.csv_table import CsvTableAdapter, CsvTableOptions
from anomaly_lab.datasets.adapters.folder_classes import (
    FolderClassesAdapter,
    FolderClassesOptions,
)
from anomaly_lab.datasets.commit import CommitResult, commit_manifests_atomically
from anomaly_lab.datasets.manifest import Manifest
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.domain.entities import Dataset
from anomaly_lab.jobs.context import JobContext

VISA_CLASSES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


class RegisterReferencePacksParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    pack_keys: list[str] = Field(default_factory=lambda: ["visa", "gkn"])


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    root: Path
    scan_root: Path
    adapter: str
    options: dict[str, Any]
    # What the catalogue says about this dataset when nobody has written anything. A VisA
    # class is named `candle` and nothing else on the card explains what that is.
    description: str = ""


@dataclass(frozen=True)
class PackSpec:
    key: str
    title: str
    root: Path
    required: tuple[Path, ...]
    datasets: tuple[DatasetSpec, ...]
    install_url: str
    # What the catalogue files this pack's datasets under. Separate from `title` because a
    # pack of one dataset otherwise heads a group with the same words as the single card
    # inside it -- "GKN Blade Surface Defect" over "GKN Blade Surface Defect".
    collection: str = ""

    def collection_name(self) -> str:
        return self.collection or self.title


def pack_specs(settings: Settings) -> tuple[PackSpec, ...]:
    base = settings.reference_datasets_dir
    visa = base / "VisA_20220922"
    gkn = base / "GKN Blade Surface Defect Dataset"
    visa_datasets = tuple(
        DatasetSpec(
            key=f"visa:{category}",
            name=category,
            root=visa / category,
            scan_root=visa,
            adapter="csv_table",
            options={
                "csv_path": "split_csv/1cls.csv",
                "filter_column": "object",
                "filter_value": category,
            },
            description=(
                f"The {category} object class of the VisA benchmark, with pixel-level "
                "masks for every anomaly and the published one-class split."
            ),
        )
        for category in VISA_CLASSES
    )
    gkn_dataset = DatasetSpec(
        key="gkn:blade-surface",
        name="GKN Blade Surface Defect",
        root=gkn,
        scan_root=gkn,
        adapter="folder_classes",
        options={
            "normal_dirs": ["Data_GKN/Good"],
            "defect_dirs": ["Data_GKN/Nick", "Data_GKN/Scratch"],
        },
        description=(
            "Surface photographs of turbine blades, labelled good against two defect "
            "kinds -- nick and scratch -- by the directory they were published in."
        ),
    )
    return (
        PackSpec(
            key="visa",
            title="VisA",
            root=visa,
            required=(visa / "split_csv" / "1cls.csv", *(visa / c for c in VISA_CLASSES)),
            datasets=visa_datasets,
            install_url="https://github.com/amazon-science/spot-diff",
        ),
        PackSpec(
            key="gkn",
            title="GKN Blade Surface Defect",
            collection="GKN",
            root=gkn,
            required=(
                gkn / "Data_GKN" / "Good",
                gkn / "Data_GKN" / "Nick",
                gkn / "Data_GKN" / "Scratch",
            ),
            datasets=(gkn_dataset,),
            install_url="https://doi.org/10.17632/3bh998k78g.1",
        ),
    )


def is_present(path: Path) -> bool:
    return path.is_file() if path.suffix else path.is_dir()


def registered_dataset_id(spec: DatasetSpec, datasets: list[Dataset]) -> int | None:
    """Match provider ownership without treating a name collision as registration.

    Early VisA imports used the pack root while current registrations use the class root,
    so both are accepted. The adapter still has to match: a user dataset called
    ``candle`` must not silently suppress the public reference class.
    """
    expected_roots = {spec.root.resolve(), spec.scan_root.resolve()}
    for dataset in datasets:
        if (
            dataset.name == spec.name
            and dataset.adapter == spec.adapter
            and Path(dataset.root_path).resolve() in expected_roots
        ):
            return dataset.id
    return None


@dataclass(frozen=True)
class PackMembership:
    """What a dataset inherits from the reference pack it was registered from."""

    collection: str
    description: str


def pack_membership(settings: Settings, datasets: list[Dataset]) -> dict[int, PackMembership]:
    """Map dataset id to its pack, resolving each spec's roots once for the whole list.

    Membership stays derived rather than written at registration. Storing it would make
    every dataset that predates the column look unaffiliated until it was re-registered,
    and would need a backfill that could only recompute exactly this. The stored
    `dataset.collection` is therefore an *override*, not the record of where a dataset
    came from.
    """
    found: dict[int, PackMembership] = {}
    for pack in pack_specs(settings):
        for spec in pack.datasets:
            dataset_id = registered_dataset_id(spec, datasets)
            if dataset_id is not None:
                found[dataset_id] = PackMembership(
                    collection=pack.collection_name(),
                    description=spec.description,
                )
    return found


def scan_spec(spec: DatasetSpec, progress: Any) -> Manifest:
    if spec.adapter == "csv_table":
        csv_options = CsvTableOptions.model_validate(spec.options)
        manifest = CsvTableAdapter.scan(
            spec.scan_root, csv_options, dataset_name=spec.name, progress=progress
        )
    else:
        folder_options = FolderClassesOptions.model_validate(spec.options)
        manifest = FolderClassesAdapter.scan(
            spec.scan_root, folder_options, dataset_name=spec.name, progress=progress
        )
    # VisA's table is rooted at the pack, but each object class is its own dataset.
    # All image/mask paths are already absolute by this point, so the class directory is
    # the honest stable identity and avoids pretending twelve datasets have one root.
    return manifest.model_copy(update={"root_path": str(spec.root)})


def run_reference_pack_job(context: JobContext) -> dict[str, Any]:
    params = RegisterReferencePacksParams.model_validate(dict(context.params))
    wanted = set(params.pack_keys)
    known = {pack.key: pack for pack in pack_specs(context.settings)}
    unknown = sorted(wanted - known.keys())
    if unknown:
        raise ValueError(f"unknown reference pack(s): {', '.join(unknown)}")

    with connection(context.settings.db_path) as conn:
        existing = datasets_repo.list_datasets(conn)
    selected: list[DatasetSpec] = []
    for key in params.pack_keys:
        pack = known[key]
        missing = [path for path in pack.required if not is_present(path)]
        if missing:
            raise FileNotFoundError(f"{pack.title} is incomplete; missing {missing[0]}")
        selected.extend(
            spec for spec in pack.datasets if registered_dataset_id(spec, existing) is None
        )

    manifests: list[Manifest] = []
    total = len(selected)
    for index, spec in enumerate(selected):
        context.raise_if_cancelled()

        def report(
            fraction: float,
            message: str | None,
            position: int = index,
            dataset_name: str = spec.name,
        ) -> None:
            context.raise_if_cancelled()
            overall = (position + fraction) / max(total, 1)
            context.progress(overall * 0.9, f"{dataset_name}: {message or 'scanning'}")

        context.log(f"scanning {spec.key}")
        manifests.append(scan_spec(spec, report))

    context.raise_if_cancelled()
    with connection(context.settings.db_path) as conn:
        results: list[CommitResult] = commit_manifests_atomically(conn, context.settings, manifests)
    context.progress(1.0, f"registered {len(results)} datasets")
    return {
        "registered": len(results),
        "already_registered": sum(len(known[key].datasets) for key in params.pack_keys)
        - len(results),
        "dataset_ids": [result.dataset_id for result in results],
    }
