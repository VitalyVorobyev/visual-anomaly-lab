"""Discovery and one-action registration of local public benchmark packs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.annotations.imported_boxes import (
    ImportedBox,
    ImportedClass,
    ensure_classes,
    write_box_truth,
)
from anomaly_lab.config import Settings
from anomaly_lab.datasets.adapters.csv_table import CsvTableAdapter, CsvTableOptions
from anomaly_lab.datasets.adapters.folder_classes import (
    FolderClassesAdapter,
    FolderClassesOptions,
)
from anomaly_lab.datasets.commit import CommitResult, commit_manifests_atomically
from anomaly_lab.datasets.manifest import Manifest
from anomaly_lab.datasets.voc import clamp_box, read_voc
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
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

# FSS-1000's few-shot panel: 20 of the 240 classes the dataset's authors hold out for
# testing (`fss_test_set.txt` in the upstream repository), taken by `evenly_spaced(240, 20)`
# over that list sorted by name -- a rule, not a choice by eye (docs/measurements.md).
FSS_PANEL = (
    "abe's_flyingfish",
    "banana_boat",
    "bucket",
    "chalk_brush",
    "clam",
    "diver",
    "electronic_stove",
    "flying_snakes",
    "hair_razor",
    "jet_aircraft",
    "little_blue_heron",
    "moist_proof_pad",
    "oriole",
    "poached_egg",
    "rally_car",
    "sealion",
    "spinach",
    "tiltrotor",
    "wandering_albatross",
    "wooden_spoon",
)

# PKU-Market-PCB's six defect kinds: the directory holding each kind's images and VOC files,
# and the class name those files use. The order is the taxonomy's, so it is the class order
# every detection run on the dataset pins.
PCB_DEFECTS = (
    ("Missing_hole", "missing_hole"),
    ("Mouse_bite", "mouse_bite"),
    ("Open_circuit", "open_circuit"),
    ("Short", "short"),
    ("Spur", "spur"),
    ("Spurious_copper", "spurious_copper"),
)


class RegisterReferencePacksParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    pack_keys: list[str] = Field(default_factory=lambda: ["visa", "gkn"])


@dataclass(frozen=True)
class BoxTruthSpec:
    """Where a pack's box annotations are: one Pascal VOC file per image.

    `annotation_dir` and `pattern` take the image's `{class}` (its directory's name) and
    `{stem}`, relative to the dataset's root. `classes` is the taxonomy the files speak, in
    order, keyed by the name a file uses; a box of any other name fails the registration
    rather than inventing a class.
    """

    annotation_dir: str
    pattern: str
    classes: tuple[ImportedClass, ...]

    def path_for(self, root: Path, image: Path) -> Path:
        fields = {"class": image.parent.name, "stem": image.stem}
        return root / self.annotation_dir.format(**fields) / self.pattern.format(**fields)


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
    # Box truth, entered as each image's first completed revision once the dataset is
    # committed: for a pack whose objects are annotated as boxes of several classes, which an
    # imported mask cannot carry.
    box_truth: BoxTruthSpec | None = None


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
    fss = base / "FSS-1000"
    fss_classes = fss / "fewshot_data"
    fss_datasets = tuple(
        DatasetSpec(
            key=f"fss1000:{target}",
            name=target,
            root=fss_classes / target,
            scan_root=fss_classes,
            adapter="folder_classes",
            options={
                # Imported masks answer for the default class alone, so the target's
                # images are its `defect` samples and the rest of the panel its confirmed
                # absences: labelled normal, with their own classes' masks left behind.
                "defect_dirs": [target],
                "normal_dirs": [other for other in FSS_PANEL if other != target],
                "mask_dir": "{dir}",
                "mask_pattern": "{stem}.png",
                "masks_for_normal_dirs": False,
                "import_unnamed_dirs": False,
                # A few classes carry a stray `.jpeg` beside the `.jpg` its mask pairs with.
                "extensions": [".jpg"],
            },
            description=(
                f"FSS-1000's {target.replace('_', ' ')} class as a few-shot target: its "
                "ten images with their masks, and the other nineteen classes of the panel "
                "as images that do not show it."
            ),
        )
        for target in FSS_PANEL
    )
    pcb = base / "PKU-PCB" / "PCB_DATASET"
    pcb_dataset = DatasetSpec(
        key="pku_pcb:pcb",
        name="PKU-Market-PCB",
        root=pcb,
        scan_root=pcb,
        adapter="folder_classes",
        options={
            "defect_dirs": [f"images/{directory}" for directory, _ in PCB_DEFECTS],
            # `rotation/` holds rotated copies with no VOC files, and `PCB_USED/` the
            # defect-free boards the defects were synthesised on; neither is the benchmark.
            "import_unnamed_dirs": False,
            "extensions": [".jpg"],
        },
        description=(
            "Printed circuit boards with six kinds of synthesised defect, every defect "
            "annotated as a box of its kind."
        ),
        box_truth=BoxTruthSpec(
            annotation_dir="Annotations/{class}",
            pattern="{stem}.xml",
            classes=tuple(
                ImportedClass(key=name, name=name.replace("_", " ").capitalize())
                for _, name in PCB_DEFECTS
            ),
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
        PackSpec(
            key="fss1000",
            title="FSS-1000",
            root=fss,
            required=tuple(fss_classes / target for target in FSS_PANEL),
            datasets=fss_datasets,
            install_url="https://github.com/HKUSTCV/FSS-1000",
        ),
        PackSpec(
            key="pku_pcb",
            title="PKU-Market-PCB",
            root=pcb,
            required=tuple(
                pcb / kind / directory
                for kind in ("images", "Annotations")
                for directory, _ in PCB_DEFECTS
            ),
            datasets=(pcb_dataset,),
            install_url="https://robotics.pkusz.edu.cn/resources/datasetENG/",
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


@dataclass
class BoxTruthResult:
    """What entering one dataset's box truth did, for the job's result and log."""

    images: int = 0
    """Images given their first revision by this pass."""
    boxes: int = 0
    kept: int = 0
    """Images that already had a revision or an open draft, which they keep."""
    without_file: int = 0
    """Images with no annotation file: left unlabelled, never read as an absence."""
    clipped: int = 0
    """Boxes cut to the image's frame."""
    reshaped: int = 0
    """Boxes whose instance box differs from the box drawn, because a smaller box overlaps it."""
    classes_added: list[str] | None = None


def register_box_truth(
    settings: Settings,
    spec: DatasetSpec,
    dataset_id: int,
    progress: Any = None,
) -> BoxTruthResult:
    """Enter `spec`'s box truth for every image of the dataset that has no truth of its own.

    Every file is read and checked before the first revision is written, so a malformed
    file or an unknown class fails the pass with nothing changed; the classes are added
    first, because a revision answers only for the classes that existed at its completion.
    """
    box_spec = spec.box_truth
    if box_spec is None:
        raise ValueError(f"{spec.key} has no box truth")
    keys = {entry.key for entry in box_spec.classes}
    with connection(settings.db_path) as conn:
        images = images_repo.list_images_for_dataset(conn, dataset_id)
    result = BoxTruthResult()
    planned: list[tuple[int, int, int, list[ImportedBox]]] = []
    for image in images:
        path = box_spec.path_for(spec.root, Path(image.path))
        if not path.is_file():
            result.without_file += 1
            continue
        annotation = read_voc(path)
        if annotation.size is not None and annotation.size != (image.width, image.height):
            raise ValueError(
                f"{path} describes a {annotation.size[0]}x{annotation.size[1]} image, but "
                f"{image.path} is {image.width}x{image.height}"
            )
        boxes: list[ImportedBox] = []
        for item in annotation.objects:
            if item.name not in keys:
                raise ValueError(f"{path}: class {item.name!r} is not one of {sorted(keys)}")
            clamped = clamp_box(item.box, image.width, image.height)
            if clamped is None:
                raise ValueError(f"{path}: a {item.name} box lies outside the image")
            result.clipped += int(clamped != item.box)
            boxes.append(ImportedBox(item.name, clamped))
        planned.append((image.id, image.width, image.height, boxes))

    result.classes_added = ensure_classes(settings, dataset_id, box_spec.classes)
    for index, (image_id, width, height, boxes) in enumerate(planned):
        written = write_box_truth(settings, image_id, width, height, boxes)
        if written.written:
            result.images += 1
            result.boxes += len(boxes)
            result.reshaped += written.reshaped
        else:
            result.kept += 1
        if progress is not None:
            progress((index + 1) / max(len(planned), 1), f"box truth {index + 1}/{len(planned)}")
    return result


def box_truth_unfinished(settings: Settings, spec: DatasetSpec, dataset_id: int) -> int:
    """Images whose box truth an earlier pass left unentered: a file and no truth of their own.

    Box truth is entered after the commit and image by image, so a cancelled or failed pass
    leaves a registered dataset partly labelled. The catalogue counts such a dataset as
    pending, which is what lets the next registration finish it.
    """
    if spec.box_truth is None:
        return 0
    with connection(settings.db_path) as conn:
        rows = conn.execute(
            """
            SELECT image.path
              FROM image JOIN sample ON sample.id = image.sample_id
             WHERE sample.dataset_id = ?
               AND NOT EXISTS (SELECT 1 FROM annotation_revision AS revision
                                WHERE revision.image_id = image.id)
               AND NOT EXISTS (SELECT 1 FROM annotation_draft AS draft
                                WHERE draft.image_id = image.id)
            """,
            (dataset_id,),
        ).fetchall()
    return sum(spec.box_truth.path_for(spec.root, Path(row["path"])).is_file() for row in rows)


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

    # Box truth is entered after the commit, for every dataset of the chosen packs that has
    # it -- also one registered earlier, so an interrupted pass is finished by the next.
    boxed = [spec for key in params.pack_keys for spec in known[key].datasets if spec.box_truth]
    scan_share = 0.5 if boxed else 0.9

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
            context.progress(overall * scan_share, f"{dataset_name}: {message or 'scanning'}")

        context.log(f"scanning {spec.key}")
        manifests.append(scan_spec(spec, report))

    context.raise_if_cancelled()
    with connection(context.settings.db_path) as conn:
        results: list[CommitResult] = commit_manifests_atomically(conn, context.settings, manifests)
        registered = datasets_repo.list_datasets(conn)

    box_truth: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(boxed):
        dataset_id = registered_dataset_id(spec, registered)
        if dataset_id is None:  # pragma: no cover - committed just above
            raise RuntimeError(f"{spec.key} is not registered")

        def box_report(
            fraction: float, message: str, position: int = index, dataset_name: str = spec.name
        ) -> None:
            context.raise_if_cancelled()
            overall = (position + fraction) / len(boxed)
            context.progress(
                scan_share + overall * (1.0 - scan_share), f"{dataset_name}: {message}"
            )

        context.log(f"entering the box truth of {spec.key}")
        entered = register_box_truth(context.settings, spec, dataset_id, box_report)
        context.log(
            f"{spec.key}: {entered.boxes} boxes on {entered.images} images; {entered.kept} kept "
            f"their own truth, {entered.without_file} have no annotation file, "
            f"{entered.clipped} boxes clipped to the frame, {entered.reshaped} reshaped by an "
            "overlapping box"
        )
        box_truth[spec.key] = asdict(entered)

    context.progress(1.0, f"registered {len(results)} datasets")
    return {
        "registered": len(results),
        "already_registered": sum(len(known[key].datasets) for key in params.pack_keys)
        - len(results),
        "dataset_ids": [result.dataset_id for result in results],
        "box_truth": box_truth,
    }
