"""Where the pixels are, and which of them a few-shot protocol is allowed to see.

Three benchmarks, read straight off disk under the gitignored `datasets/` tree. The
campaign never touches the application database: an experiment there is a catalogue entry
a human curates, and a sweep that leaves nine hundred of them behind has destroyed the
catalogue to produce a table (ADR-0038).

**A few-shot protocol is mostly a statement about what the fit set may contain**, so the
enumeration is the part worth being careful about:

  * **VisA** ships `split_csv/1cls.csv`, the official one-class split every paper reports
    against. It is read rather than reconstructed — a directory walk would silently
    disagree with the published numbers about which images are test images.
  * **MVTec-AD** carries its split in its directory layout: `train/good` is normal-only by
    construction and `test/<defect>` holds the rest, with a mask per anomalous image.
  * **GKN** has no published split at all, which is the interesting case. Drawing the fit
    images from the same pool the test set is scored on would let a 1-shot run be graded on
    an image it was fitted to. So the normals are cut once, deterministically, by sorted
    filename: the first `GKN_FIT_POOL` are the only images any seed may draw from, and the
    rest are test normals. The cut does not depend on the arm's seed, so every arm is
    scored on exactly the same test set.

`rotation_safe` carries the paper's one category-level carve-out. MVTec's `transistor` is
orientation-sensitive — the defect *is* a misplaced part — so rotating its normals teaches
the subspace that every orientation is normal, and the method stops being able to see the
defect at all.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

VISA_DIRNAME = "VisA_20220922"
MVTEC_DIRNAME = "MVTec-AD"
GKN_DIRNAME = "GKN Blade Surface Defect Dataset"

GKN_FIT_POOL = 50
"""How many of GKN's normals are reserved as the only images a fit may draw from.

Large enough that seeds draw genuinely different 4-shot sets, small enough that the 153
remaining normals still outnumber the 197 defects by the same order. The number is
arbitrary in the way a split is always arbitrary; what matters is that it is fixed here
rather than derived from the arm, so no run is scored on an image it saw.
"""

# MVTec-AD's own category-level exception, stated in the paper's implementation details:
# rotation augmentation is applied "except for the orientation-sensitive transistor
# category". Rotating those normals would put every orientation inside the normal
# subspace, and a misplaced transistor is exactly a wrong orientation.
ROTATION_UNSAFE: frozenset[tuple[str, str]] = frozenset({("mvtec", "transistor")})


@dataclass(frozen=True)
class ScoredImage:
    """One image on the test side of a split: the file, its label, and its truth."""

    path: Path
    label: int
    """1 for anomalous, 0 for normal — the convention every metric in `anomaly_lab.eval`
    reads."""
    mask_path: Path | None
    defect: str
    """The defect subtype directory, or `"good"`. Carried so a per-category report can say
    *which* defects a configuration misses rather than only how often."""


@dataclass(frozen=True)
class CategorySplit:
    """Everything one (benchmark, category) contributes to a campaign."""

    benchmark: str
    category: str
    fit_pool: tuple[Path, ...]
    """Normal images a seed may draw its k shots from, in a stable order."""
    test_items: tuple[ScoredImage, ...]

    @property
    def key(self) -> str:
        return f"{self.benchmark}:{self.category}"

    @property
    def rotation_safe(self) -> bool:
        return (self.benchmark, self.category) not in ROTATION_UNSAFE

    @property
    def mask_count(self) -> int:
        return sum(1 for item in self.test_items if item.mask_path is not None)

    @property
    def anomaly_count(self) -> int:
        return sum(item.label for item in self.test_items)


class BenchmarkMissingError(FileNotFoundError):
    """A benchmark was asked for and its directory is not on this machine.

    Its own type so a campaign can report "MVTec-AD is not downloaded" as a prerequisite
    rather than as an unexplained traceback four layers into a path join.
    """


def visa_splits(root: Path) -> list[CategorySplit]:
    """The twelve VisA classes, from the official one-class CSV."""
    table = root / "split_csv" / "1cls.csv"
    if not table.is_file():
        msg = (
            f"VisA's official split table is missing at {table}. Download the pack from "
            "https://github.com/amazon-science/spot-diff and unpack it under datasets/."
        )
        raise BenchmarkMissingError(msg)

    fit: dict[str, list[Path]] = {}
    tests: dict[str, list[ScoredImage]] = {}
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = row["object"]
            image = root / row["image"]
            if row["split"] == "train":
                fit.setdefault(category, []).append(image)
                continue
            anomalous = row["label"] == "anomaly"
            mask = row["mask"].strip()
            tests.setdefault(category, []).append(
                ScoredImage(
                    path=image,
                    label=int(anomalous),
                    mask_path=root / mask if mask else None,
                    defect="anomaly" if anomalous else "good",
                )
            )

    return [
        CategorySplit(
            benchmark="visa",
            category=category,
            fit_pool=tuple(sorted(fit.get(category, []))),
            test_items=tuple(sorted(tests[category], key=lambda item: item.path)),
        )
        for category in sorted(tests)
    ]


def mvtec_splits(root: Path) -> list[CategorySplit]:
    """The fifteen MVTec-AD categories, from the directory layout that *is* the split.

    The mirror this campaign reads renames the original `ground_truth/` to `masks/` and
    keeps the `_mask` suffix. A mask that does not exist where it is expected is a fault
    rather than a normal image: the layout says the image is anomalous, so scoring it with
    no truth would quietly drop it from the pixel metrics while keeping it in the image
    ones.
    """
    images = root / "images"
    if not (images / "train").is_dir():
        msg = (
            f"MVTec-AD is not unpacked at {root}. It is a prerequisite for the SubspaceAD "
            "campaign; see docs/measurements.md for the mirror the campaign was run "
            "against, and note the dataset's CC BY-NC-SA 4.0 licence."
        )
        raise BenchmarkMissingError(msg)

    splits: list[CategorySplit] = []
    for category_dir in sorted(path for path in (images / "train").iterdir() if path.is_dir()):
        category = category_dir.name
        fit_pool = tuple(sorted(_pictures(category_dir / "good")))
        items: list[ScoredImage] = []
        for defect_dir in sorted(path for path in (images / "test" / category).iterdir()):
            if not defect_dir.is_dir():
                continue
            anomalous = defect_dir.name != "good"
            for picture in sorted(_pictures(defect_dir)):
                mask = None
                if anomalous:
                    mask = root / "masks" / "test" / category / defect_dir.name
                    mask = mask / f"{picture.stem}_mask{picture.suffix}"
                    if not mask.is_file():
                        msg = f"{picture} is labelled anomalous but has no mask at {mask}"
                        raise BenchmarkMissingError(msg)
                items.append(
                    ScoredImage(
                        path=picture,
                        label=int(anomalous),
                        mask_path=mask,
                        defect=defect_dir.name,
                    )
                )
        splits.append(
            CategorySplit(
                benchmark="mvtec",
                category=category,
                fit_pool=fit_pool,
                test_items=tuple(items),
            )
        )
    return splits


def gkn_splits(root: Path) -> list[CategorySplit]:
    """GKN as one category, with the fit pool cut off the front of the sorted normals.

    No masks anywhere in this dataset, so `mask_path` is `None` throughout and the campaign
    reports image-level metrics only for it — which is the honest thing to do and the
    reason a metric that cannot be computed is `None` rather than zero.
    """
    data = root / "Data_GKN"
    if not (data / "Good").is_dir():
        msg = f"GKN is not unpacked at {root}; expected a Data_GKN/Good directory."
        raise BenchmarkMissingError(msg)

    normals = sorted(_pictures(data / "Good"))
    if len(normals) <= GKN_FIT_POOL:
        msg = (
            f"GKN has {len(normals)} normal images, which does not leave a test set after "
            f"reserving {GKN_FIT_POOL} for fitting."
        )
        raise BenchmarkMissingError(msg)

    items = [
        ScoredImage(path=picture, label=0, mask_path=None, defect="good")
        for picture in normals[GKN_FIT_POOL:]
    ]
    for defect in ("Nick", "Scratch"):
        items.extend(
            ScoredImage(path=picture, label=1, mask_path=None, defect=defect.lower())
            for picture in sorted(_pictures(data / defect))
        )
    return [
        CategorySplit(
            benchmark="gkn",
            category="blade-surface",
            fit_pool=tuple(normals[:GKN_FIT_POOL]),
            test_items=tuple(items),
        )
    ]


_LOADERS = {"visa": (VISA_DIRNAME, visa_splits), "mvtec": (MVTEC_DIRNAME, mvtec_splits)}


def load_splits(datasets_dir: Path, benchmark: str) -> list[CategorySplit]:
    """Every category of one benchmark, named the way the campaign's arms name it."""
    if benchmark == "gkn":
        return gkn_splits(datasets_dir / GKN_DIRNAME)
    entry = _LOADERS.get(benchmark)
    if entry is None:
        known = ", ".join(("gkn", *sorted(_LOADERS)))
        msg = f"unknown benchmark {benchmark!r}; known benchmarks are {known}"
        raise ValueError(msg)
    dirname, loader = entry
    return loader(datasets_dir / dirname)


_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"})


def _pictures(directory: Path) -> Iterator[Path]:
    """Image files in one directory, with macOS's `.DS_Store` and friends left out.

    GKN's tree carries `.DS_Store` entries; a glob that trusts the directory listing reads
    one as an image and fails inside Pillow, several hundred images into a pass.
    """
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_file() and path.suffix in _SUFFIXES and not path.name.startswith("."):
            yield path
