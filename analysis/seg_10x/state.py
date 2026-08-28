"""Segmentation state for the per-odor curation GUI, with no GUI in it."""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..session.merge import merge_masks
from .watershed import GLOM_10X_DEFAULTS, segment_image

PHASE_TUNE = "tune"
PHASE_MERGE = "merge"
PHASE_CURATE = "curate"

PHASES = (PHASE_TUNE, PHASE_MERGE, PHASE_CURATE)

# Defaults for combining per-odor masks. `min_overlap` is the one worth
# sweeping: with few odors the merge is nearly a union, but across sixteen the
# detections-per-ROI curve is what tells you where to put it.
MERGE_DEFAULTS = {
    "min_overlap": 0.5,
    "metric": "iou",
    "linkage": "complete",
    "min_detections": 1,
    "consensus_fraction": 0.0,
}

MERGE_SLIDER_RANGES = {
    "min_overlap": (0.05, 0.95, 0.05),
    "min_detections": (1, 16, 1),
    "consensus_fraction": (0.0, 1.0, 0.1),
}

# Slider ranges. min/max diameter deliberately reach past GLOM_10X_DEFAULTS in
# both directions -- the old GUI's min_area slider stopped at 200 px when the
# settled minimum was 314, so the interface could not express its own defaults.
SLIDER_RANGES = {
    "blur_sigma_px": (0.0, 10.0, 0.5),
    "threshold_pctl": (0.0, 99.5, 0.5),
    "adaptive_block_px": (0, 401, 10),
    "min_diameter_px": (2.0, 80.0, 1.0),
    "max_diameter_px": (10.0, 200.0, 1.0),
    "peak_distance_px": (2.0, 60.0, 1.0),
    "border_px": (0, 100, 1),
}

MANUAL_MIN_SCALE = 0.5
MANUAL_MAX_SCALE = 1.5


@dataclass
class Curation:
    """Manual edits to the merged mask. Only meaningful in the curate phase."""

    deleted: set[int] = field(default_factory=set)
    # Watershed seed points, not fixed disks: a hand-placed ROI should follow
    # the image the way an automatic one does, so its footprint is comparable.
    added_seeds: list[tuple[int, int]] = field(default_factory=list)
    # Indices into `added_seeds` that were subsequently deleted. Seeds are
    # tombstoned rather than removed so the surviving ones keep their indices,
    # which is what `provenance` refers to.
    deleted_seeds: set[int] = field(default_factory=set)
    exclude_polygons: list[list[tuple[int, int]]] = field(default_factory=list)

    def live_seeds(self) -> list[tuple[int, int, int]]:
        """(index, y, x) for seeds that have not been deleted."""
        return [
            (i, y, x)
            for i, (y, x) in enumerate(self.added_seeds)
            if i not in self.deleted_seeds
        ]

    def is_empty(self) -> bool:
        return not (
            self.deleted
            or self.live_seeds()
            or self.exclude_polygons
        )


def border_mask(shape: tuple[int, int], border_px: int) -> np.ndarray:
    """True within `border_px` of any edge."""

    mask = np.zeros(shape, dtype=bool)

    if border_px <= 0:
        return mask

    b = int(border_px)
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True

    return mask


def grow_seed(
    image: np.ndarray,
    seed: tuple[int, int],
    *,
    unclaimed: np.ndarray,
    min_diameter_px: float,
    max_diameter_px: float,
    threshold_pctl: float = 60.0,
) -> np.ndarray:
    """Grow one hand-placed seed into an ROI by watershed, as `segment_image` does."""

    from skimage.segmentation import watershed

    h, w = image.shape
    y, x = int(seed[0]), int(seed[1])

    if not (0 <= y < h and 0 <= x < w):
        return np.zeros((h, w), dtype=bool)

    # Window generous enough to contain the largest allowed ROI.
    half = int(max_diameter_px)
    r0, r1 = max(y - half, 0), min(y + half + 1, h)
    c0, c1 = max(x - half, 0), min(x + half + 1, w)

    window = image[r0:r1, c0:c1]
    free = unclaimed[r0:r1, c0:c1]

    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return np.zeros((h, w), dtype=bool)

    allowed = free & np.isfinite(window) & (window > np.percentile(finite, threshold_pctl))

    ly, lx = y - r0, x - c0

    def fallback_disk() -> np.ndarray:
        """A disk of the minimum size, over whatever pixels are free."""
        yy, xx = np.ogrid[: window.shape[0], : window.shape[1]]
        radius = min_diameter_px / 2
        disk = ((yy - ly) ** 2 + (xx - lx) ** 2 <= radius**2) & free
        out = np.zeros((h, w), dtype=bool)
        out[r0:r1, c0:c1] = disk
        return out

    if not allowed[ly, lx]:
        # Clicked below threshold or on a taken pixel: fall back to a disk so
        # the click still does something predictable.
        return fallback_disk()

    markers = np.zeros(window.shape, dtype=np.int32)

    if r0 > 0:
        markers[0, :] = 2
    if r1 < h:
        markers[-1, :] = 2
    if c0 > 0:
        markers[:, 0] = 2
    if c1 < w:
        markers[:, -1] = 2

    markers[ly, lx] = 1

    flooded = watershed(
        -np.where(np.isfinite(window), window, -np.inf),
        markers=markers,
        mask=allowed,
    )

    region = flooded == 1

    # Respect the same ceiling the automatic path uses.
    max_area = np.pi * (max_diameter_px / 2) ** 2
    if region.sum() > max_area:
        distance = (np.indices(window.shape) - np.array([[[ly]], [[lx]]])) ** 2
        distance = distance.sum(axis=0)
        keep = np.argsort(distance[region])[: int(max_area)]
        trimmed = np.zeros_like(region)
        ys, xs = np.nonzero(region)
        trimmed[ys[keep], xs[keep]] = True
        region = trimmed

    out = np.zeros((h, w), dtype=bool)
    out[r0:r1, c0:c1] = region

    min_area = np.pi * (min_diameter_px / 2) ** 2
    if out.sum() < min_area:
        disk = fallback_disk()
        if disk.sum() > out.sum():
            return disk

    return out


def manual_bounds(params: dict) -> dict:
    """Size and threshold bounds for a hand-placed seed, relaxed from the shared segmentation parameters."""

    return {
        "min_diameter_px": params["min_diameter_px"] * MANUAL_MIN_SCALE,
        "max_diameter_px": params["max_diameter_px"] * MANUAL_MAX_SCALE,
        "threshold_pctl": params["threshold_pctl"],
    }


class SegmentationState:
    """Parameters, phase, and edits for a set of per-odor images."""

    def __init__(self, images: dict, params: None | dict = None):
        if not images:
            raise ValueError("No images given.")

        shapes = {np.asarray(v).shape for v in images.values()}
        if len(shapes) != 1:
            raise ValueError(f"Images have differing shapes: {shapes}.")

        self.images = {k: np.asarray(v) for k, v in images.items()}
        self.keys = sorted(images, key=repr)
        self.shape = shapes.pop()

        self.shared = {**GLOM_10X_DEFAULTS, "border_px": 0}
        if params:
            self.shared.update(params)

        # Per-odor overrides, empty until a key is explicitly overridden.
        self.overrides: dict = {}
        self.merge_params = dict(MERGE_DEFAULTS)

        # One curation, applied to the merged mask -- not one per odor.
        self.curation = Curation()

        self.phase = PHASE_TUNE
        self.active = self.keys[0]

        self._masks: dict = {}
        self._merged = None

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    def params_for(self, key) -> dict:
        """Effective parameters for one odor: shared, plus any override."""
        return {**self.shared, **self.overrides.get(key, {})}

    def set_param(self, name: str, value, *, this_odor_only: bool) -> None:
        """Change a parameter. Refused once curation has begun."""

        if self.phase != PHASE_TUNE:
            raise RuntimeError(
                f"Segmentation parameters are frozen in the {self.phase} phase. "
                "Go back to tuning first."
            )

        if name not in self.shared:
            raise KeyError(f"Unknown parameter {name!r}.")

        if this_odor_only:
            self.overrides.setdefault(self.active, {})[name] = value
        else:
            self.shared[name] = value
            # A shared change is meant to be visible, so clear any override of
            # the same parameter that would mask it.
            for override in self.overrides.values():
                override.pop(name, None)

        self._masks.clear()
        self._merged = None

    def clear_override(self, key=None) -> None:
        self.overrides.pop(self.active if key is None else key, None)
        self._masks.clear()
        self._merged = None

    # ------------------------------------------------------------------ #
    # Phase
    # ------------------------------------------------------------------ #

    def set_merge_param(self, name: str, value) -> None:
        """Change a merge parameter. Only live in the merge phase."""

        if self.phase != PHASE_MERGE:
            raise RuntimeError(
                f"Merge parameters are only adjustable in the merge phase, "
                f"not {self.phase}."
            )

        if name not in self.merge_params:
            raise KeyError(f"Unknown merge parameter {name!r}.")

        self.merge_params[name] = value
        self._merged = None

    def begin_merge(self) -> None:
        """Freeze segmentation, segment every odor, and preview the merge."""
        self.segment_all()
        self.phase = PHASE_MERGE
        self._merged = None

    def begin_curation(self) -> None:
        """Freeze the merge and open editing on the merged mask."""
        if self.phase == PHASE_TUNE:
            self.begin_merge()
        self.merged()
        self.phase = PHASE_CURATE

    def has_curation(self) -> bool:
        return not self.curation.is_empty()

    def back(self, *, discard_curation: bool = False) -> str:
        """
        Step one phase back. Curation cannot survive a parameter change, so
        leaving the curate phase discards it and says how much.
        """

        if self.phase == PHASE_TUNE:
            return self.phase

        if self.phase == PHASE_CURATE:
            if self.has_curation() and not discard_curation:
                edits = self.curation
                raise RuntimeError(
                    f"Going back discards {len(edits.deleted)} deletion(s), "
                    f"{len(edits.live_seeds())} addition(s) and "
                    f"{len(edits.exclude_polygons)} exclusion(s). "
                    "Pass discard_curation=True to confirm."
                )
            self.curation = Curation()
            self.phase = PHASE_MERGE
        else:
            self.phase = PHASE_TUNE

        return self.phase

    # Kept so existing callers and tests keep working.
    def return_to_tuning(self, *, discard_curation: bool = False) -> None:
        while self.phase != PHASE_TUNE:
            self.back(discard_curation=discard_curation)

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #

    def segment(self, key) -> np.ndarray:
        """Auto-segment one odor, cached until a parameter changes."""

        if key in self._masks:
            return self._masks[key]

        params = self.params_for(key)
        border = params.pop("border_px", 0)

        mask, record = segment_image(
            self.images[key],
            exclude_mask=border_mask(self.shape, border),
            **params,
        )

        record["border_px"] = int(border)
        record["group_key"] = repr(key)

        self._masks[key] = mask
        self._records = getattr(self, "_records", {})
        self._records[key] = record

        return mask

    def segment_all(self) -> dict:
        return {key: self.segment(key) for key in self.keys}

    def combined_image(self) -> np.ndarray:
        """Per-pixel maximum across odors -- what the merged mask sits on."""

        if getattr(self, "_combined", None) is None:
            self._combined = np.nanmax(
                np.stack([self.images[k] for k in self.keys]), axis=0
            )

        return self._combined

    def merged(self):
        """Per-odor masks combined into one consensus mask, cached."""

        if self._merged is None:
            self._merged = merge_masks(
                [self.segment(k) for k in self.keys],
                min_area_px=np.pi * (self.shared["min_diameter_px"] / 2) ** 2,
                **self.merge_params,
            )

        return self._merged

    def merged_provenance(self) -> dict[int, list]:
        """Which odors found each merged ROI, by key rather than index."""
        return {
            roi: [self.keys[source] for source, _ in members]
            for roi, members in self.merged().provenance.items()
        }

    def curated_mask(self, key=None) -> np.ndarray:
        """The merged mask with manual edits applied."""

        mask = self.merged().labels.copy()
        edits = self.curation
        min_area = np.pi * (self.shared["min_diameter_px"] / 2) ** 2

        for roi_id in edits.deleted:
            mask[mask == roi_id] = 0

        for polygon in edits.exclude_polygons:
            mask[_polygon_mask(self.shape, polygon)] = 0

        if edits.exclude_polygons:
            mask, self._clipped = _drop_clipped(mask, min_area_px=min_area)
        else:
            self._clipped = {}

        # Renumber so labels stay contiguous after deletions.
        remaining = [i for i in np.unique(mask) if i > 0]
        out = np.zeros_like(mask)
        for new_id, old_id in enumerate(remaining, start=1):
            out[mask == old_id] = new_id

        # Where each output label came from, so a click can be traced back to
        # the thing that produced it: ("merged", id) or ("seed", index).
        provenance = {
            new_id: ("merged", int(old_id))
            for new_id, old_id in enumerate(remaining, start=1)
        }

        self._rejected_seeds: dict[int, int] = {}

        next_id = len(remaining) + 1
        image = self.combined_image()

        # The odor on screen when the click happened, overrides included.
        bounds = manual_bounds(self.params_for(self.active))
        manual_min_area = np.pi * (bounds["min_diameter_px"] / 2) ** 2

        for index, y, x in edits.live_seeds():
            region = grow_seed(image, (y, x), unclaimed=out == 0, **bounds)

            if region.sum() >= manual_min_area:
                out[region] = next_id
                provenance[next_id] = ("seed", index)
                next_id += 1
            else:
                self._rejected_seeds[index] = int(region.sum())

        self._label_provenance = provenance

        return out

    # ------------------------------------------------------------------ #
    # Edits
    # ------------------------------------------------------------------ #

    def _require_curating(self) -> None:
        if self.phase != PHASE_CURATE:
            raise RuntimeError("Manual edits are only available while curating.")

    def delete_at(self, y: int, x: int) -> None | tuple[str, int]:
        """Delete whatever ROI is under (y, x), automatic or hand-added."""

        self._require_curating()

        labels = self.curated_mask()
        roi_id = int(labels[y, x])

        if roi_id <= 0:
            return None

        source, ident = self._label_provenance[roi_id]

        if source == "merged":
            self.curation.deleted.add(ident)
        else:
            self.curation.deleted_seeds.add(ident)

        return source, ident

    def add_at(self, y: int, x: int) -> int:
        """Place a watershed seed; the ROI grows to fit the image under it."""
        self._require_curating()
        self.curation.added_seeds.append((int(y), int(x)))
        return len(self.curation.added_seeds) - 1

    def exclude_polygon(self, vertices: list[tuple[int, int]]) -> None:
        self._require_curating()
        if len(vertices) >= 3:
            self.curation.exclude_polygons.append(list(vertices))

    def reset_curation(self, key=None) -> None:
        self._require_curating()
        self.curation = Curation()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        per_odor = {}
        for key in self.keys:
            mask = self.segment(key)
            per_odor[repr(key)] = {
                "n_rois": int(mask.max()),
                "coverage_pct": round(100 * float((mask > 0).mean()), 2),
                "overridden_params": sorted(self.overrides.get(key, {})),
            }

        merged = self.curated_mask()
        areas = np.bincount(merged.ravel())[1:]
        diameters = 2 * np.sqrt(areas / np.pi) if len(areas) else np.array([])
        detections = (
            self.merged().detections_per_roi() if self.merged().n_rois else np.array([])
        )

        return {
            "per_odor": per_odor,
            "merged": {
                "n_rois": int(merged.max()),
                "median_diameter_px": (
                    round(float(np.median(diameters)), 1) if len(diameters) else None
                ),
                "coverage_pct": round(100 * float((merged > 0).mean()), 2),
                "found_by_multiple_odors": int((detections > 1).sum()),
                "deleted": len(self.curation.deleted),
                "added": len(self.curation.live_seeds()),
                "excluded_polygons": len(self.curation.exclude_polygons),
                "added_rejected_too_small": len(getattr(self, "_rejected_seeds", {})),
                "clipped_below_floor": len(getattr(self, "_clipped", {})),
                "min_area_px": round(
                    float(np.pi * (self.shared["min_diameter_px"] / 2) ** 2), 1
                ),
                "manual_min_area_px": round(
                    float(np.pi * (manual_bounds(self.params_for(self.active))["min_diameter_px"] / 2) ** 2),
                    1,
                ),
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write curated masks plus everything needed to reproduce them."""

        path = Path(path).with_suffix(".h5")
        path.parent.mkdir(parents=True, exist_ok=True)

        # The curated merged mask is the deliverable; the per-odor masks are
        # kept so the merge can be re-derived or re-tuned without re-reading
        # any movies.
        config = {
            "shared": self.shared,
            "overrides": {repr(k): v for k, v in self.overrides.items()},
            "merge_params": self.merge_params,
            "curation": {
                "deleted": sorted(self.curation.deleted),
                "added_seeds": [list(s) for s in self.curation.added_seeds],
                "deleted_seeds": sorted(self.curation.deleted_seeds),
                "exclude_polygons": self.curation.exclude_polygons,
            },
            "provenance": {
                str(roi): [repr(k) for k in keys]
                for roi, keys in self.merged_provenance().items()
            },
            "summary": self.summary(),
        }

        import h5py
        with h5py.File(path, "w") as handle:
            handle.attrs["file_type"] = "odyn_10x_working_mask"
            handle.attrs["config_json"] = json.dumps(config)
            masks = handle.create_group("masks")
            masks.create_dataset("labels", data=self.curated_mask(), compression="gzip")
            for key in self.keys:
                masks.create_dataset(str(key), data=self.segment(key), compression="gzip")

        return path


def _drop_clipped(
    labels: np.ndarray, *, min_area_px: float
) -> tuple[np.ndarray, dict[int, int]]:
    """Re-judge every ROI after an exclusion polygon has cut into the mask."""

    from scipy.ndimage import label as connected_components

    out = labels.copy()
    dropped: dict[int, int] = {}

    for roi_id in [int(i) for i in np.unique(labels) if i > 0]:
        region = labels == roi_id

        pieces, n_pieces = connected_components(region)
        sizes = np.bincount(pieces.ravel())[1:]
        largest = int(sizes.argmax()) + 1

        if n_pieces > 1:
            out[region & (pieces != largest)] = 0

        if sizes[largest - 1] < min_area_px:
            out[region] = 0
            dropped[roi_id] = int(sizes[largest - 1])

    return out, dropped


def _polygon_mask(shape: tuple[int, int], vertices: list[tuple[int, int]]) -> np.ndarray:
    from skimage.draw import polygon as sk_polygon

    ys = np.array([v[0] for v in vertices])
    xs = np.array([v[1] for v in vertices])

    mask = np.zeros(shape, dtype=bool)
    rr, cc = sk_polygon(ys, xs, shape=shape)
    mask[rr, cc] = True

    return mask
