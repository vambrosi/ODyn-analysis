"""Headless watershed segmentation of glomeruli, per odor."""

from __future__ import annotations

import numpy as np

from skimage.feature import peak_local_max
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import watershed

# Starting points only. Pixel scale varies several-fold between sessions, so
# these must be checked against each session's images rather than trusted.
DEFAULT_MIN_DIAMETER_PX = 20.0
DEFAULT_MAX_DIAMETER_PX = 55.0

GLOM_10X_DEFAULTS = {
    "mode": "watershed",
    "blur_sigma_px": 0.0,
    "threshold_pctl": 60.0,
    "adaptive_block_px": 101,
    "min_diameter_px": 20.0,
    "max_diameter_px": 55.0,
    "peak_distance_px": 20.0,
    "split_oversized": True,
}


# Reference scale used to initialize the GUI for another 10x session.
REFERENCE_UM_PER_PX = 2.0

PEAK_DISTANCE_SCALE_EXPONENT = 0.6


def scale_params(
    params: dict,
    *,
    from_um_per_px: float = REFERENCE_UM_PER_PX,
    to_um_per_px: float,
) -> dict:
    """Carry pixel size parameters from one session's scale to another's."""

    if from_um_per_px <= 0 or to_um_per_px <= 0:
        raise ValueError(
            f"Scales must be positive, got {from_um_per_px} and {to_um_per_px}."
        )

    # More um per pixel means fewer pixels per object, so pixel sizes scale
    # with the inverse ratio.
    factor = from_um_per_px / to_um_per_px

    out = dict(params)

    for name in ("min_diameter_px", "max_diameter_px"):
        if out.get(name) is not None:
            out[name] = round(float(out[name]) * factor, 1)

    if out.get("peak_distance_px") is not None:
        out["peak_distance_px"] = round(
            float(out["peak_distance_px"]) * factor**PEAK_DISTANCE_SCALE_EXPONENT, 1
        )

    # The adaptive window is a background estimator: it should stay a few
    # object-widths across, so it scales with the objects.
    if out.get("adaptive_block_px"):
        block = int(round(out["adaptive_block_px"] * factor))
        out["adaptive_block_px"] = block + 1 if block % 2 == 0 else block

    return out


def _area_bounds_px(min_diameter_px: float, max_diameter_px: float) -> tuple[float, float]:
    """Disk-equivalent area bounds for a diameter range, both in pixels."""

    if min_diameter_px <= 0:
        raise ValueError(f"min_diameter_px must be positive, got {min_diameter_px}.")

    if min_diameter_px >= max_diameter_px:
        raise ValueError(
            f"min_diameter_px ({min_diameter_px}) must be below "
            f"max_diameter_px ({max_diameter_px})."
        )

    return np.pi * (min_diameter_px / 2) ** 2, np.pi * (max_diameter_px / 2) ** 2


def _split_oversized(
    labels: np.ndarray,
    *,
    image: np.ndarray,
    binary: np.ndarray,
    max_area: float,
    peak_distance_px: float,
    threshold: float,
    max_rounds: int = 4,
) -> np.ndarray:
    """Re-seed and re-flood only the regions that came out larger than `max_area`."""

    out = labels.copy()
    next_label = int(out.max()) + 1
    spacing = peak_distance_px

    for _ in range(max_rounds):
        oversized = [r for r in regionprops(out) if r.area > max_area]
        if not oversized:
            break

        spacing = max(2.0, spacing / 2)

        for region in oversized:
            r0, c0, r1, c1 = region.bbox
            sub_mask = out[r0:r1, c0:c1] == region.label
            sub_image = image[r0:r1, c0:c1]

            coords = peak_local_max(
                np.where(np.isfinite(sub_image) & sub_mask, sub_image, -np.inf),
                min_distance=max(1, int(round(spacing))),
                threshold_abs=threshold,
                exclude_border=False,
            )

            # One seed means nothing to split by; leave it for the size filter.
            if len(coords) < 2:
                continue

            seeds = np.zeros(sub_mask.shape, dtype=np.int32)
            for i, (y, x) in enumerate(coords, start=1):
                if sub_mask[y, x]:
                    seeds[y, x] = i

            if seeds.max() < 2:
                continue

            pieces = watershed(
                -np.where(np.isfinite(sub_image), sub_image, -np.inf),
                markers=seeds,
                mask=sub_mask,
            )

            out[r0:r1, c0:c1][sub_mask] = 0
            for piece in range(1, int(pieces.max()) + 1):
                sel = pieces == piece
                if sel.any():
                    out[r0:r1, c0:c1][sel] = next_label
                    next_label += 1

    return out


def segment_image(
    image: np.ndarray,
    *,
    threshold_pctl: float = 90.0,
    mode: str = "watershed",
    blur_sigma_px: float = 0.0,
    min_diameter_px: float = DEFAULT_MIN_DIAMETER_PX,
    max_diameter_px: float = DEFAULT_MAX_DIAMETER_PX,
    peak_distance_px: None | float = None,
    adaptive_block_px: None | int = None,
    adaptive_offset: float = 0.0,
    split_oversized: bool = True,
    exclude_mask: None | np.ndarray = None,
) -> tuple[np.ndarray, dict]:
    """Segment one image into a label mask."""

    if mode not in ("watershed", "threshold"):
        raise ValueError(f"mode must be 'watershed' or 'threshold', got {mode!r}.")

    image = np.asarray(image, dtype=np.float32)

    finite_pixels = np.isfinite(image)

    if not finite_pixels.any():
        raise ValueError("Image has no finite values.")

    if blur_sigma_px:
        # Blurs everything downstream, threshold and seeds alike: the ROI
        # outline is the edge of `image > threshold`, decided pixel by pixel,
        # so smoothing only what watershed floods changes nothing at all.
        # Seeds too, on purpose, so a cluster of bright specks is found once
        # instead of once per speck.
        #
        # NOTE: this moves the intensity distribution, so `threshold_pctl` cuts
        # somewhere else and regions grow. Expect to retune it and the diameter
        # bounds after touching this, rather than treating it as independent.
        from skimage.filters import gaussian

        # Filled first: one NaN would otherwise spread over the whole image,
        # and the non-finite pixels are put back so the checks below still see
        # them. Median rather than zero, so an edge does not appear at the hole.
        filled = np.where(finite_pixels, image, np.median(image[finite_pixels]))
        image = gaussian(filled, sigma=blur_sigma_px, preserve_range=True)
        image = np.where(finite_pixels, image, np.nan).astype(np.float32)

    finite = image[finite_pixels]

    threshold = float(np.percentile(finite, threshold_pctl))
    binary = np.isfinite(image) & (image > threshold)

    if adaptive_block_px:
        # Local threshold: a pixel must stand above its own neighbourhood.
        # This is what recovers glomeruli in the dimmer parts of a field whose
        # brightness varies several-fold across the bulb.
        from skimage.filters import threshold_local

        block = int(adaptive_block_px)
        if block % 2 == 0:
            block += 1  # threshold_local requires an odd window

        filled = np.where(np.isfinite(image), image, threshold)
        local = threshold_local(
            filled, block_size=block, method="gaussian", offset=-adaptive_offset
        )

        binary &= filled > local

    if exclude_mask is not None:
        binary &= ~exclude_mask.astype(bool)

    min_area, max_area = _area_bounds_px(min_diameter_px, max_diameter_px)

    if mode == "threshold" or not binary.any():
        labels = cc_label(binary, connectivity=1)
        n_seeds = 0

    else:
        if peak_distance_px is None:
            peak_distance_px = min_diameter_px

        min_distance = max(1, int(round(peak_distance_px)))

        coords = peak_local_max(
            np.where(np.isfinite(image), image, -np.inf),
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=False,
        )
        n_seeds = len(coords)

        if n_seeds == 0:
            labels = np.zeros(image.shape, dtype=np.int32)
        else:
            seeds = np.zeros(image.shape, dtype=np.int32)
            for i, (y, x) in enumerate(coords, start=1):
                seeds[y, x] = i

            # Flood the inverted correlation landscape: basins grow outward
            # from each peak until they meet, so a boundary lands where the
            # image dips between two glomeruli.
            labels = watershed(
                -np.where(np.isfinite(image), image, -np.inf),
                markers=seeds,
                mask=binary,
            )

    if split_oversized:
        labels = _split_oversized(
            labels,
            image=image,
            binary=binary,
            max_area=max_area,
            peak_distance_px=peak_distance_px or min_diameter_px,
            threshold=threshold,
        )

    kept = [
        region.label
        for region in regionprops(labels)
        if min_area <= region.area <= max_area
    ]

    out = np.zeros(labels.shape, dtype=np.int32)
    for new_label, old_label in enumerate(sorted(kept), start=1):
        out[labels == old_label] = new_label

    params = {
        "image_shape_px": list(image.shape),
        "threshold_pctl": float(threshold_pctl),
        "threshold_value": threshold,
        "mode": mode,
        "min_diameter_px": float(min_diameter_px),
        "max_diameter_px": float(max_diameter_px),
        "min_area_px": round(min_area, 1),
        "max_area_px": round(max_area, 1),
        "peak_distance_px": None if peak_distance_px is None else float(peak_distance_px),
        "adaptive_block_px": None if not adaptive_block_px else int(adaptive_block_px),
        "adaptive_offset": float(adaptive_offset),
        "pixels_above_threshold_pct": round(100 * float(binary.mean()), 2),
        "n_seeds": int(n_seeds),
        "split_oversized": bool(split_oversized),
        "n_before_size_filter": int(labels.max()),
        "n_rois": int(out.max()),
    }

    return out, params


def segment_per_group(images: dict, **kwargs) -> tuple[list[np.ndarray], list, list[dict]]:
    """Segment one image per group key, ready for `merge.merge_masks`."""

    keys = sorted(images, key=repr)

    masks, params = [], []
    for key in keys:
        mask, record = segment_image(images[key], **kwargs)
        record["group_key"] = repr(key)
        masks.append(mask)
        params.append(record)

    return masks, keys, params
