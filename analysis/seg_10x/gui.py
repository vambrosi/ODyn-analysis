"""Three-phase segmentation GUI, rendered with Bokeh in the notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .state import (
    MERGE_SLIDER_RANGES, PHASE_CURATE, PHASE_MERGE, PHASE_TUNE,
    SLIDER_RANGES, SegmentationState, manual_bounds,
)

DEFAULT_ALPHA = 130

# Distinct hues for ROI labels, cycled. Alpha is applied separately so the
# background stays visible underneath.
_PALETTE = np.array([
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
    (26, 188, 156), (241, 196, 15), (142, 68, 173), (52, 152, 219),
    (231, 76, 60), (46, 204, 113), (155, 89, 182), (241, 90, 34),
    (39, 174, 96), (211, 84, 0), (127, 140, 141),
], dtype=np.uint8)


def _label_rgba(labels: np.ndarray, alpha: int = DEFAULT_ALPHA) -> np.ndarray:
    """Label image -> uint32 RGBA for `figure.image_rgba`, 0 transparent."""

    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    nonzero = labels > 0
    if nonzero.any():
        idx = (labels[nonzero] - 1) % len(_PALETTE)
        rgba[nonzero, :3] = _PALETTE[idx]
        rgba[nonzero, 3] = alpha

    return rgba.view(np.uint32).reshape(h, w)


def _grey_image(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, (1, 99.5))
    return np.nan_to_num(image, nan=float(lo)), float(lo), float(hi)


class SegmentationGUI:
    """Bokeh widgets over a `SegmentationState`."""

    def __init__(self, state: SegmentationState, save_path: str | Path):
        self.state = state
        self.save_path = Path(save_path)
        self.tool = "delete"
        self.poly_verts: list[tuple[int, int]] = []
        self._confirm_revert = False

    # ------------------------------------------------------------------ #
    # Document
    # ------------------------------------------------------------------ #

    def modify_doc(self, doc):
        from bokeh.layouts import column, row
        from bokeh.models import (
            Button, CheckboxGroup, ColumnDataSource, Div,
            LinearColorMapper, RadioButtonGroup, Slider,
        )
        from bokeh.palettes import Greys256
        from bokeh.plotting import figure
        from bokeh.events import Tap

        h, w = self.state.shape
        image, lo, hi = _grey_image(self.state.images[self.state.active])

        self.fig = figure(
            width=760, height=int(760 * h / w) + 30,
            x_range=(0, w), y_range=(0, h),
            tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
            toolbar_location="above",
        )
        self.fig.axis.visible = False
        self.fig.grid.visible = False

        self.bg_source = ColumnDataSource(dict(image=[image]))
        self.fig.image(
            image="image", x=0, y=0, dw=w, dh=h, source=self.bg_source,
            color_mapper=LinearColorMapper(palette=Greys256, low=lo, high=hi),
        )

        self.roi_source = ColumnDataSource(dict(image=[_label_rgba(np.zeros((h, w), int))]))
        self.fig.image_rgba(image="image", x=0, y=0, dw=w, dh=h, source=self.roi_source)

        self.poly_source = ColumnDataSource(dict(x=[], y=[]))
        self.fig.line("x", "y", source=self.poly_source, color="red", line_width=2)
        self.fig.scatter("x", "y", source=self.poly_source, color="red", size=7)

        self.fig.on_event(Tap, self._on_tap)

        # ---- widgets ----
        self.status = Div(text="", width=760)
        self.detail = Div(text="", width=760, styles={"font-size": "88%", "color": "#555"})

        self.w_odor = RadioButtonGroup(
            labels=[repr(k) for k in self.state.keys], active=0, width=300
        )
        self.w_odor.on_change("active", self._on_odor)

        self.w_mode = RadioButtonGroup(
            labels=["watershed", "threshold"],
            active=0 if self.state.shared["mode"] == "watershed" else 1, width=300,
        )
        self.w_mode.on_change("active", self._on_mode)

        self.w_flags = CheckboxGroup(
            labels=["adaptive threshold", "split oversized", "this odor only"],
            active=[i for i, on in enumerate([
                bool(self.state.shared["adaptive_block_px"]),
                bool(self.state.shared["split_oversized"]), False,
            ]) if on],
            width=300,
        )
        self.w_flags.on_change("active", self._on_flags)

        self.sliders = {}
        labels = {
            "blur_sigma_px": "blur sigma px",
            "threshold_pctl": "threshold %", "adaptive_block_px": "adaptive block",
            "min_diameter_px": "min diameter px", "max_diameter_px": "max diameter px",
            "peak_distance_px": "peak distance px", "border_px": "border exclude px",
        }
        for name, text in labels.items():
            low, high, step = SLIDER_RANGES[name]
            slider = Slider(
                start=low, end=high, step=step,
                value=float(self.state.shared[name] or 0), title=text, width=300,
            )
            slider.on_change("value_throttled", self._make_slider_cb(name))
            self.sliders[name] = slider

        self.w_alpha = Slider(
            start=0, end=255, step=5, value=DEFAULT_ALPHA,
            title="ROI opacity", width=300,
        )
        self.w_alpha.on_change("value", lambda a, o, n: self._refresh())

        # Merge controls: live only in the merge phase.
        self.merge_sliders = {}
        for name, text in (
            ("min_overlap", "merge: min overlap"),
            ("min_detections", "merge: min odors detecting"),
            ("consensus_fraction", "merge: consensus fraction"),
        ):
            low, high, step = MERGE_SLIDER_RANGES[name]
            slider = Slider(
                start=low, end=high, step=step,
                value=self.state.merge_params[name], title=text, width=300,
            )
            slider.on_change("value_throttled", self._make_merge_cb(name))
            self.merge_sliders[name] = slider

        self.w_tool = RadioButtonGroup(
            labels=["delete ROI", "add ROI", "exclude polygon"], active=0, width=300
        )
        self.w_tool.on_change("active", self._on_tool)

        self.b_phase = Button(label="Lock -> Merge", button_type="primary", width=145)
        self.b_phase.on_click(self._advance)

        self.b_back = Button(label="Back", width=145)
        self.b_back.on_click(self._go_back)

        self.b_reset = Button(label="Reset", width=300)
        self.b_reset.on_click(self._on_reset)

        self.b_close_poly = Button(label="Close polygon", width=145)
        self.b_close_poly.on_click(self._close_polygon)

        self.b_cancel_poly = Button(label="Cancel polygon", width=145)
        self.b_cancel_poly.on_click(lambda: (self._clear_poly(), self._refresh()))

        self.b_save = Button(
            label="Save local checkpoint (not publish)",
            button_type="success", width=300,
        )
        self.b_save.on_click(self._on_save)

        controls = column(
            Div(text="<b>odor</b>"), self.w_odor,
            Div(text="<b>mode</b>"), self.w_mode,
            self.w_flags,
            *self.sliders.values(),
            self.w_alpha,
            Div(text="<b>merge</b> (phase 2)"), *self.merge_sliders.values(),
            Div(text="<b>curate tool</b> (phase 3)"), self.w_tool,
            row(self.b_close_poly, self.b_cancel_poly),
            row(self.b_phase, self.b_back), self.b_reset, self.b_save,
        )

        doc.add_root(column(self.status, row(self.fig, controls), self.detail))
        self._refresh()

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _this_odor_only(self) -> bool:
        return 2 in self.w_flags.active

    def _make_slider_cb(self, name):
        def callback(attr, old, new):
            self._set(name, new)
        return callback

    def _set(self, name: str, value) -> None:
        if self.state.phase != PHASE_TUNE:
            self._say("Parameters are frozen while curating.")
            return

        if name == "adaptive_block_px" and 0 not in self.w_flags.active:
            value = None
        elif name == "border_px":
            value = int(value)
        elif name != "mode":
            value = float(value)

        try:
            self.state.set_param(name, value, this_odor_only=self._this_odor_only())
        except (RuntimeError, KeyError, ValueError) as error:
            self._say(str(error))
            return

        self._refresh()

    def _on_mode(self, attr, old, new) -> None:
        self._set("mode", self.w_mode.labels[new])

    def _on_flags(self, attr, old, new) -> None:
        if self.state.phase != PHASE_TUNE:
            self._say("Parameters are frozen while curating.")
            return
        block = int(self.sliders["adaptive_block_px"].value) or 101
        self._set("adaptive_block_px", block if 0 in new else None)
        self._set("split_oversized", 1 in new)

    def _on_odor(self, attr, old, new) -> None:
        self.state.active = self.state.keys[new]
        self._refresh()

    def _on_tool(self, attr, old, new) -> None:
        self.tool = ["delete", "add", "exclude"][new]
        self._clear_poly()

    def _make_merge_cb(self, name):
        def callback(attr, old, new):
            if self.state.phase != PHASE_MERGE:
                self._say("Merge parameters are only live in the merge phase.")
                return
            value = int(new) if name == "min_detections" else float(new)
            self.state.set_merge_param(name, value)
            self._refresh()
        return callback

    def _advance(self) -> None:
        if self.state.phase == PHASE_TUNE:
            self.state.begin_merge()
            self._say("Merging. Segmentation parameters frozen.")
        elif self.state.phase == PHASE_MERGE:
            self.state.begin_curation()
            self._say("Curating the merged mask. Everything else frozen.")
        else:
            self._say("Already in the final phase.")
            return
        self._confirm_revert = False
        self._refresh()

    def _go_back(self) -> None:
        try:
            self.state.back()
        except RuntimeError as error:
            if self._confirm_revert:
                self.state.back(discard_curation=True)
                self._confirm_revert = False
            else:
                self._confirm_revert = True
                self._say(f"{error} Press Back again to confirm.")
                return
        self._confirm_revert = False
        self._refresh()

    def _on_reset(self) -> None:
        if self.state.phase == PHASE_CURATE:
            self.state.reset_curation()
        else:
            self.state.clear_override()
        self._refresh()

    def _on_save(self) -> None:
        path = self.state.save(self.save_path)
        self._say(f"Saved {path.name} + {path.with_suffix('.json').name}")

    def _on_tap(self, event) -> None:
        if self.state.phase != PHASE_CURATE:
            self._say("Lock parameters first to curate.")
            return

        # Bokeh y runs bottom-up; the image was drawn from y=0 upward, so the
        # array row is the y coordinate directly.
        y, x = int(round(event.y)), int(round(event.x))
        if not (0 <= y < self.state.shape[0] and 0 <= x < self.state.shape[1]):
            return

        if self.tool == "delete":
            self.state.delete_at(y, x)
        elif self.tool == "add":
            index = self.state.add_at(y, x)
            self._report_seed(index)
        else:
            self.poly_verts.append((y, x))
            self._draw_poly()
            return

        self._refresh()

    def _report_seed(self, index: int) -> None:
        """Say what happened to a seed, rather than letting it vanish silently."""

        # Rebuilding the mask is what populates the rejection record.
        self.state.curated_mask()

        area = getattr(self.state, "_rejected_seeds", {}).get(index)
        if area is None:
            return

        bounds = manual_bounds(self.state.params_for(self.state.active))
        floor = int(np.pi * (bounds["min_diameter_px"] / 2) ** 2)
        self._say(
            f"Seed rejected: only {area} px were free there, below the "
            f"{floor} px manual floor. The spot is most likely already inside "
            f"a neighbouring ROI -- delete that one first, then re-add."
        )

    def _close_polygon(self) -> None:
        if self.state.phase != PHASE_CURATE:
            self._say("Lock parameters first to curate.")
            return

        if len(self.poly_verts) < 3:
            self._say(
                f"A polygon needs at least 3 vertices; {len(self.poly_verts)} placed. "
                "Select 'exclude polygon' and tap the image to add them."
            )
            return

        n = len(self.poly_verts)
        self.state.exclude_polygon(self.poly_verts)
        self._clear_poly()
        self._refresh()
        self._say(f"Excluded a {n}-vertex polygon.")

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def _clear_poly(self) -> None:
        self.poly_verts = []
        self.poly_source.data = dict(x=[], y=[])

    def _draw_poly(self) -> None:
        xs = [v[1] for v in self.poly_verts]
        ys = [v[0] for v in self.poly_verts]
        self.poly_source.data = dict(x=xs + xs[:1], y=ys + ys[:1])

    def _say(self, message: str) -> None:
        self.detail.text = message

    def _refresh(self) -> None:
        phase = self.state.phase
        key = self.state.active
        tuning = phase == PHASE_TUNE

        # Tune shows the odor being tuned; merge and curate show the combined
        # image, since the mask on screen spans every odor and judging it
        # against one odor's map would hide the structure behind half the ROIs.
        image, lo, hi = _grey_image(
            self.state.images[key] if tuning else self.state.combined_image()
        )
        self.bg_source.data = dict(image=[image])

        # Rescale the greyscale mapper: tune and merge show different images.
        for renderer in self.fig.renderers:
            mapper = getattr(renderer.glyph, "color_mapper", None)
            if mapper is not None:
                mapper.low, mapper.high = lo, hi

        if tuning:
            mask = self.state.segment(key)
        elif phase == PHASE_MERGE:
            mask = self.state.merged().labels
        else:
            mask = self.state.curated_mask()

        self.roi_source.data = dict(image=[_label_rgba(mask, int(self.w_alpha.value))])

        areas = np.bincount(mask.ravel())[1:]
        diameters = 2 * np.sqrt(areas / np.pi) if len(areas) else np.array([0])

        self.b_phase.label = {
            PHASE_TUNE: "Lock -> Merge",
            PHASE_MERGE: "Lock -> Curate",
            PHASE_CURATE: "(final phase)",
        }[phase]
        self.b_phase.disabled = phase == PHASE_CURATE
        self.b_back.disabled = tuning

        scope = f"odor <b>{key!r}</b>" if tuning else "<b>merged</b>"
        self.status.text = (
            f"<b>[{phase.upper()}]</b> &nbsp; {scope} &nbsp;&nbsp; "
            f"<b>{int(mask.max())}</b> ROIs &nbsp; median "
            f"<b>{np.median(diameters):.0f}</b> px &nbsp; coverage "
            f"<b>{100 * (mask > 0).mean():.1f}%</b>"
        )

        if tuning:
            params = self.state.params_for(key)
            override = sorted(self.state.overrides.get(key, {}))
            self._say(
                f"thr p{params['threshold_pctl']:.0f} · "
                f"adaptive {params['adaptive_block_px'] or 'off'} · "
                f"diam {params['min_diameter_px']:.0f}-{params['max_diameter_px']:.0f} · "
                f"peak {params['peak_distance_px']:.0f} · border {params['border_px']} · "
                f"split {params['split_oversized']}"
                + (f" &nbsp; <b>OVERRIDES:</b> {override}" if override else "")
                + "  |  sliders to tune, then Lock"
            )
        else:
            merged = self.state.merged()
            detections = merged.detections_per_roi() if merged.n_rois else np.array([])
            mp = self.state.merge_params
            per_odor = " ".join(
                f"{k!r}:{self.state.segment(k).max()}" for k in self.state.keys
            )
            self._say(
                f"per-odor {per_odor} → merged {merged.n_rois} · "
                f"found by &gt;1 odor: {int((detections > 1).sum())} · "
                f"overlap {mp['min_overlap']:.2f} {mp['linkage']}/{mp['metric']} · "
                f"min_detections {mp['min_detections']} · "
                f"consensus {mp['consensus_fraction']:.1f}"
                + ("  |  click to curate" if phase == PHASE_CURATE
                   else "  |  merge sliders live, then Lock")
            )


def launch(images: dict, *, save_path: str | Path, params: None | dict = None):
    """Open the segmentation GUI on `{odor key: image}` inside a notebook."""

    import bokeh.plotting as bpl
    from ..session.bokeh import ensure_notebook_output, stop_notebook_servers

    ensure_notebook_output()

    gui = SegmentationGUI(SegmentationState(images, params), save_path)
    # Use a stable VS Code-visible port after releasing the previous server.
    stop_notebook_servers()
    bpl.show(gui.modify_doc, port=5008)

    return gui
