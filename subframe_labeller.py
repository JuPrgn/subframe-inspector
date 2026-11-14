# -*- coding: utf-8 -*-
"""Hand-label the previews of a session, to build ground truth.

subframe_inspector.py sorts a night into categories; this is where those
verdicts get checked against a pair of eyes. It shows the JPG previews in
the order they were shot, says what the detector made of each one, and
records what you saw instead. What comes out is a CSV of labelled frames
-- the thing every threshold in the detector was tuned against, and the
thing there is never enough of.

It reads REVIEW/report.csv and the previews, and writes exactly one file:
<session>/labels.csv. The labels deliberately do *not* live under REVIEW/,
because every run of the detector empties that folder.

Nothing here duplicates the detector: it does not open a FITS, measure a
star or re-derive a verdict. It reads the CSV the detector already wrote,
so the two cannot drift apart. That also keeps the dependencies down to
tkinter and Pillow -- no numpy, no astropy -- so it runs on whatever
interpreter happens to be on PATH.

    python subframe_labeller.py "LBN 550_sub"
    python subframe_labeller.py                # opens the session list
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from session_layout import (LABELLER_NAME, configured_layout, load_config,
                            resolve, session_root)

# The console output uses emoji. On a Windows console still running
# cp1252 the very first print would abort with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ======================================================================
#  Vocabulary
# ======================================================================

@dataclass(frozen=True)
class Category:
    """One button: what gets written down, and what the eye looks for."""
    key: str
    caption: str
    shortcut: str
    hint: str


CLEAN = "Clean"

# The detector's own category names, so a label and a verdict can be
# compared without a translation table. Two are not folders of its own.
# 'Clean' is the one the corpus most lacks: there is no solid negative
# anywhere in it, which is what keeps precision unmeasurable on
# unlabelled frames. 'Dew' -- fog on the lens -- the detector cannot yet
# tell from cloud: its veil rule catches it and files it under
# Clouds_Haze, so the two compare as agreeing (LABEL_TO_DETECTOR below).
# Keeping the tick apart is what will let the two be separated later.
#
# A 'Doubling' button was here and has been removed. A jolt that prints
# the field twice is a real and distinct fault -- the detector measures
# it and reports doubling_px -- but it is not a category of its own: the
# detector files it under Trailing, and the owner ruled the same way.
# Over nine sessions and 648 labelled frames it was never once ticked.
CATEGORIES = (
    Category(CLEAN, "Clean — nothing wrong", "1", "the frame is good"),
    Category("Trailing", "Trailing", "2", "stars drawn out, or printed twice"),
    Category("Clouds_Haze", "Clouds / haze", "3", "sky washed out"),
    Category("Satellites", "Satellite", "4", "thin, regular straight track"),
    Category("Meteors", "Meteor", "5", "bright head, fading tail, glow"),
    Category("Aircraft", "Aircraft", "6", "wide track, evenly lit"),
    Category("Dew", "Dew", "7", "fog on the lens: halos, washed-out field"),
)

# How a label reads against a verdict. 'Doubling' is kept although
# nothing writes it any more: a labels.csv written before the button
# went must still compare as agreeing with a Trailing verdict. 'Dew' is
# reported by the detector as Clouds_Haze, see above. replay_cases.py
# carries the same aliases for the same reasons.
LABEL_TO_DETECTOR = {"Doubling": "Trailing", "Dew": "Clouds_Haze"}

# The three images a frame can have, in the order D walks through them.
SOURCE_PREVIEW = "JPG preview"
SOURCE_REVIEW = "REVIEW copy"
SOURCE_FITS = "FITS"

LABELS_FILENAME = "labels.csv"
LABEL_FIELDS = ("frame", "labels", "note", "detected", "updated")

# Everything a frame's name can end with. Used instead of splitext, see
# base_name.
FRAME_EXTENSIONS = (".fit", ".fits", ".jpg", ".jpeg")

# The measurements worth having on screen while judging a frame, how to
# phrase them, and how small a value stops being worth reading. Below
# that the entry is dropped rather than shown as a zero -- the detector
# measured no track, no doubling, no haze worth the name. Everything
# else stays in report.csv.
REPORT_READOUT = (
    ("elongation_ratio", "elong {:.2f}x", 0.0),
    ("doubling_px", "doubling {:.0f}px", 0.5),
    ("haze_sigma", "haze {:.1f}σ", 0.05),
    ("track_span", "track {:.0f}px", 0.5),
    ("track_rms_width", "width {:.2f}px", 0.005),
    ("track_green", "green {:.1f}σ", 1.0),
    ("shift_px", "shift {:.0f}px", 0.5),
)

# The timestamp the Seestar puts in every file name, e.g.
# 'Light_LBN 550_60.0s_IRCUT_20260830-002717'. Sorting on it rather than
# on the whole name keeps the session in shooting order even when the
# filter or the exposure changed mid-night and the prefix with it.
SHOT_TIME = re.compile(r"(\d{8})-(\d{6})")


# ======================================================================
#  Session
# ======================================================================

@dataclass(frozen=True)
class FilterView:
    """One entry in the filter menu.

    `hide_when_empty` is for the views over one's own labels: a category
    nobody has ticked yet is noise in the list, and it appears by itself
    the moment it holds something, since the menu is filled when opened.
    """
    name: str
    keep: Callable[["Frame"], bool]
    hide_when_empty: bool = False


@dataclass
class Frame:
    """One frame: the images available for it, and its verdict.

    Three of them, any of which may be missing: the untouched preview in
    JPG/, the copy REVIEW filed (which carries the --debug drawings),
    and the FITS itself.
    """
    name: str
    preview: str | None
    annotated: str | None
    fits: str | None
    detected: tuple[str, ...]
    readout: str


@dataclass
class Label:
    """One line of labels.csv: what the eye saw, and when."""
    frame: str
    labels: tuple[str, ...]
    note: str
    detected: str
    updated: str


def shot_order(name: str) -> tuple[str, str]:
    """Sort key placing a frame by when it was shot, name as tiebreak."""
    match = SHOT_TIME.search(name)
    return (match.group(0) if match else "", name)


def short_label(name: str) -> str:
    """The part of a frame name that actually tells it from the others.

    Every frame in a session shares the target, the exposure and the
    filter; only the timestamp differs, and that is what anyone reads
    off a frame or types into a search box.
    """
    match = SHOT_TIME.search(name)
    return match.group(0) if match else name


def base_name(filename: str) -> str:
    """Drop the extension, so a .fit row and its .jpg preview agree.

    Only the extensions actually in play are stripped, and splitext is
    not used: a Seestar name carries its exposure as '60.0s', so
    splitext takes '.0s_IRCUT_20260830-000305' for an extension and
    leaves 'Light_LBN 550_60'. Frame names in labels.csv have no
    extension at all, which is exactly where that bites.
    """
    stem = os.path.basename(filename)
    lowered = stem.lower()
    for extension in FRAME_EXTENSIONS:
        if lowered.endswith(extension):
            return stem[:-len(extension)]
    return stem


def format_readout(row: dict[str, str]) -> str:
    """The handful of measurements worth seeing beside the image."""
    parts = []
    for column, template, minimum in REPORT_READOUT:
        try:
            value = float(row.get(column, "") or 0.0)
        except ValueError:
            continue
        # A green glow below the sky's own colour is noise, not a finding;
        # showing it as "green -2.6σ" only gets in the way of spotting
        # the real ones.
        if column == "track_green" and value < 0:
            continue
        if abs(value) > minimum:
            parts.append(template.format(value))
    return " · ".join(parts)


def read_report(session: str) -> dict[str, dict[str, str]]:
    """Read REVIEW/report.csv, keyed by frame name without extension.

    Missing is not an error: the session may simply not have been run
    through the detector yet, and labelling from scratch is legitimate.
    """
    path = os.path.join(session, "REVIEW", "report.csv")
    if not os.path.isfile(path):
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {base_name(row["frame"]): row for row in csv.DictReader(handle)
                if row.get("frame")}


def find_previews(folder: str) -> dict[str, str]:
    """Map frame name to JPG path for one folder, empty if it is absent."""
    if not os.path.isdir(folder):
        return {}
    return {base_name(name): os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if name.lower().endswith((".jpg", ".jpeg"))}


def build_frames(session: str) -> list[Frame]:
    """Assemble the session: every frame, every image of it, its verdict.

    The frame list is the union of what is there -- report.csv when
    there is one, because it holds the clean frames too, plus the
    previews and the FITS, found with the configured layout. A session
    with no previews is perfectly workable: the frames are read from the
    FITS instead.
    """
    found = resolve(session, configured_layout())
    report = read_report(session)
    previews = found.previews
    raw = {base_name(path): path for path in found.frames}

    annotated: dict[str, str] = {}
    review = os.path.join(session, "REVIEW")
    if os.path.isdir(review):
        for entry in sorted(os.listdir(review)):
            annotated.update(find_previews(os.path.join(review, entry)))

    names = set(report) | set(previews) | set(raw) | set(annotated)

    frames = []
    for name in sorted(names, key=shot_order):
        row = report.get(name, {})
        frames.append(Frame(
            name=name,
            preview=previews.get(name),
            annotated=annotated.get(name),
            fits=raw.get(name),
            detected=tuple((row.get("categories") or "").split()),
            readout=format_readout(row),
        ))
    return frames


# ======================================================================
#  Reading a FITS
# ======================================================================

# A raw frame is linear and 16-bit; shown as it is, it is black. These
# are the numbers of the usual screen stretch: the black point is put
# 2.8 robust deviations below the sky, and the midtone curve is set so
# the sky itself lands here on a 0..1 scale. Measured against the
# Seestar's own JPG, 0.16 puts the two within a shade of each other.
FITS_SHADOW_CLIP = 2.8
FITS_BACKGROUND = 0.16

_FITS_READY: bool | None = None


def fits_available() -> bool:
    """Whether FITS can be read here, decided once.

    numpy and astropy are what the detector needs anyway, but this tool
    is meant to run on whatever interpreter is at hand, so their absence
    costs the FITS source and nothing else.
    """
    global _FITS_READY
    if _FITS_READY is None:
        try:
            import numpy  # noqa: F401
            from astropy.io import fits  # noqa: F401
            _FITS_READY = True
        except ImportError:
            _FITS_READY = False
    return _FITS_READY


def read_fits(path: str):
    """Read a raw frame and stretch it for the screen.

    Two things make this as cheap as decoding the JPEG rather than twice
    its price, which is what a naive version measured. The statistics
    come from every fourth pixel in each direction, and for 16-bit data
    -- which is everything a Seestar writes -- the curve is applied
    through a 65536-entry lookup table rather than in floating point
    over eight million pixels. Measured on a 2160x3840 frame: 57 ms,
    against 77 ms to decode the matching JPEG.

    What comes out is greyscale, and on a Seestar it is a Bayer mosaic,
    so the sky carries a fine checkerboard the JPG does not have. Star
    shapes and trails read the same on both -- which is what this view
    is for -- but it is not a debayered picture and does not pretend to
    be one.
    """
    import numpy as np
    from astropy.io import fits
    from PIL import Image

    data = fits.getdata(path)
    if data.ndim == 3:
        # A colour cube, whichever way round it is stored.
        data = data[0] if data.shape[0] <= 4 else data[..., 0]

    sample = np.asarray(data[::4, ::4], dtype=np.float32)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median))) * 1.4826 or 1.0
    black = median - FITS_SHADOW_CLIP * mad
    white = float(sample.max())
    span = max(white - black, 1.0)

    # The midtone that lands the sky on FITS_BACKGROUND.
    sky = min(max((median - black) / span, 1e-4), 0.9999)
    midtone = sky * (1 - FITS_BACKGROUND) / (
        sky * (1 - 2 * FITS_BACKGROUND) + FITS_BACKGROUND - sky)
    midtone = min(max(midtone, 1e-4), 0.9999)

    def curve(values):
        scaled = np.clip((values - black) / span, 0.0, 1.0)
        shaped = ((midtone - 1) * scaled) / ((2 * midtone - 1) * scaled - midtone)
        return (np.clip(shaped, 0.0, 1.0) * 255).astype(np.uint8)

    if data.dtype == np.uint16:
        lookup = curve(np.arange(65536, dtype=np.float32))
        return Image.fromarray(lookup[data], mode="L")
    return Image.fromarray(curve(np.asarray(data, dtype=np.float32)), mode="L")


# ======================================================================
#  The labels file
# ======================================================================

def load_labels(path: str) -> dict[str, Label]:
    """Read labels.csv, so a session can be labelled over several days."""
    if not os.path.isfile(path):
        return {}
    labels = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = base_name(row.get("frame", "") or "")
            if not name:
                continue
            labels[name] = Label(
                frame=name,
                labels=tuple((row.get("labels") or "").split()),
                note=row.get("note", "") or "",
                detected=row.get("detected", "") or "",
                updated=row.get("updated", "") or "",
            )
    return labels


def save_labels(path: str, labels: dict[str, Label]) -> None:
    """Rewrite labels.csv, via a temporary file so it is never truncated.

    The whole file goes out on every validation. It is a few hundred
    short rows, and a full rewrite through os.replace is the one form of
    update that cannot leave a half-written line behind if the tool is
    closed mid-save.
    """
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_FIELDS)
        for name in sorted(labels, key=shot_order):
            entry = labels[name]
            writer.writerow([entry.frame, " ".join(entry.labels), entry.note,
                             entry.detected, entry.updated])
    os.replace(temporary, path)


def as_detector_categories(labels: Sequence[str]) -> set[str]:
    """The labels in the detector's own vocabulary, for comparing them."""
    return {LABEL_TO_DETECTOR.get(key, key) for key in labels if key != CLEAN}


# ======================================================================
#  The window
# ======================================================================

BACKGROUND = "#161618"
PANEL = "#232327"
TEXT = "#e8e8ea"
MUTED = "#93939c"
ACCENT = "#5aa9ff"
FLAGGED = "#ffb454"
GOOD = "#7bd88f"

SIDE_WIDTH = 340
SIDE_PADDING = 14

# The close-up beside the image. Judging trailing means looking at star
# shapes, and star shapes are a few pixels across on a preview the whole
# of which is 3840 pixels tall -- so the frame is shown twice, once whole
# and once magnified. It gets a column of its own rather than a corner of
# the panel: it is the thing being looked at, not an accessory.
LOUPE_WIDTH = 460

# 1:1 by default, which is already 4x the scale the whole frame is shown
# at. Looked at side by side on a trailed frame: at 1:1 the close-up
# holds about eight stars, all leaning the same way and unmistakable; at
# 2:1 it holds two, which is a shape rather than a pattern. More zoom is
# a keypress away when one star needs a closer look.
LOUPE_ZOOM = 1.0
LOUPE_ZOOM_LIMITS = (0.5, 12.0)
LOUPE_STEP = 1.5

# How far the pointer may travel and still count as a click rather than a
# drag. Placing the close-up and panning share the left button, and a
# click is never perfectly still.
CLICK_SLOP = 4

# The bar along the bottom of the image. Paging through a category is
# done with the eyes on the picture, not on the panel, so how far along
# it is has to be visible from there -- and running off the end has to be
# impossible to miss. Wrapping silently back to the first frame, which is
# what this replaces, looked exactly like carrying on.
# Tall enough to read as a control rather than as a hairline, and to
# take a handle. At 5 px it was neither: nothing about it said it could
# be dragged, which is the only way anyone finds out that it can.
PROGRESS_HEIGHT = 12
PROGRESS_THUMB = 7

# The grab zone stays a little wider than the paint.
SCRUB_GRAB = PROGRESS_HEIGHT + 6

# Page Up / Page Down. Ten is about a screenful of thinking: far enough
# to be worth a key, near enough to still know where you are.
JUMP_STEP = 10

# How many suggestions the search box offers. Enough to cover the frames
# a couple of typed digits narrow it to, short enough to read at a
# glance; the box says so when there are more.
SEARCH_LIMIT = 12
PROGRESS_TRACK = "#31313a"
PROGRESS_LABELLED = "#7bd88f"

ZOOM_STEP = 1.25
ZOOM_MAX = 16.0

# Decoded previews kept in hand, so stepping back and forth over the same
# few frames is instant. A Seestar preview is 2160x3840, which is 25 MB
# once decoded, so this is not a free number to raise.
CACHE_SIZE = 4


class Labeller:
    """The whole interface: an image, seven boxes and a note."""

    def __init__(self, master, session: str, frames: list[Frame],
                 labels_path: str, labels: dict[str, Label]) -> None:
        from PIL import Image, ImageDraw, ImageTk
        self._Image = Image
        self._ImageDraw = ImageDraw
        self._ImageTk = ImageTk

        self.master = master
        self.session = session
        self.frames = frames
        self.labels_path = labels_path
        self.labels = labels

        self.filters = self._build_filters()
        self.filter_predicates = {view.name: view.keep
                                  for view in self.filters if view}
        self.visible: list[Frame] = []
        self.index = 0

        self.image = None
        self.photo = None
        self.cache: dict[str, object] = {}
        self.loupe_photo = None
        self.loupe_zoom = LOUPE_ZOOM
        # Where the close-up looks, in image coordinates. None means the
        # centre of the frame. It is deliberately kept across frames: the
        # field barely moves between exposures, so holding one spot is
        # what lets two frames be compared at all.
        self.loupe_centre: tuple[float, float] | None = None
        self.zoom = 1.0
        self.centre = (0.0, 0.0)
        # Whether the view is still the automatic fit. It is, until the
        # first zoom or pan, and while it is the fit follows the window:
        # the canvas has no size at all when the first frame is loaded,
        # so without this the session opens at a zoom computed against a
        # one-pixel canvas.
        self.fitted = True
        # Which of the three images to show. Sticky across frames, and
        # fallen back on per frame, so a session where only some subs
        # have a JPG still pages through without a gap.
        self.source = SOURCE_PREVIEW
        self.lock_view = False
        self.drag_from: tuple[int, int] | None = None
        self.press_at: tuple[int, int] | None = None
        # Where the scrubber is being dragged to. The frame is loaded on
        # release rather than on every step: dragging across a session
        # would otherwise decode a couple of hundred images to show two.
        self.scrub_to: int | None = None
        self.cursor = ""

        self._build_widgets()
        self._apply_filter()

    # -- layout ---------------------------------------------------------

    def _build_widgets(self) -> None:
        import tkinter as tk

        self.master.configure(bg=BACKGROUND)
        self.master.title(f"{LABELLER_NAME} — {os.path.basename(self.session)}")
        self.master.geometry("1560x980")

        top = tk.Frame(self.master, bg=PANEL, padx=10, pady=6)
        top.pack(side="top", fill="x")

        # Built by hand rather than from OptionMenu's own list, for the
        # separators between the groups: the list is long enough that
        # 'what it flagged' and 'what I ticked' need to read as two
        # halves. postcommand rather than a rebuild on every label
        # change, so the counts are worked out only when the menu is
        # actually opened.
        self.filter_var = tk.StringVar(value=self.filters[0].name)
        self.filter_menu = tk.OptionMenu(top, self.filter_var, self.filters[0].name)
        self.filter_menu.configure(bg=PANEL, fg=TEXT, activebackground=ACCENT,
                                   width=24, highlightthickness=0, bd=0,
                                   anchor="w", takefocus=0)
        self.filter_menu["menu"].configure(bg=PANEL, fg=TEXT,
                                           activebackground=ACCENT,
                                           font=("Consolas", 9),
                                           postcommand=self._refresh_filter_menu)
        self.filter_menu.pack(side="left")
        self._refresh_filter_menu()

        self.position_label = tk.Label(top, text="", bg=PANEL, fg=MUTED)
        self.position_label.pack(side="left", padx=14)

        self.frame_label = tk.Label(top, text="", bg=PANEL, fg=TEXT,
                                    font=("Consolas", 10))
        self.frame_label.pack(side="left")

        # A menu rather than the plain label it replaces: D cycles the
        # source, but a shortcut you have to know about is not an
        # interface. Its entries are rebuilt per frame, so it only ever
        # offers what that frame actually has.
        self.source_var = tk.StringVar(value=SOURCE_PREVIEW)
        self.source_menu = tk.OptionMenu(top, self.source_var, SOURCE_PREVIEW)
        self.source_menu.configure(bg=PANEL, fg=TEXT, activebackground=ACCENT,
                                   width=12, highlightthickness=0, bd=0,
                                   anchor="w", takefocus=0)
        self.source_menu["menu"].configure(bg=PANEL, fg=TEXT,
                                           activebackground=ACCENT)
        self.source_menu.pack(side="left", padx=12)

        self.readout_label = tk.Label(top, text="", bg=PANEL, fg=MUTED)
        self.readout_label.pack(side="right")

        # Packed after the readout, so it lands to its left. What anyone
        # types here is a time -- '0040' -- because that is the only
        # part of a frame name that differs from its neighbours.
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(top, textvariable=self.search_var, width=16,
                                     bg=BACKGROUND, fg=TEXT, insertbackground=TEXT,
                                     relief="flat", highlightthickness=1,
                                     highlightbackground="#3a3a40",
                                     highlightcolor=ACCENT)
        self.search_entry.pack(side="right", ipady=2)
        tk.Label(top, text="find  (Ctrl+F)", bg=PANEL, fg=MUTED).pack(
            side="right", padx=(16, 6))

        # Placed rather than packed: it hangs over the image, and only
        # while it is being used.
        self.suggestions = tk.Listbox(
            self.master, bg=PANEL, fg=TEXT, selectbackground=ACCENT,
            selectforeground="#101014", relief="flat", highlightthickness=1,
            highlightbackground=ACCENT, font=("Consolas", 10),
            activestyle="none", takefocus=0)
        self.suggestion_frames: list[Frame] = []

        help_bar = tk.Label(
            self.master,
            text=("←/→ frames   ·   1-7 categories   ·   "
                  "0 clear   ·   A accept detector   ·   "
                  "Enter validate + next   ·   "
                  "wheel zoom   ·   drag pan   ·   F fit   ·   "
                  "click / [ ] / C close-up   ·   D image source   ·   "
                  "L lock view   ·   Q quit\n"
                  "Home/End first, last   ·   PgUp/PgDn ±10   ·   "
                  "drag the bar at the bottom to scrub   ·   "
                  "Ctrl+F find a frame by its time"),
            bg=PANEL, fg=MUTED, pady=5, justify="left")
        help_bar.pack(side="bottom", fill="x")

        side = tk.Frame(self.master, bg=PANEL, padx=SIDE_PADDING, pady=12,
                        width=SIDE_WIDTH)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="DETECTOR", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.detected_label = tk.Label(side, text="", bg=PANEL, fg=TEXT,
                                       wraplength=300, justify="left")
        self.detected_label.pack(anchor="w", pady=(0, 14))

        tk.Label(side, text="WHAT I SEE", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

        self.checks: dict[str, tk.BooleanVar] = {}
        for category in CATEGORIES:
            variable = tk.BooleanVar(value=False)
            self.checks[category.key] = variable
            tk.Checkbutton(
                side, text=f"{category.shortcut}   {category.caption}",
                variable=variable, bg=PANEL, fg=TEXT, selectcolor=BACKGROUND,
                activebackground=PANEL, activeforeground=ACCENT, anchor="w",
                highlightthickness=0, bd=0, takefocus=0,
                command=lambda key=category.key: self._exclusive(key),
            ).pack(anchor="w", fill="x")
            tk.Label(side, text=f"        {category.hint}", bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 4))

        tk.Label(side, text="NOTE", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(10, 0))
        self.note_var = tk.StringVar()
        self.note_entry = tk.Entry(side, textvariable=self.note_var, bg=BACKGROUND,
                                   fg=TEXT, insertbackground=TEXT, relief="flat",
                                   highlightthickness=1,
                                   highlightbackground="#3a3a40",
                                   highlightcolor=ACCENT)
        self.note_entry.pack(fill="x", pady=(2, 12), ipady=4)

        # takefocus=0 throughout: every control here has a key of its
        # own, and a button holding the keyboard focus would answer Enter
        # itself as well as through the binding, validating twice.
        tk.Button(side, text="Validate  (Enter)", command=self.validate,
                  bg=ACCENT, fg="#101014", activebackground=ACCENT,
                  relief="flat", pady=6, takefocus=0).pack(fill="x")
        tk.Button(side, text="Clear this frame's label", command=self.clear_label,
                  bg=PANEL, fg=MUTED, activebackground=PANEL, relief="flat",
                  pady=4, takefocus=0).pack(fill="x", pady=(6, 0))

        self.status_label = tk.Label(side, text="", bg=PANEL, fg=MUTED,
                                     wraplength=300, justify="left")
        self.status_label.pack(anchor="w", pady=(14, 0))

        self.count_label = tk.Label(side, text="", bg=PANEL, fg=MUTED,
                                    wraplength=300, justify="left")
        self.count_label.pack(side="bottom", anchor="w")

        # Packed before the main canvas and after the panel, so it sits
        # between the two: image on the left, close-up beside it, the
        # controls on the right.
        column = tk.Frame(self.master, bg=PANEL, width=LOUPE_WIDTH,
                          padx=SIDE_PADDING, pady=12)
        column.pack(side="right", fill="y")
        column.pack_propagate(False)
        self.loupe_title = tk.Label(column, text="", bg=PANEL, fg=MUTED,
                                    font=("Segoe UI", 8, "bold"))
        self.loupe_title.pack(anchor="w")
        self.loupe_canvas = tk.Canvas(column, bg=BACKGROUND,
                                      highlightthickness=0)
        self.loupe_canvas.pack(fill="both", expand=True, pady=(2, 0))
        self.loupe_item = self.loupe_canvas.create_image(0, 0, anchor="nw")

        self.canvas = tk.Canvas(self.master, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw")
        # Drawn over the image area rather than in the panel: at the end
        # of a list there is no image to look at, and this is where the
        # eyes already are.
        self.end_title = self.canvas.create_text(
            0, 0, text="", fill=ACCENT, font=("Segoe UI", 21, "bold"))
        self.end_detail = self.canvas.create_text(
            0, 0, text="", fill=TEXT, font=("Segoe UI", 11))
        self.end_hint = self.canvas.create_text(
            0, 0, text="", fill=MUTED, font=("Segoe UI", 10))

        self._bind_keys()

    def _bind_keys(self) -> None:
        master = self.master
        master.bind("<Left>", lambda _: self._guard(self.previous))
        master.bind("<Right>", lambda _: self._guard(self.next))
        master.bind("<Return>", lambda _: self.validate())
        master.bind("<KP_Enter>", lambda _: self.validate())
        master.bind("<Escape>", lambda _: master.focus_set())
        for category in CATEGORIES:
            master.bind(category.shortcut,
                        lambda _, key=category.key: self._guard(self._check, key))
        master.bind("0", lambda _: self._guard(self.clear_ticks))
        for key, action in (("a", self.accept_detector), ("f", self.fit),
                            ("d", self.cycle_source), ("l", self.toggle_lock),
                            ("q", self.quit)):
            master.bind(key, lambda _, action=action: self._guard(action))
            master.bind(key.upper(), lambda _, action=action: self._guard(action))
        master.bind("+", lambda _: self._guard(self._zoom_by, ZOOM_STEP))
        master.bind("-", lambda _: self._guard(self._zoom_by, 1 / ZOOM_STEP))
        master.bind("<Home>", lambda _: self._guard(self.go_to, 0))
        master.bind("<End>",
                    lambda _: self._guard(self.go_to, len(self.visible) - 1))
        master.bind("<Prior>", lambda _: self._guard(self.jump, -JUMP_STEP))
        master.bind("<Next>", lambda _: self._guard(self.jump, JUMP_STEP))
        master.bind("<Control-f>", lambda _: (self.focus_search(), "break")[1])
        master.bind("<Control-F>", lambda _: (self.focus_search(), "break")[1])

        # Bound on the widgets themselves and returning "break", so that
        # Return searches instead of validating while the box has focus.
        self.search_entry.bind("<KeyRelease>", self._update_suggestions)
        self.search_entry.bind("<Return>", self._take_suggestion)
        self.search_entry.bind("<Down>", lambda _: self._move_in_suggestions(1))
        self.search_entry.bind("<Up>", lambda _: self._move_in_suggestions(-1))
        self.search_entry.bind("<Escape>", self._leave_search)
        self.suggestions.bind("<ButtonRelease-1>", self._take_suggestion)

        master.bind("[", lambda _: self._guard(self._zoom_loupe, 1 / LOUPE_STEP))
        master.bind("]", lambda _: self._guard(self._zoom_loupe, LOUPE_STEP))
        master.bind("c", lambda _: self._guard(self.centre_loupe))
        master.bind("C", lambda _: self._guard(self.centre_loupe))

        self.canvas.bind("<Configure>", lambda _: self.fit() if self.fitted
                         else self.render())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.loupe_canvas.bind("<Configure>", lambda _: self.render_loupe())
        self.loupe_canvas.bind("<MouseWheel>", self._on_loupe_wheel)

    def _guard(self, action: Callable, *arguments) -> str:
        """Let the note field keep its own keystrokes.

        Every shortcut is a bare letter or digit, so without this a note
        typed into the entry would page through the session instead.
        """
        if self.master.focus_get() in (self.note_entry, self.search_entry):
            return ""
        action(*arguments)
        return "break"

    # -- finding a frame --------------------------------------------------

    def matching_frames(self, query: str) -> list[Frame]:
        """The frames a search string points at.

        Matched on the timestamp and on the time alone, not only on the
        whole name: every frame of a session shares its first thirty
        characters, so a prefix of the full name is no use to anybody.
        Typing '0040' finds 20260830-004055.
        """
        query = query.strip().lower()
        if not query:
            return []
        hits = []
        for frame in self.frames:
            stamp = short_label(frame.name).lower()
            if (stamp.startswith(query)
                    or stamp.rpartition("-")[2].startswith(query)
                    or frame.name.lower().startswith(query)):
                hits.append(frame)
        return hits

    def _update_suggestions(self, _event=None) -> None:
        """Offer what the typed string points at, as it is typed."""
        hits = self.matching_frames(self.search_var.get())
        if not hits:
            self._hide_suggestions()
            return
        self.suggestion_frames = hits[:SEARCH_LIMIT]
        self.suggestions.delete(0, "end")
        for frame in self.suggestion_frames:
            entry = self.labels.get(frame.name)
            flagged = " ".join(frame.detected) or "-"
            mine = " ".join(entry.labels) if entry else ""
            self.suggestions.insert(
                "end", f"{short_label(frame.name)}   {flagged:<22}"
                       + (f"✔ {mine}" if mine else ""))
        if len(hits) > SEARCH_LIMIT:
            self.suggestions.insert("end", f"   … {len(hits) - SEARCH_LIMIT} more")
        self.suggestions.selection_clear(0, "end")
        self.suggestions.selection_set(0)

        self.master.update_idletasks()
        widget = self.search_entry
        # Sized in rows and left to ask for its own height: guessing it
        # in pixels means guessing the font, and the last row went under
        # the edge. Held inside the window, since the box hangs off a
        # field that sits near the right of it.
        self.suggestions.configure(height=self.suggestions.size())
        width = max(widget.winfo_width(), 300)
        left = widget.winfo_rootx() - self.master.winfo_rootx()
        self.suggestions.place(
            x=min(left, max(self.master.winfo_width() - width, 0)),
            y=widget.winfo_rooty() - self.master.winfo_rooty()
              + widget.winfo_height() + 2,
            width=width)
        self.suggestions.lift()

    def _hide_suggestions(self) -> None:
        self.suggestions.place_forget()
        self.suggestion_frames = []

    def _move_in_suggestions(self, step: int) -> str:
        if not self.suggestion_frames:
            return "break"
        current = self.suggestions.curselection()
        position = (current[0] if current else 0) + step
        position = min(max(position, 0), len(self.suggestion_frames) - 1)
        self.suggestions.selection_clear(0, "end")
        self.suggestions.selection_set(position)
        self.suggestions.see(position)
        return "break"

    def _take_suggestion(self, _event=None) -> str:
        """Go to the highlighted frame, or to the first one offered."""
        if not self.suggestion_frames:
            return "break"
        current = self.suggestions.curselection()
        position = current[0] if current else 0
        if position >= len(self.suggestion_frames):    # the '… n more' line
            return "break"
        self.jump_to(self.suggestion_frames[position])
        self.search_var.set("")
        self._hide_suggestions()
        self.master.focus_set()
        return "break"

    def jump_to(self, target: Frame) -> None:
        """Show a frame, widening the filter if the view does not hold it.

        A search that answered 'not in this list' would be a search that
        makes you work out which list to be in first, which is the thing
        it is there to save.
        """
        position = self._position_of(target)
        widened = position is None
        if widened:
            self._pick_filter(self.filters[0].name)
            position = self._position_of(target) or 0
        self.index = position
        self.load_frame()
        if widened:
            # After load_frame, which writes the status line itself.
            self.status_label.configure(
                text="Filter widened to All frames to reach that frame.",
                fg=MUTED)

    def _position_of(self, target: Frame) -> int | None:
        return next((position for position, frame in enumerate(self.visible)
                     if frame.name == target.name), None)

    def _leave_search(self, _event=None) -> str:
        self.search_var.set("")
        self._hide_suggestions()
        self.master.focus_set()
        return "break"

    def focus_search(self) -> None:
        self.search_entry.focus_set()
        self.search_entry.select_range(0, "end")

    # -- filtering ------------------------------------------------------

    def _build_filters(self) -> list:
        """The views over the session: what the detector said, what I said.

        The two halves are named apart and kept apart, because they are
        different questions -- 'what did it flag as Trailing' and 'what
        did I tick Trailing' are different lists, and a pass made over
        one thinking it was the other is a pass wasted. A None in the
        list is a separator in the menu.

        The detector's categories are only those the session holds; mine
        appear as I fill them, so the list stays as short as what I have
        actually done.
        """
        present = sorted({category for frame in self.frames
                          for category in frame.detected})
        items: list = [
            FilterView("All frames", lambda frame: True),
            None,
            FilterView("Flagged by the detector", lambda frame: bool(frame.detected)),
        ]
        items += [FilterView(f"Flagged: {category}",
                             lambda frame, category=category:
                             category in frame.detected)
                  for category in present]
        items += [
            FilterView("Clean for the detector", lambda frame: not frame.detected),
            None,
            FilterView("Labelled by me", lambda frame: frame.name in self.labels),
            FilterView("Not labelled yet", lambda frame: frame.name not in self.labels),
        ]
        for category in CATEGORIES:
            caption = ("Clean for me" if category.key == CLEAN
                       else f"Mine: {category.key}")
            items.append(FilterView(
                caption,
                lambda frame, key=category.key: self._i_said(frame, key),
                hide_when_empty=True))
        items += [
            None,
            FilterView("Detector and I disagree", self._disagrees),
        ]
        return items

    def _i_said(self, frame: Frame, key: str) -> bool:
        entry = self.labels.get(frame.name)
        return entry is not None and key in entry.labels

    def _refresh_filter_menu(self) -> None:
        """Fill the filter menu, each view with how many frames it holds.

        The count is what turns a list of names into something you can
        choose from: 'Mine: Satellites (7)' answers the question before
        you pick it, and a view holding nothing says so without costing
        a pass through the session.
        """
        menu = self.filter_menu["menu"]
        menu.delete(0, "end")
        pending_separator = False
        for view in self.filters:
            if view is None:
                pending_separator = True
                continue
            total = sum(1 for frame in self.frames if view.keep(frame))
            if not total and view.hide_when_empty:
                continue
            if pending_separator and menu.index("end") is not None:
                menu.add_separator()
            pending_separator = False
            menu.add_command(
                label=f"{view.name:<26}{total:>4}",
                command=lambda name=view.name: self._pick_filter(name))

    def _pick_filter(self, name: str) -> None:
        self.filter_var.set(name)
        self._apply_filter()

    def _disagrees(self, frame: Frame) -> bool:
        entry = self.labels.get(frame.name)
        if entry is None:
            return False
        return as_detector_categories(entry.labels) != set(frame.detected)

    def _apply_filter(self) -> None:
        """Re-select the visible frames, staying on the current one if it survives."""
        # self.frame, not self.visible[self.index]: the index sits one
        # past the last frame when the end card is showing.
        current = self.frame.name if self.frame else None
        keep = self.filter_predicates[self.filter_var.get()]
        self.visible = [frame for frame in self.frames if keep(frame)]
        self.index = 0
        if current is not None:
            for position, frame in enumerate(self.visible):
                if frame.name == current:
                    self.index = position
                    break
        self.load_frame()

    # -- navigation -----------------------------------------------------

    def previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.load_frame()

    def go_to(self, position: int) -> None:
        """Show the frame at a position, clamped to the list."""
        if not self.visible:
            return
        self.index = min(max(position, 0), len(self.visible) - 1)
        self.load_frame()

    def jump(self, step: int) -> None:
        """Move by more than one frame, without running through them."""
        if self.at_end and step < 0:
            self.go_to(len(self.visible) - 1)
        else:
            self.go_to(self.index + step)

    def next(self) -> None:
        """Move on: the last frame, then the end card, then round again.

        The card is a seam, not a wall. A list that wraps round silently
        is a list you go through twice without noticing; one that
        refuses to wrap is a nuisance when a second pass is what you
        wanted. Costing one deliberate extra press settles both.
        """
        if self.at_end:
            self.index = 0
        elif self.index < len(self.visible):
            self.index += 1
        else:
            return
        self.load_frame()

    @property
    def frame(self) -> Frame | None:
        if 0 <= self.index < len(self.visible):
            return self.visible[self.index]
        return None

    @property
    def at_end(self) -> bool:
        """Past the last frame of a list that does have frames in it."""
        return bool(self.visible) and self.index >= len(self.visible)

    def load_frame(self) -> None:
        """Show a frame, with whatever was recorded about it last time."""
        frame = self.frame
        self._refresh_counts()
        if frame is None:
            self.image = None
            self.frame_label.configure(text="")
            self.readout_label.configure(text="")
            self.detected_label.configure(text="")
            self._show_source()
            # Otherwise the boxes of the frame just left sit there ticked
            # against no frame at all, and Enter would look meaningful.
            self.clear_ticks()
            self._show_end_card()
            self.render_loupe()
            self.render()
            return

        self._clear_end_card()
        self.position_label.configure(text=f"{self.index + 1} / {len(self.visible)}")
        self.frame_label.configure(text=frame.name)
        self.readout_label.configure(text=frame.readout)
        if frame.detected:
            self.detected_label.configure(text=", ".join(frame.detected), fg=FLAGGED)
        else:
            self.detected_label.configure(text="clean", fg=GOOD)

        entry = self.labels.get(frame.name)
        for category in CATEGORIES:
            self.checks[category.key].set(bool(entry)
                                          and category.key in entry.labels)
        self.note_var.set(entry.note if entry else "")
        if entry:
            recorded = ", ".join(entry.labels) or "(note only)"
            self.status_label.configure(
                text=f"✔ you said: {recorded}\n   on {entry.updated}", fg=GOOD)
        else:
            self.status_label.configure(text="— not labelled yet", fg=MUTED)

        self.image = self._load_image(frame)
        self._show_source()
        if self.image is None:
            self.status_label.configure(text="No image at all for this frame.",
                                        fg=FLAGGED)
        self.render_loupe()
        if self.lock_view and self.image is not None:
            self.render()
        else:
            self.fit()

    def _sources(self, frame: Frame) -> list[tuple[str, str]]:
        """Every image this frame has, in the order D walks through them."""
        found = []
        for kind, path in ((SOURCE_PREVIEW, frame.preview),
                           (SOURCE_REVIEW, frame.annotated),
                           (SOURCE_FITS, frame.fits)):
            if not path or not os.path.isfile(path):
                continue
            if kind == SOURCE_FITS and not fits_available():
                continue
            found.append((kind, path))
        return found

    def _chosen_source(self, available) -> tuple[str, str] | None:
        """The preferred image among those a frame has, or its first.

        Takes the list rather than the frame so that a caller which
        already has it -- the source menu does -- does not go back to
        the disk for it. The preference itself is never narrowed to one
        frame: a frame with no REVIEW copy falls back for itself alone,
        and the next frame that has one shows it again.
        """
        for entry in available:
            if entry[0] == self.source:
                return entry
        return available[0] if available else None

    def _image_path(self, frame: Frame) -> str | None:
        chosen = self._chosen_source(self._sources(frame))
        return chosen[1] if chosen else None

    def _load_image(self, frame: Frame):
        """Read a frame's image, keeping the last few for fast paging."""
        path = self._image_path(frame)
        if path is None:
            return None
        if path in self.cache:
            # Move it to the end: what gets dropped below is then the
            # least recently *used*, which is what paging back and forth
            # over a handful of frames needs. Insertion order alone
            # would drop one just looked at.
            self.cache[path] = self.cache.pop(path)
        else:
            try:
                if path.lower().endswith((".fit", ".fits")):
                    # Converted here rather than left greyscale: every
                    # render pastes it into an RGB canvas, and paying
                    # for that once per frame beats paying per zoom.
                    self.cache[path] = read_fits(path).convert("RGB")
                else:
                    image = self._Image.open(path)
                    image.load()
                    self.cache[path] = image.convert("RGB")
            except (OSError, ValueError, IndexError) as error:
                print(f"⚠️ Could not read {path}: {error}")
                return None
            while len(self.cache) > CACHE_SIZE:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[path]

    # -- the end of a list ----------------------------------------------

    def _show_end_card(self) -> None:
        """Say, in the middle of the image area, that the list is done.

        It reports how much of the list has been labelled rather than
        just 'the end', because that is the question actually being
        asked: have I been through this category or not.
        """
        total = len(self.visible)
        labelled = sum(1 for frame in self.visible if frame.name in self.labels)
        chosen = self.filter_var.get()
        if total == 0:
            title = "Nothing here"
            detail = f"No frame matches “{chosen}”."
            hint = "Choose another filter."
            self.position_label.configure(text="0 / 0")
        else:
            title = "End of the list"
            left = total - labelled
            detail = (f"{chosen} — {total} frames, {labelled} labelled"
                      + (f", {left} still not" if left else ", every one of them"))
            hint = ("←  the last frame        ·        "
                    "→  round to the first        ·        or another filter")
            self.position_label.configure(text=f"{total} / {total} · end")
        self.canvas.itemconfigure(self.end_title, text=title)
        self.canvas.itemconfigure(self.end_detail, text=detail)
        self.canvas.itemconfigure(self.end_hint, text=hint)
        self.status_label.configure(text="", fg=MUTED)

    def _clear_end_card(self) -> None:
        for item in (self.end_title, self.end_detail, self.end_hint):
            self.canvas.itemconfigure(item, text="")

    def _place_end_card(self) -> None:
        """Keep the card centred, whatever the window does."""
        x = max(self.canvas.winfo_width(), 1) // 2
        y = max(self.canvas.winfo_height(), 1) // 2
        for item, offset in ((self.end_title, -34), (self.end_detail, 6),
                             (self.end_hint, 40)):
            self.canvas.coords(item, x, y + offset)

    # -- the close-up ---------------------------------------------------

    def _loupe_size(self) -> tuple[int, int]:
        return (max(self.loupe_canvas.winfo_width(), 1),
                max(self.loupe_canvas.winfo_height(), 1))

    def loupe_view(self):
        """The magnified region, as an image.

        Nearest-neighbour on purpose: a smoothed star is a rounder star,
        and rounder is exactly the judgement being made here.
        """
        width, height = self._loupe_size()
        view = self._Image.new("RGB", (width, height), BACKGROUND)
        if self.image is None:
            return view

        left, top, right, bottom = self.loupe_box()
        patch = self.image.resize(
            (max(1, round((right - left) * self.loupe_zoom)),
             max(1, round((bottom - top) * self.loupe_zoom))),
            self._Image.NEAREST, box=(left, top, right, bottom))
        view.paste(patch, (0, 0))
        return view

    def loupe_box(self) -> tuple[float, float, float, float]:
        """The region on show, in image coordinates, kept inside the frame."""
        width, height = self._loupe_size()
        span_x = min(width / self.loupe_zoom, float(self.image.width))
        span_y = min(height / self.loupe_zoom, float(self.image.height))
        x, y = self.loupe_centre or (self.image.width / 2, self.image.height / 2)
        left = min(max(x - span_x / 2, 0.0), self.image.width - span_x)
        top = min(max(y - span_y / 2, 0.0), self.image.height - span_y)
        return (left, top, left + span_x, top + span_y)

    def render_loupe(self) -> None:
        """Put the close-up on screen, and say what it is showing."""
        self.loupe_photo = self._ImageTk.PhotoImage(self.loupe_view())
        self.loupe_canvas.itemconfigure(self.loupe_item, image=self.loupe_photo)
        where = "centre" if self.loupe_centre is None else "%d, %d" % (
            self.loupe_centre[0], self.loupe_centre[1])
        # Keys deliberately not repeated here: spelled out, the line
        # overflowed the column and was clipped at both ends. They live
        # in the help bar along the bottom.
        self.loupe_title.configure(
            text=f"CLOSE-UP — {self.loupe_zoom:.1f}x at {where}")

    def _zoom_loupe(self, factor: float) -> None:
        zoom = min(max(self.loupe_zoom * factor, LOUPE_ZOOM_LIMITS[0]),
                   LOUPE_ZOOM_LIMITS[1])
        if zoom == self.loupe_zoom:
            return
        self.loupe_zoom = zoom
        self.render_loupe()
        self.render()

    def centre_loupe(self) -> None:
        """Send the close-up back to the middle of the frame."""
        self.loupe_centre = None
        self.render_loupe()
        self.render()

    def _place_loupe(self, x: int, y: int) -> None:
        """Point the close-up at what was clicked on the main image."""
        point = self._image_point(x, y)
        if point is None:
            return
        self.loupe_centre = point
        self.render_loupe()
        self.render()

    def _image_point(self, x: int, y: int) -> tuple[float, float] | None:
        """Canvas coordinates to image coordinates, None outside the frame."""
        if self.image is None:
            return None
        width, height = self._view_size()
        left = self.centre[0] - width / (2 * self.zoom)
        top = self.centre[1] - height / (2 * self.zoom)
        point = (left + x / self.zoom, top + y / self.zoom)
        if not (0 <= point[0] < self.image.width
                and 0 <= point[1] < self.image.height):
            return None
        return point

    # -- the image ------------------------------------------------------

    def _view_size(self) -> tuple[int, int]:
        """The canvas less the bar along the bottom.

        The bar is drawn over the canvas but the picture must not sit
        under it: at 12 px it would hide the bottom of the frame, which
        is exactly where the end of a trail tends to be.
        """
        return (max(self.canvas.winfo_width(), 1),
                max(self.canvas.winfo_height() - PROGRESS_HEIGHT, 1))

    def _fit_zoom(self) -> float:
        """The scale at which the whole frame is in view.

        Also the floor for zooming out: there is nothing to see in a
        preview shown smaller than the window it is already inside.
        """
        if self.image is None:
            return 1.0
        width, height = self._view_size()
        return min(width / self.image.width, height / self.image.height)

    def fit(self) -> None:
        """Scale the whole frame into the window."""
        self.fitted = True
        if self.image is None:
            self.render()
            return
        self.zoom = self._fit_zoom()
        self.centre = (self.image.width / 2, self.image.height / 2)
        self.render()

    def _clamp(self) -> None:
        """Keep the view over the image, centred when it is smaller."""
        if self.image is None:
            return
        view_width, view_height = self._view_size()
        width = view_width / self.zoom
        height = view_height / self.zoom
        x, y = self.centre
        if width >= self.image.width:
            x = self.image.width / 2
        else:
            x = min(max(x, width / 2), self.image.width - width / 2)
        if height >= self.image.height:
            y = self.image.height / 2
        else:
            y = min(max(y, height / 2), self.image.height - height / 2)
        self.centre = (x, y)

    def canvas_view(self):
        """The visible part of the image, and only that part.

        Cropping in source coordinates before scaling is what keeps a 10x
        zoom instant: the work is proportional to the window, not to the
        zoom factor.
        """
        width, height = self._view_size()
        canvas_image = self._Image.new(
            "RGB", (width, max(self.canvas.winfo_height(), 1)), BACKGROUND)
        draw = self._ImageDraw.Draw(canvas_image)

        if self.image is not None:
            self.zoom = min(max(self.zoom, self._fit_zoom()), ZOOM_MAX)
            self._clamp()
            left = self.centre[0] - width / (2 * self.zoom)
            top = self.centre[1] - height / (2 * self.zoom)
            x0 = max(0.0, left)
            y0 = max(0.0, top)
            x1 = min(float(self.image.width), left + width / self.zoom)
            y1 = min(float(self.image.height), top + height / self.zoom)
            if x1 > x0 and y1 > y0:
                target = (max(1, round((x1 - x0) * self.zoom)),
                          max(1, round((y1 - y0) * self.zoom)))
                # Nearest above 1:1, so a zoom onto a track shows the
                # pixels the detector measured rather than a blur.
                resample = (self._Image.NEAREST if self.zoom > 1.5
                            else self._Image.BILINEAR)
                patch = self.image.resize(target, resample=resample,
                                          box=(x0, y0, x1, y1))
                canvas_image.paste(patch, (round((x0 - left) * self.zoom),
                                           round((y0 - top) * self.zoom)))
                # Where the close-up is looking. Without it the two views
                # are hard to relate to each other on a field of dots.
                region = self.loupe_box()
                draw.rectangle(
                    [round((region[0] - left) * self.zoom),
                     round((region[1] - top) * self.zoom),
                     round((region[2] - left) * self.zoom),
                     round((region[3] - top) * self.zoom)],
                    outline=ACCENT)

        if self.at_end:
            # A border round the whole area, so the end of a list reads
            # as a state and not as a frame that failed to load.
            draw.rectangle([0, 0, width - 1, height - PROGRESS_HEIGHT - 1],
                           outline=ACCENT)

        if self.visible:
            self._draw_scrubber(draw, width, max(self.canvas.winfo_height(), 1))

        return canvas_image

    def _draw_scrubber(self, draw, width: int, height: int) -> None:
        """The bar along the bottom: where you are, and where you have been.

        Three layers, because it answers three questions at once. The
        track carries a tick per frame already labelled, so the session
        reads as a map of what is done, and the handle sits at the
        current frame and says the whole thing can be dragged -- which
        nothing about a flat line ever did. A fill running to the handle
        was tried and dropped: it repeated the handle's own position and
        muddled the map behind it.
        """
        total = len(self.visible)
        position = self.scrub_to if self.scrub_to is not None else (
            total - 1 if self.at_end else self.index)
        top = height - PROGRESS_HEIGHT
        draw.rectangle([0, top, width, height], fill=PROGRESS_TRACK)

        for number, frame in enumerate(self.visible):
            if frame.name in self.labels:
                left = round(number * width / total)
                draw.rectangle([left, top, max(left + 1,
                                               round((number + 1) * width / total) - 1),
                                height], fill=PROGRESS_LABELLED)

        done = round(width * (position + 1) / total)
        thumb = min(max(done, PROGRESS_THUMB // 2), width - PROGRESS_THUMB // 2)
        draw.rectangle([thumb - PROGRESS_THUMB // 2, top - 2,
                        thumb + PROGRESS_THUMB // 2, height],
                       fill=ACCENT)

    def render(self) -> None:
        """Put the main view on screen."""
        self.photo = self._ImageTk.PhotoImage(self.canvas_view())
        self.canvas.itemconfigure(self.canvas_image, image=self.photo)
        self._place_end_card()

    def _zoom_by(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        """Zoom, keeping the point under the cursor where it is."""
        if self.image is None:
            return
        width, height = self._view_size()
        anchor = anchor or (width // 2, height // 2)
        new_zoom = min(max(self.zoom * factor, self._fit_zoom()), ZOOM_MAX)
        if new_zoom == self.zoom:
            return
        self.fitted = False
        left = self.centre[0] - width / (2 * self.zoom)
        top = self.centre[1] - height / (2 * self.zoom)
        fixed = (left + anchor[0] / self.zoom, top + anchor[1] / self.zoom)
        self.zoom = new_zoom
        self.centre = (fixed[0] - anchor[0] / new_zoom + width / (2 * new_zoom),
                       fixed[1] - anchor[1] / new_zoom + height / (2 * new_zoom))
        self.render()

    def _on_wheel(self, event) -> None:
        self._zoom_by(ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP,
                      (event.x, event.y))

    def _on_press(self, event) -> None:
        self.master.focus_set()
        self._hide_suggestions()
        if self._in_scrub_zone(event.y) and self.visible:
            self.scrub_to = self._scrub_index(event.x)
            self.go_to(self.scrub_to)
            return
        self.drag_from = (event.x, event.y)
        self.press_at = (event.x, event.y)

    def _on_release(self, event) -> None:
        """A click that did not drag means 'look here', not 'pan'."""
        if self.scrub_to is not None:
            target, self.scrub_to = self.scrub_to, None
            self.go_to(target)
            return
        self.drag_from = None
        if self.press_at is None:
            return
        moved = (abs(event.x - self.press_at[0])
                 + abs(event.y - self.press_at[1]))
        self.press_at = None
        if moved <= CLICK_SLOP:
            self._place_loupe(event.x, event.y)

    def _on_hover(self, event) -> None:
        """Change the pointer over the bar. It is the only affordance a
        drawn rectangle has, and without it nobody tries to drag it."""
        wanted = "sb_h_double_arrow" if self._in_scrub_zone(event.y) else ""
        if wanted != self.cursor:
            self.cursor = wanted
            self.canvas.configure(cursor=wanted)

    def _in_scrub_zone(self, y: int) -> bool:
        return y >= max(self.canvas.winfo_height(), 1) - SCRUB_GRAB

    def _scrub_index(self, x: int) -> int:
        """Which frame a point along the bar stands for."""
        width = max(self.canvas.winfo_width(), 1)
        share = min(max(x / width, 0.0), 0.9999)
        return int(share * len(self.visible))

    def _on_right_click(self, event) -> None:
        self.master.focus_set()
        self._place_loupe(event.x, event.y)

    def _on_loupe_wheel(self, event) -> None:
        self._zoom_loupe(LOUPE_STEP if event.delta > 0 else 1 / LOUPE_STEP)

    def _on_drag(self, event) -> None:
        if self.scrub_to is not None:
            self.scrub_to = self._scrub_index(event.x)
            # The name, not the picture: decoding one image per step
            # would make the drag crawl, and the timestamp is what tells
            # you where you have got to anyway.
            target = self.visible[self.scrub_to]
            self.position_label.configure(
                text=f"→ {self.scrub_to + 1} / {len(self.visible)}"
                     f"  {short_label(target.name)}")
            self.render()
            return
        if self.drag_from is None or self.image is None:
            return
        dx = event.x - self.drag_from[0]
        dy = event.y - self.drag_from[1]
        self.drag_from = (event.x, event.y)
        self.fitted = False
        self.centre = (self.centre[0] - dx / self.zoom,
                       self.centre[1] - dy / self.zoom)
        self.render()

    def cycle_source(self) -> None:
        """Show the next image this frame has.

        The JPG preview, the copy REVIEW filed -- which carries the
        --debug drawings -- and the FITS itself, stretched for the
        screen. The FITS is a Bayer mosaic, so its sky carries a fine
        checkerboard; star shapes and trails read the same on both.
        """
        frame = self.frame
        if frame is None:
            return
        available = self._sources(frame)
        kinds = [kind for kind, _ in available]
        if len(kinds) < 2:
            self.status_label.configure(
                text=f"Only one image for this frame ({kinds[0] if kinds else 'none'}).",
                fg=MUTED)
            return
        chosen = self._chosen_source(available)
        self.source = kinds[(kinds.index(chosen[0]) + 1) % len(kinds)]
        self.image = self._load_image(frame)
        self._show_source()
        self.render_loupe()
        self.render()

    def _show_source(self) -> None:
        """Offer the images this frame has, and show the one on screen."""
        frame = self.frame
        available = self._sources(frame) if frame else []
        menu = self.source_menu["menu"]
        menu.delete(0, "end")
        for kind, _ in available:
            # Rebuilding the entries by hand means setting the variable
            # by hand too: that is what OptionMenu's own command does.
            menu.add_command(label=kind,
                             command=lambda kind=kind: self._pick_source(kind))
        self.source_menu.configure(state="normal" if available else "disabled")
        chosen = self._chosen_source(available)
        self.source_var.set(chosen[0] if chosen else "")

    def _pick_source(self, kind: str) -> None:
        """Choose an image from the menu, as D does from the keyboard."""
        frame = self.frame
        if frame is None:
            return
        self.source = kind
        self.image = self._load_image(frame)
        self._show_source()
        self.render_loupe()
        self.render()

    def toggle_lock(self) -> None:
        """Carry the zoom and pan across frames, to compare the same corner."""
        self.lock_view = not self.lock_view
        state = "locked" if self.lock_view else "refit on each frame"
        self.status_label.configure(text=f"View {state}.", fg=MUTED)

    # -- labelling ------------------------------------------------------

    def _check(self, key: str) -> None:
        """Tick a box. Pressing the key again does nothing.

        Not a toggle, deliberately. Going through a category means
        pressing the same key on frame after frame, and a toggle turns
        the second press on a frame already ticked into a silent
        unticking. Use the mouse, or 0, to undo one.
        """
        self.checks[key].set(True)
        self._exclusive(key)

    def clear_ticks(self) -> None:
        """Untick everything on this frame, note included."""
        for variable in self.checks.values():
            variable.set(False)
        self.note_var.set("")

    def _exclusive(self, key: str) -> None:
        """'Clean' and a fault cannot both hold, so one clears the other."""
        self.master.focus_set()
        if not self.checks[key].get():
            return
        if key == CLEAN:
            for other in self.checks:
                if other != CLEAN:
                    self.checks[other].set(False)
        else:
            self.checks[CLEAN].set(False)

    def accept_detector(self) -> None:
        """Tick exactly what the detector said, ready to be validated.

        Deliberately not the default state of a frame: a box pre-ticked
        is a box agreed with, and ground truth that merely echoes the
        detector would tell the next tuning round nothing.
        """
        frame = self.frame
        if frame is None:
            return
        for category in CATEGORIES:
            self.checks[category.key].set(category.key in frame.detected)
        if not frame.detected:
            self.checks[CLEAN].set(True)

    def validate(self) -> str:
        """Record the frame and move on."""
        frame = self.frame
        if frame is None:
            return "break"
        chosen = tuple(category.key for category in CATEGORIES
                       if self.checks[category.key].get())
        note = self.note_var.get().strip()
        if not chosen and not note:
            self.status_label.configure(
                text="Nothing ticked — press 1 if the frame is clean.",
                fg=FLAGGED)
            return "break"

        self.labels[frame.name] = Label(
            frame=frame.name,
            labels=chosen,
            note=note,
            detected=" ".join(frame.detected),
            updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        save_labels(self.labels_path, self.labels)
        self.master.focus_set()
        self.next()
        return "break"

    def clear_label(self) -> None:
        """Take a frame back out of the labels file."""
        self.master.focus_set()
        frame = self.frame
        if frame is None or frame.name not in self.labels:
            return
        del self.labels[frame.name]
        save_labels(self.labels_path, self.labels)
        self.load_frame()

    def _refresh_counts(self) -> None:
        done = sum(1 for frame in self.frames if frame.name in self.labels)
        self.count_label.configure(
            text=f"{done} / {len(self.frames)} frames labelled\n"
                 f"{os.path.basename(self.session)}/{LABELS_FILENAME}")

    def quit(self) -> None:
        self.master.destroy()


# ======================================================================
#  Entry point
# ======================================================================

def choose_session() -> str | None:
    """Show the session list and return the one picked, or None.

    The window that replaced the numbered console prompt: pick a folder,
    see every session under it with what has been done to it so far,
    double-click one.
    """
    import tkinter as tk
    from tkinter import ttk

    from session_browser import SessionBrowser, install_error_hooks

    chosen: list[str] = []
    root = tk.Tk()
    root.title(f"{LABELLER_NAME} — choose a session")
    root.geometry("900x560")
    install_error_hooks(root, LABELLER_NAME)
    config = load_config()

    def pick(session) -> None:
        chosen.append(session.root)
        root.destroy()

    body = ttk.Frame(root, padding=10)
    body.pack(fill="both", expand=True)
    browser = SessionBrowser(body, config, on_open=pick)
    browser.pack(fill="both", expand=True)
    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(8, 0))
    open_button = ttk.Button(buttons, text="Label this session",
                             command=lambda: browser._opened())
    open_button.pack(side="right")
    ttk.Label(buttons, text="Double-click a session to open it.").pack(side="left")
    root.mainloop()
    return chosen[0] if chosen else None


def show_problem(message: str) -> None:
    """Say what is wrong in a dialog: there may be no console to read."""
    print(f"⚠️ {message}")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(LABELLER_NAME, message, parent=root)
        root.destroy()
    except Exception:  # no display at all: the print above is all there is
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label the previews of a session by eye, to build ground truth.",
        epilog="With no folder, a window lists the sessions to choose from. "
               "Labels go to <session>/labels.csv.")
    parser.add_argument("folder", nargs="?", help="the session folder")
    arguments = parser.parse_args(argv)

    try:
        import tkinter as tk
        from PIL import Image  # noqa: F401  -- checked here, used in Labeller
    except ImportError as error:
        show_problem(f"This tool needs tkinter and Pillow: {error}\n"
                     f"pip install pillow")
        return 1

    folder = arguments.folder
    if folder is None:
        folder = choose_session()
        if folder is None:
            return 0
    if not os.path.isdir(folder):
        show_problem(f"'{folder}' is not a folder.")
        return 1
    folder = session_root(folder, configured_layout())

    frames = build_frames(folder)
    if not frames:
        show_problem(f"No light frames, previews or report in '{folder}'.\n"
                     f"Check the folder layout in the session list.")
        return 1

    labels_path = os.path.join(folder, LABELS_FILENAME)
    labels = load_labels(labels_path)
    flagged = sum(1 for frame in frames if frame.detected)
    previews = sum(1 for frame in frames if frame.preview)
    raw = sum(1 for frame in frames if frame.fits)
    missing = sum(1 for frame in frames
                  if not frame.preview and not frame.annotated
                  and not (frame.fits and fits_available()))

    print(f"\U0001f4c2 {folder}")
    print(f"\U0001f50e {len(frames)} frames · {flagged} flagged by the detector")
    print(f"🖼️ {previews} JPG previews · {raw} FITS"
          + ("" if fits_available() else " (unreadable: no numpy/astropy)"))
    if missing:
        print(f"   ⚠️ {missing} of them have no image to show at all.")
    print(f"\U0001f4dd {labels_path} · {len(labels)} already labelled\n")

    from session_browser import install_error_hooks
    root = tk.Tk()
    install_error_hooks(root, LABELLER_NAME)
    Labeller(root, folder, frames, labels_path, labels)
    root.mainloop()

    print(f"\U0001f4dd {len(load_labels(labels_path))} frames labelled "
          f"in {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
