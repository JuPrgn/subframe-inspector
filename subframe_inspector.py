# -*- coding: utf-8 -*-
"""Fast triage of Seestar light frames, to be run before stacking.

This is a sorting tool, not an astrometry package. It never registers,
transforms or resamples an image, and it pulls in nothing heavier than
numpy, scipy and astropy. Every frame is read once, downsampled, and
described by a handful of cheap measurements; the whole session is then
classified from those numbers.

What it flags
-------------
Cloud or haze
    The sky background rises well above the session median.
Trailing
    The stars are drawn out into ovals, or a jolt during the exposure
    printed every one of them twice. This is about the *shape of the
    stars*, nothing else: a field that merely moved between exposures
    keeps round stars, stacking registers it away, and it is not
    reported.
Satellite, meteor or aircraft
    A straight track crossing the field. The Seestar's field is small
    enough that anything passing through it flies dead straight, so the
    detector looks for one line, not for scattered elongated blobs --
    and reassembles it when noise breaks it into fragments. Aircraft
    need a track that is both wide *and* unusually bright, which keeps
    ordinary satellites out of that category.

A frame can raise several alerts at once; each one is decided
independently.

Every threshold that describes the sky is written in arcseconds and
converted to pixels once, from the FITS header, so the same settings
hold for an S30, an S50 and an S50 Pro despite their different focal
lengths and sensors.

A session is any folder holding light frames, laid out the way the
Seestar writes them or the way the user configured -- see
``session_layout``. Typically::

    <Target>_sub/Lights/    the light frames to inspect (or at the root)
    <Target>_sub/JPG/       previews, one .jpg per .fit (optional)
    <Target>_sub/REVIEW/    created here, one subfolder per alert

Only the JPG previews are ever copied or moved. The FITS files are left
untouched, so a mistaken threshold costs you nothing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Callable, Sequence

import numpy as np
from astropy.io import fits
from scipy.ndimage import grey_opening, label
from scipy.spatial import cKDTree

from session_layout import (Session, configured_layout, find_sessions,
                            frame_stem, resolve, session_root)

# The report uses emoji. On a Windows console still running cp1252 --
# cmd.exe out of the box, or any launch outside an IDE -- the very first
# print would abort the run with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ======================================================================
#  Settings
# ======================================================================

# Eight-connectivity, diagonals included. scipy's default is the
# four-neighbour cross, which shatters a diagonal satellite trail into a
# line of isolated pixels instead of recognising one component.
EIGHT_CONNECTED = np.ones((3, 3), dtype=int)

CATEGORY_CLOUDS = "Clouds_Haze"
CATEGORY_TRAILING = "Trailing"
CATEGORY_AIRCRAFT = "Aircraft"
CATEGORY_METEORS = "Meteors"
CATEGORY_SATELLITES = "Satellites"

# Every category a frame can be filed under, in the order they are
# reported. Also the vocabulary --categories accepts.
ALL_CATEGORIES = (CATEGORY_CLOUDS, CATEGORY_TRAILING, CATEGORY_SATELLITES,
                  CATEGORY_METEORS, CATEGORY_AIRCRAFT)

# Every folder a run of this script may have created under REVIEW/. The
# retired names are kept here so that a folder left behind by an older
# version is tidied away rather than lingering beside the current ones.
GENERATED_FOLDERS = (CATEGORY_CLOUDS, CATEGORY_TRAILING, CATEGORY_AIRCRAFT,
                     CATEGORY_METEORS, CATEGORY_SATELLITES,
                     "Field_Shift", "Satellites_Meteors", "_debug")

# 206265 arcsec in a radian: the small-angle conversion from a pixel
# pitch in microns and a focal length in millimetres to a plate scale.
ARCSEC_PER_RADIAN = 206264.806


@dataclass(frozen=True)
class Settings:
    """Every tunable threshold, in one place.

    Anything that measures the sky is quoted in **arcseconds**, never in
    pixels. :class:`Geometry` reads the focal length and pixel pitch out
    of the FITS header and :class:`Thresholds` turns those arcseconds
    into pixels once per session, which is what lets one set of numbers
    cover the whole Seestar range. The handful of values still given in
    pixels are marked as such and apply to the working image, whose
    scale ``working_scale`` holds roughly constant across models.

    The defaults were calibrated on real S30 Pro sessions (LBN 550 and
    NGC 6910, 60 s subs, 3.74 arcsec/px). They are a starting point, not
    physical constants.
    """

    # -- Working image --------------------------------------------------
    # The frame is decimated to about this scale before anything else
    # happens. Keeping the working scale fixed rather than the decimation
    # factor is what makes the pixel-space thresholds below behave the
    # same on a 1.6 arcsec/px S50 Pro as on a 3.7 arcsec/px S30.
    working_scale: float = 7.5          # arcsec per working pixel
    # How far the working image may land from that before the run is
    # declined. Decimation can only ever make a scale coarser, so a lens
    # already coarser than working_scale cannot be brought back and every
    # pixel-valued setting below silently means something else.
    #
    # Measured: the Seestar main lens gives 3.74 to 3.99 arcsec/px, so
    # 7.48 to 7.98 working; a finer scope hits the step cap of 4 and lands
    # near 6.4. The wide-angle lens -- 6 mm focal, 1.6 micron pixels --
    # gives 55 arcsec/px and cannot decimate at all, which put
    # segment_min_length at 0.55 working px and flagged 59 frames of 59 as
    # satellites. 1.5 clears every main lens and catches that.
    working_scale_tolerance: float = 1.5
    opening_size: int = 5               # working px
    bright_percentile: float = 98.5
    min_component_pixels: int = 3

    # -- Straight tracks: satellites, meteors, aircraft -----------------
    # Two bars, and the gap between them is the point. A track can only
    # be *started* by a fragment thin enough to be nothing but a trail:
    # checked one by one against their previews, the longest fragment of
    # a real trail measured between 10 and 184 times as long as it is
    # wide, a smeared star peaked at 6, and the blobs that used to be
    # reported on dense fields -- merged star pairs, mostly -- sat
    # between 2.7 and 4.3. Once the line exists, weaker fragments are
    # allowed to *join* it, which is what recovers a faint trail: those
    # come through as a dotted line of specks barely longer than wide,
    # and at the strict bar only two or three of them survive.
    segment_min_length: float = 30.0    # arcsec
    segment_min_ratio: float = 4.0      # to join a track
    # The seed bar exists to stop a merged pair of stars from starting a
    # line; measured, those reach 4.3 at the very most, so this sits ten
    # percent clear of them. It used to be 6, which was guesswork from
    # before the residual test below existed -- that test is now what
    # really guards against a line built out of nothing, and dropping the
    # bar to here recovered a real trail whose thinnest piece was 4.97.
    seed_min_ratio: float = 4.75        # to start one
    # Fragments of one trail stay on its line; the tolerance covers the
    # error on the line's direction, which is refit from the fragments.
    track_tolerance: float = 75.0       # arcsec, perpendicular
    track_max_tilt: float = 25.0        # degrees between fragment and line
    track_min_span: float = 150.0       # arcsec, end to end
    # Trailed stars are all smeared the same way, so a line drawn along
    # their common direction does pick up a few of them by chance. What
    # separates those from a genuine dotted trail is not how many there
    # are -- a faint trail can leave as few as three -- but how well they
    # sit on the line. A real trail *is* a straight line, so its pieces
    # land on it to a fraction of a pixel; dashes gathered by luck are
    # spread through the whole tolerance band.
    #
    # Measured over nine confirmed trails and the chance lines found on
    # trailed frames: real 0.07 to 0.30 working px of RMS residual,
    # chance 1.02 and 7.12. An earlier "at least six fragments" rule sat
    # in a gap it had no data for and missed a four-fragment satellite.
    #
    # So: three pieces or more are judged on how straight they are, and
    # nothing else -- one or two, which fit any line exactly, have to be
    # continuous instead.
    track_min_coverage: float = 0.5
    track_max_residual: float = 4.5     # arcsec, RMS across the line
    track_min_fragments: int = 3        # below this a residual means nothing
    # On a frame whose own stars are trailed, a sparse line is suspect
    # however straight it is: three dashes fall into line often enough to
    # matter, and the residual cannot tell those from a faint trail (the
    # chance ones measured 0.03 to 0.47, a real one 0.15). What does tell
    # them apart is how many pieces there are -- the chance lines had
    # three, the real trail twenty-two. So on those frames a track has to
    # be either continuous or made of this many pieces.
    track_fragments_if_soft: int = 6
    # -- Wide tracks: aircraft and meteors ------------------------------
    # A satellite trail is a straight line convolved with the seeing
    # disc and nothing else, so its width is set by the optics, not by
    # how bright it is. Measured over 26 tracks confirmed as satellites
    # by eye, spanning 69x in peak brightness, the flux-weighted width
    # never passed 11 arcsec. Anything materially wider is not an
    # ordinary satellite: a meteor burns a plasma trail with a glow
    # around it, an aircraft carries several lights a wingspan apart.
    #
    # Use the flux-weighted width for this, never the geometric one:
    # a handful of outlying pixels pushed the widest satellites to 30
    # arcsec on that measure, right through the old threshold.
    wide_track_width: float = 18.0      # arcsec, flux-weighted, 60% of margin
    # Told apart by how the light sits along the trail: an aircraft is
    # lit evenly for the whole exposure, a meteor has a bright head and
    # a fading tail. UNVALIDATED -- nobody has confirmed either kind on
    # this data. The cost of getting it wrong is one preview filed in
    # the wrong folder, which is why it ships anyway.
    meteor_taper: float = 2.0           # brighter half over dimmer half
    aircraft_brightness: float = 1.5    # x the session noise
    # A meteor lasts a fraction of a second and an aircraft crosses the
    # field in a few, so neither can leave the same line on two frames in
    # a row. A satellite in a high orbit does: on M 45 one sat in eight
    # consecutive frames, measured 3.16 wide on one of them and was filed
    # as a meteor. A track found again on the previous or the next frame,
    # at the same angle and on the same line, is therefore kept as a
    # satellite whatever its width. Measured on that session: the same
    # object on consecutive frames lies within 2.2 working px (16 arcsec)
    # and 1.1 degrees of its own line; the nearest unrelated pair was
    # 20 px off. The point is only to clear a false meteor, so it never
    # removes an alert -- it decides which folder a track goes to.
    track_recur_offset: float = 40.0    # arcsec, across the line
    track_recur_tilt: float = 3.0       # degrees
    # The second way into Meteors, and the one that caught both confirmed
    # meteors, which were too thin for the width gate: a diffuse green
    # glow beside the trail along its whole length (measure_track_green).
    # Measured: 9.4 and 18.8 sigma on the two meteors of M 45, at most 1.3
    # over 77 satellites from eleven sessions, dense Milky Way fields and
    # the LP filter included. 4 sits three times above the satellites and
    # twice below the fainter meteor. A track found again on the next
    # frame is a satellite whatever its colour.
    meteor_green_sigma: float = 4.0

    # -- Cloud and haze -------------------------------------------------
    # Three signatures, and the third one carries most of the recall.
    #
    # The sigma test only works while haze is the exception on a night:
    # measured over 222 ticked frames across nine sessions, the frames it
    # misses have a background sitting *at* the night's own median,
    # because a third of some of those nights is veiled. Lowering the bar
    # does not help -- 2.0 buys seven frames and costs fourteen -- and
    # neither does referencing a low percentile of the night instead of
    # its median (+10 frames, 15 wrong). Both were swept; do not re-run.
    #
    # What does work is that a veil *dims the stars* rather than
    # brightening the sky, and it shows up twice over. The count of stars
    # collapses -- 0.44 of the session median on the ticked frames
    # against 0.96 on the ones judged and passed, the widest separation
    # of any signal measured here -- and so does the frame's standard
    # deviation, which over the working image is carried by stars and
    # structure rather than by read noise.
    #
    # Both are needed, as an AND. Either alone reaches the same recall
    # for two to twelve times the wrong flags: a sparse field or a focus
    # change moves one of them, a veil moves both. Together they took
    # cloud recall from 42% to 73% for eleven flags on frames the owner
    # had ticked for something else.
    #
    # Two things measured and rejected, so nobody re-runs them: the
    # *brightness* of the surviving stars barely separates at all (0.92
    # against 1.06), because the detection floor is a percentile and what
    # survives a veil is always the bright end -- it is their number that
    # goes, not their light. And referencing the count to the night's p75
    # or p90 rather than its median gains nothing (0.41 against 0.44),
    # unlike the sky background, whose median a veiled night really does
    # drag with it.
    haze_sigma: float = 2.5
    haze_background_rise: float = 1.10   # 1.15 missed ten frames for nothing
    haze_contrast_drop: float = 0.90     # x the session's own noise
    haze_star_drop: float = 0.85         # x the stars a good frame shows

    # -- Trailing -------------------------------------------------------
    # Measured on the flux-weighted second moments of a few hundred stars
    # at full resolution, compared with the sharpest quarter of the
    # session. Two populations, no gap between them -- wind smears
    # progressively -- so this is a severity dial, not a constant.
    #
    # On the 230-frame LBN 550 session all fourteen frames the owner
    # labelled by eye land at 1.38x or above, the sharpest quarter sits
    # at 1.00x by construction and the median frame at 1.04x.
    # Measured against 131 frames judged by eye, counting an unlabelled
    # frame as clean: at 1.30 the oval test agrees with the eye 97% of
    # the time, and dropping it to 1.25 buys one more real frame at the
    # price of thirteen the owner had looked at and passed over. The
    # frames in between are the ones they call borderline anyway.
    #   1.20 -> strict; two of every three flags will be marginal
    #   1.30 -> agrees with the eye, and misses only what the eye doubts
    #   1.45 -> only obvious ovals
    # The owner's own two rulings bracket this: 023406 at 1.26x is real
    # trailing they can see, 034739 at 1.22x has sound stars. There is no
    # gap to sit in -- the 139 missed ticks are spread flat from 1.00 to
    # 1.30, 20 to 25 per 0.05 band -- so this is a severity dial, not a
    # constant, and every step down costs more than the last.
    trailing_elongation_ratio: float = 1.25
    # Stars can also lose their shape without leaning anywhere -- focus
    # drift, or a spell of bad seeing -- and the owner counts that as the
    # same fault. It needs its own test: on the frames labelled that way,
    # the stars ran 1.36 to 1.54 times the session's sharpest in radius
    # while their elongation stayed near the session median, so the oval
    # test cannot see them. This is the half of the trailing verdict that
    # the labels actually bought: seven more frames caught, and not one
    # extra flag on a frame the owner had passed over.
    trailing_radius_ratio: float = 1.35
    star_max_length: float = 90.0       # arcsec; longer is a trail, not a star
    sharpness_centre_fraction: float = 0.5
    sharpness_opening_size: float = 34.0  # arcsec
    sharpness_percentile: float = 99.5
    sharpness_min_stars: int = 30

    # -- Star doubling --------------------------------------------------
    # A brief jolt during the exposure prints every star twice, offset by
    # one shared vector. The stars stay round, so no shape measurement
    # sees it. Below the lower bound we would only be pairing the two
    # halves of a slightly smeared star; above the upper one it is a
    # proper trail, already covered by the elongation test.
    doubling_min_distance: float = 22.0   # arcsec
    # How far apart the two images may be, as a share of the shorter side
    # of the region the stars are measured in. Geometric on purpose: past
    # this, so few stars keep their partner inside the crop that the
    # fraction test stops meaning anything, and that limit belongs to the
    # measurement rather than to any particular mount or night.
    #
    # It was an absolute 300 arcsec, on the reasoning that a wider offset
    # "must be a trail, covered elsewhere". That reasoning is wrong: a
    # trail needs continuous motion, while a knock prints two sharp
    # images as far apart as the knock was large. A real one measured
    # 192 px -- 719 arcsec, 0.2 degrees -- and went unreported because of
    # it, on a frame whose stars are neither oval nor swollen nor leaning
    # anywhere. They are simply twice as many.
    doubling_max_fraction: float = 0.33
    doubling_tolerance: float = 19.0      # arcsec
    # Doubling is systematic -- every star moves by the same vector --
    # and that is what separates it from a handful of genuine doubles or
    # from chance alignments in a crowded field. Measured on a real dense
    # field: the best vector of a clean frame never gathers more than 8%
    # of the stars, while a doubling faint enough to be half the primary
    # brightness still gathers 27%.
    doubling_min_fraction: float = 0.20
    doubling_min_pairs: int = 3
    doubling_max_stars: int = 150
    doubling_max_scatter: float = 75.0    # arcsec, see find_star_doubling

    # -- Field shift between consecutive frames -------------------------
    # Reported in the CSV and nowhere else. A field that moved but kept
    # its stars round is not a fault: stacking registers it away and the
    # owner crops the edges regardless. It stays measured because it is
    # the context that explains a trailed frame -- and because it costs
    # almost nothing.
    shift_star_count: int = 120
    shift_search_radius: float = 1120.0   # arcsec
    shift_vote_bin: float = 15.0          # arcsec

    # Left at None, which lets the pool pick. Capping it is for a machine
    # that runs out of memory rather than one that runs out of cores:
    # every worker holds a full-resolution frame and several working
    # copies, so a dozen of them at once can ask for two gigabytes. The
    # measurements do not change with the worker count -- only whether
    # they all complete. See the note on parallelism in .claude/CLAUDE.md before
    # capping it for speed, which is a different and mostly losing game.
    max_workers: int | None = None

    # -- Output -----------------------------------------------------------
    review_folder: str = "REVIEW"
    preview_folder: str = "JPG"
    preview_extensions: tuple[str, ...] = (".jpg", ".JPG", ".jpeg", ".JPEG")
    # One row per frame, every measurement, flagged or not -- so a
    # threshold can be re-tried in a spreadsheet without re-reading 4 GB
    # of FITS. This is the useful artefact; the Markdown one is a summary
    # for reading.
    report_csv: str = "report.csv"
    report_markdown: str = "report.md"
    write_reports: bool = True
    # -- Case export ----------------------------------------------------
    # A run can write out everything it measured, minus the pixels: a
    # couple of megabytes against the four gigabytes of FITS behind them.
    # That is what makes a night's hard cases keepable and shareable, and
    # what lets a threshold be re-scored months later, on every session
    # ever exported, without the frames. See the Case export section.
    #
    # Segments are exported below the bars the run itself uses, so a
    # replay can lower the join bar as well as raise it. Nothing can be
    # replayed below this floor, so it is set with room to spare rather
    # than at the bar of the day.
    export_cases: bool = False
    # The bundle is named after the night it holds, not after this
    # setting alone: every session folder would otherwise produce a file
    # called the same thing, and they are meant to be collected together.
    cases_prefix: str = "cases"
    export_segment_min_ratio: float = 3.0
    export_segment_length_factor: float = 0.7
    # The resolved primitives, for hypotheses nobody has tested yet: a
    # local cloud, trailing in the corners. See spatial_summary. A 4x4
    # grid puts a few hundred components in each tile, where the 120
    # landmarks the shift uses give 13 -- and at 13 the counting noise
    # alone swamps anything a cloud would do.
    export_grid: int = 4
    export_star_map: int = 400

    # Debug mode writes an annotated copy of every flagged preview into
    # REVIEW/_debug/, showing what the detector actually saw. It needs
    # Pillow, keeps far more per-frame data alive, and roughly doubles
    # the run time -- for working out why a frame was flagged, not for
    # everyday use.
    debug: bool = False


DEFAULT_SETTINGS = Settings()

# Used only when the header carries no focal length or pixel pitch, which
# no Seestar firmware seen so far omits. Roughly an S30 Pro.
FALLBACK_PIXEL_SCALE = 3.74  # arcsec per full-resolution pixel


# ======================================================================
#  Camera geometry
# ======================================================================

@dataclass(frozen=True)
class Geometry:
    """What the camera makes of an arcsecond, worked out from the header.

    An S30 spreads the sky over 3.7 arcsec per pixel, an S50 Pro over
    less than half that. Nothing else in this file needs to know which
    model took the frames: it asks this object to convert.
    """

    width: int          # full-resolution columns
    height: int         # full-resolution rows
    pixel_scale: float  # arcsec per full-resolution pixel
    step: int           # decimation factor of the working image
    model: str          # for the console banner
    # True when the header gave no focal length or pixel pitch and the
    # scale below is a guess. Every angular threshold then rests on that
    # guess, so the run says so out loud rather than quietly reporting
    # nonsense on somebody else's camera.
    scale_is_assumed: bool = False
    # The colour filter layout of the raw mosaic, e.g. 'GRBG', or '' when
    # the header gives none -- then nothing colour-based is measured.
    bayer: str = ""

    @classmethod
    def of(cls, path: str, settings: Settings) -> "Geometry":
        """Read one frame's header and derive the session's geometry."""
        with fits.open(path, memmap=False) as hdu_list:
            header = hdu_list[0].header
            shape = next((h.data.shape for h in hdu_list if h.data is not None),
                         (header.get("NAXIS2", 0), header.get("NAXIS1", 0)))
        return cls.from_header(header, shape, settings)

    @classmethod
    def from_header(cls, header, shape: Sequence[int],
                    settings: Settings) -> "Geometry":
        """Work out the geometry from an already-read header.

        Split from :meth:`of` so the arithmetic can be exercised without a
        file on disk, and so a caller with the header in hand does not
        have to re-open the frame.
        """
        height, width = int(shape[-2]), int(shape[-1])
        focal_length = float(header.get("FOCALLEN", 0.0) or 0.0)   # mm
        pixel_pitch = float(header.get("XPIXSZ", 0.0) or 0.0)      # microns
        binning = max(int(header.get("XBINNING", 1) or 1), 1)

        assumed = not (focal_length > 0 and pixel_pitch > 0)
        if assumed:
            scale = FALLBACK_PIXEL_SCALE
        else:
            scale = (ARCSEC_PER_RADIAN * pixel_pitch * binning
                     / (focal_length * 1000.0))

        # Decimate to roughly the target working scale. Clamped: a factor
        # of 1 would cost the speed the whole design is built around, and
        # past 4 the stars stop being resolved at all.
        step = min(max(int(round(settings.working_scale / scale)), 1), 4)

        model = str(header.get("CREATOR") or header.get("TELESCOP") or "unknown")
        bayer = str(header.get("BAYERPAT") or "").strip().upper()
        if len(bayer) != 4 or set(bayer) - set("RGB") or len(shape) != 2:
            bayer = ""
        return cls(width=width, height=height, pixel_scale=scale, step=step,
                   model=model, scale_is_assumed=assumed, bayer=bayer)

    @property
    def working_scale(self) -> float:
        """Arcseconds per pixel of the decimated image the pipeline works on."""
        return self.pixel_scale * self.step

    @property
    def field_of_view(self) -> tuple[float, float]:
        """Field covered, in degrees."""
        return (self.width * self.pixel_scale / 3600.0,
                self.height * self.pixel_scale / 3600.0)

    def px(self, arcsec: float) -> float:
        """Full-resolution pixels covering that many arcseconds."""
        return arcsec / self.pixel_scale

    def working_px(self, arcsec: float) -> float:
        """Working-image pixels covering that many arcseconds."""
        return arcsec / (self.pixel_scale * self.step)

    def odd_px(self, arcsec: float) -> int:
        """Full-resolution pixels, rounded to an odd size for a kernel."""
        size = max(int(round(self.px(arcsec))), 3)
        return size if size % 2 else size + 1


@dataclass(frozen=True)
class Thresholds:
    """The settings, converted to pixels once, in one place.

    Two distances live in this file and mixing them up used to be the
    easiest way to get nonsense out of it: the working image covers
    ``step`` times as much sky per pixel as the full-resolution one. Every
    field below states which space it belongs to, and nothing outside this
    class converts anything.
    """

    # -- working image --------------------------------------------------
    segment_min_length: float
    track_tolerance: float
    track_max_residual: float
    track_min_span: float
    track_cos_tilt: float          # cosine of the largest allowed tilt
    wide_track_width: float
    track_recur_offset: float
    star_max_length: float
    shift_search_radius: float
    shift_vote_bin: float

    # -- full resolution ------------------------------------------------
    sharpness_opening_size: int
    star_max_length_full: float
    doubling_min_distance: float
    doubling_max_distance: float
    doubling_tolerance: float
    doubling_max_scatter: float

    @classmethod
    def derive(cls, settings: Settings, geometry: Geometry) -> "Thresholds":
        working = geometry.working_px
        full = geometry.px
        return cls(
            segment_min_length=working(settings.segment_min_length),
            track_tolerance=working(settings.track_tolerance),
            track_max_residual=working(settings.track_max_residual),
            track_min_span=working(settings.track_min_span),
            track_cos_tilt=math.cos(math.radians(settings.track_max_tilt)),
            wide_track_width=working(settings.wide_track_width),
            track_recur_offset=working(settings.track_recur_offset),
            star_max_length=working(settings.star_max_length),
            shift_search_radius=working(settings.shift_search_radius),
            shift_vote_bin=working(settings.shift_vote_bin),
            sharpness_opening_size=geometry.odd_px(settings.sharpness_opening_size),
            star_max_length_full=full(settings.star_max_length),
            doubling_min_distance=full(settings.doubling_min_distance),
            doubling_max_distance=(settings.doubling_max_fraction
                                   * settings.sharpness_centre_fraction
                                   * min(geometry.width, geometry.height)),
            doubling_tolerance=full(settings.doubling_tolerance),
            doubling_max_scatter=full(settings.doubling_max_scatter),
        )


# ======================================================================
#  Image loading
# ======================================================================

def unsupported_scale(geometry: Geometry, settings: Settings) -> str | None:
    """Why this session cannot be judged, or None when it can.

    The whole design rests on the working image landing near
    ``working_scale``, which is what lets the few genuinely pixel-valued
    settings hold across the Seestar range. Decimation only coarsens, so
    a lens whose plate scale is already coarser than the target cannot be
    brought back to it: ``step`` clamps at 1 and every pixel-valued
    threshold quietly comes to mean something else.

    That is not a threshold to retune but a limit of the tool, and saying
    so is worth far more than the verdicts it would otherwise emit -- on
    the wide-angle lens it reported all 59 frames of a session as
    satellites, which is the sort of output that costs an evening to
    disbelieve.
    """
    limit = settings.working_scale * settings.working_scale_tolerance
    if geometry.working_scale <= limit:
        return None
    return (f"{geometry.pixel_scale:.1f}\" per pixel decimates to "
            f"{geometry.working_scale:.1f}\" per working pixel, and this tool "
            f"is built around {settings.working_scale:.1f}\" "
            f"(tolerance {limit:.1f}\").\n"
            f"   Decimation only coarsens a scale, so there is nothing to be "
            f"done here by decimating differently: the pixel-valued "
            f"thresholds would be off by {geometry.working_scale / settings.working_scale:.1f}x "
            f"and the verdicts would be noise.\n"
            f"   A field this wide needs its own thresholds and a test for "
            f"chance alignments — see the wide-angle note in DETECTION.md.")


def load_frame(path: str) -> np.ndarray | None:
    """Return the first image plane of a FITS file, or None if it has none.

    Colour frames are averaged down to a single plane; we only ever look
    at shapes here, so keeping the channels apart would cost time and buy
    nothing.

    ``memmap=False`` on purpose. Seestar frames carry BZERO, so astropy
    cannot memory-map them anyway -- it has to read and rescale -- and
    asking for a mapping it will refuse costs a few milliseconds per
    frame for nothing.
    """
    with fits.open(path, memmap=False) as hdu_list:
        for hdu in hdu_list:
            if hdu.data is not None:
                data = hdu.data
                break
        else:
            return None

    if data.ndim == 3:
        data = np.mean(data, axis=0)
    return data


# ======================================================================
#  Component geometry
# ======================================================================

def _principal_axes(cxx: np.ndarray, cyy: np.ndarray,
                    cxy: np.ndarray) -> tuple[np.ndarray, ...]:
    """Eigen-decompose a whole field of symmetric 2x2 matrices at once.

    Closed form, because there is one of these per connected component
    and a dense field yields tens of thousands of them. Returns the major
    and minor eigenvalues and the unit vector along the major axis.
    """
    trace = cxx + cyy
    determinant = cxx * cyy - cxy * cxy
    spread = np.sqrt(np.maximum(trace * trace / 4.0 - determinant, 0.0))
    major = trace / 2.0 + spread
    minor = np.maximum(trace / 2.0 - spread, 0.0)

    # (cxy, major - cxx), except where cxy vanishes: the matrix is already
    # diagonal there and the axis is one of the two coordinate axes.
    off_diagonal = np.abs(cxy) > 1e-12
    axis_x = np.where(off_diagonal, cxy, np.where(cxx >= cyy, 1.0, 0.0))
    axis_y = np.where(off_diagonal, major - cxx, np.where(cxx >= cyy, 0.0, 1.0))
    norm = np.hypot(axis_x, axis_y)
    norm = np.where(norm == 0, 1.0, norm)
    return major, minor, axis_x / norm, axis_y / norm


def working_image(data: np.ndarray, step: int) -> np.ndarray:
    """Decimate a frame to the working scale.

    A packed copy rather than a view: it pays for itself several times
    over in the morphology that follows.

    One of the fingerprinted primitives -- see MEASUREMENT_PARTS. Every
    step that turns pixels into numbers lives in a function like this
    one, so that ``analyse_frame`` is left holding nothing but assembly
    and a case bundle does not go stale when the assembly changes.
    """
    return np.ascontiguousarray(data[::step, ::step])


def bright_structure(small: np.ndarray,
                     settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Top-hat the working image and threshold what is left standing.

    The opening erases the point sources; the residual keeps trails,
    haze, and anything that is not a well-behaved star. The floor is a
    percentile of that residual rather than a multiple of the noise --
    the residual carries column structure, so it is not white noise and a
    sigma would not mean what it says.

    Returns the residual and the mask above the floor.
    """
    opening = (settings.opening_size,) * 2
    cleaned = small - grey_opening(small, size=opening)
    return cleaned, cleaned > high_percentile(cleaned, settings.bright_percentile)


def measure_components(labels: np.ndarray, mask: np.ndarray,
                       shape: tuple[int, int], count: int,
                       min_pixels: int = 3) -> dict[str, np.ndarray]:
    """Describe every connected component at once, by its principal axis.

    Orientation-independent by construction, which matters more than it
    sounds: a satellite trail at 45 degrees has an almost square bounding
    box, so a plain height/width ratio misses it entirely.

    A dense field yields around 22,000 components, of which barely 1,400
    clear the size floor. Handling them one at a time in Python used to
    account for half the script's runtime. Here the second-order moments
    are accumulated with ``bincount`` -- same numbers, an order of
    magnitude faster.

    Length and width are geometric extents along the component's own
    axes, which is what a trail needs; stars are measured elsewhere, from
    flux-weighted moments, because an extent quantised to whole pixels is
    useless on an object three pixels across.

    Returns arrays indexed by component number. ``valid`` marks the
    background and anything below ``min_pixels``; the other arrays are
    meaningless where it is False.
    """
    rows, cols = np.nonzero(mask)
    size = count + 1

    if len(rows) == 0:
        empty = np.zeros(size, dtype=float)
        return {"valid": np.zeros(size, dtype=bool),
                "cx": empty, "cy": empty, "length": empty, "width": empty,
                "ratio": empty, "pixels": empty,
                "touches_edge": np.zeros(size, dtype=bool),
                "axis_x": empty, "axis_y": empty}

    component = labels[rows, cols]
    x = cols.astype(np.float64)
    y = rows.astype(np.float64)

    def total(weights: np.ndarray) -> np.ndarray:
        return np.bincount(component, weights=weights, minlength=size)

    pixel_count = np.bincount(component, minlength=size).astype(np.float64)
    valid = pixel_count >= min_pixels
    safe_count = np.where(valid, pixel_count, 1.0)

    sum_x, sum_y = total(x), total(y)
    cx, cy = sum_x / safe_count, sum_y / safe_count

    # Covariance, normalised by (n-1) to match np.cov.
    dof = np.maximum(pixel_count - 1.0, 1.0)
    cov_xx = (total(x * x) - sum_x * sum_x / safe_count) / dof
    cov_yy = (total(y * y) - sum_y * sum_y / safe_count) / dof
    cov_xy = (total(x * y) - sum_x * sum_y / safe_count) / dof
    _, _, axis_x, axis_y = _principal_axes(cov_xx, cov_yy, cov_xy)

    # Project every pixel onto the axis of its own component.
    px, py = axis_x[component], axis_y[component]
    dx = x - cx[component]
    dy = y - cy[component]
    along = dx * px + dy * py
    across = -dx * py + dy * px

    def extent(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = np.full(size, np.inf)
        high = np.full(size, -np.inf)
        np.minimum.at(low, component, values)
        np.maximum.at(high, component, values)
        return low, high

    along_low, along_high = extent(along)
    across_low, across_high = extent(across)
    row_low, row_high = extent(y)
    col_low, col_high = extent(x)

    length = np.where(valid, along_high - along_low, 0.0)
    width = np.maximum(np.where(valid, across_high - across_low, 0.0), 1.0)

    height, image_width = shape
    touches_edge = ((row_low <= 1) | (row_high >= height - 2) |
                    (col_low <= 1) | (col_high >= image_width - 2)) & valid

    return {"valid": valid, "cx": cx, "cy": cy, "length": length,
            "width": width, "ratio": length / width, "pixels": pixel_count,
            "touches_edge": touches_edge, "axis_x": axis_x, "axis_y": axis_y}


# ======================================================================
#  Star shape
# ======================================================================

@dataclass(frozen=True)
class StarShape:
    """What the stars of one frame look like, at full resolution.

    ``elongation`` is the discriminant the trailing verdict rests on:
    the median of sqrt(major/minor) over a few hundred flux-weighted
    stars. ``radius`` is their median size and ``coherence`` says whether
    they all lean the same way -- both reported, neither used to decide.
    """

    count: int
    radius: float
    elongation: float
    coherence: float
    blobs: list[dict] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.count > 0

    @classmethod
    def none(cls) -> "StarShape":
        return cls(count=0, radius=0.0, elongation=0.0, coherence=0.0, blobs=[])


def high_percentile(image: np.ndarray, percentile: float) -> float:
    """The detection threshold, from a sample rather than every pixel.

    ``np.percentile`` sorts, so on a two-megapixel frame it costs as much
    as several morphological passes. Every third pixel in each direction
    is a ninth of the work and, measured over real frames, moves the
    threshold by about 0.2% -- far inside the noise on the pixels it
    then selects.
    """
    sample = image[::3, ::3] if image.size > 500_000 else image
    return float(np.percentile(sample, percentile))


def centre_crop_bounds(shape: tuple[int, int],
                       settings: Settings) -> tuple[int, int, int, int]:
    """``(left, top, right, bottom)`` of the full-resolution centre crop.

    Shared with the debug overlay, which has to draw the stars back at
    their place in the whole frame and therefore needs the same origin.
    """
    height, width = shape
    half_height = int(height * settings.sharpness_centre_fraction / 2)
    half_width = int(width * settings.sharpness_centre_fraction / 2)
    return (width // 2 - half_width, height // 2 - half_height,
            width // 2 + half_width, height // 2 + half_height)


def crop_centre(data: np.ndarray, settings: Settings) -> np.ndarray:
    """The central region the full-resolution pass works on."""
    left, top, right, bottom = centre_crop_bounds(data.shape, settings)
    # Contiguous on purpose: scipy's morphology runs about 12% faster on
    # a packed array than on a strided view of the original frame, and
    # the copy itself costs under 3 ms.
    return np.ascontiguousarray(data[top:bottom, left:right])


def measure_stars(centre: np.ndarray, settings: Settings,
                  thresholds: Thresholds) -> StarShape:
    """Measure the stars of a full-resolution crop by their flux moments.

    Two things need the resolution that decimation throws away, and both
    happen here so the frame is only swept once.

    The first is the shape of the stars, which is the whole trailing
    verdict. An earlier version measured it as a bounding box on the
    decimated image and it did not separate anything: a star spans two or
    three pixels there, so its extent takes a handful of discrete values
    with nothing in between them. Flux-weighted second moments at full
    resolution are continuous, and on the labelled session they put every
    frame the owner called trailed above 1.38x the session's sharpest
    quarter, against 1.04x for the median frame.

    The second is the star positions the doubling test needs. Decimated,
    a mild doubling sinks below centroid noise -- 8% of stars against 28%
    at full resolution on the same genuinely doubled frame.

    Components touching the edge of the crop are dropped: they are cut
    off, and a truncated star reads as an elongated one.
    """
    if centre.size == 0:
        return StarShape.none()

    opening = (thresholds.sharpness_opening_size,) * 2
    cleaned = (centre - grey_opening(centre, size=opening)).astype(np.float32)
    mask = cleaned > high_percentile(cleaned, settings.sharpness_percentile)
    labels, count = label(mask, structure=EIGHT_CONNECTED)
    if count == 0:
        return StarShape.none()

    rows, cols = np.nonzero(mask)
    component = labels[rows, cols]
    weight = cleaned[rows, cols].astype(np.float64)
    size = count + 1

    def total(values: np.ndarray) -> np.ndarray:
        return np.bincount(component, weights=values, minlength=size)

    pixel_count = np.bincount(component, minlength=size)
    flux = total(weight)
    usable = (pixel_count >= max(settings.min_component_pixels, 4)) & (flux > 0)
    safe_flux = np.where(usable, flux, 1.0)

    x = cols.astype(np.float64)
    y = rows.astype(np.float64)
    cx = total(weight * x) / safe_flux
    cy = total(weight * y) / safe_flux
    dx = x - cx[component]
    dy = y - cy[component]
    cov_xx = total(weight * dx * dx) / safe_flux
    cov_yy = total(weight * dy * dy) / safe_flux
    cov_xy = total(weight * dx * dy) / safe_flux
    major, minor, _, _ = _principal_axes(cov_xx, cov_yy, cov_xy)

    height, width = centre.shape
    on_edge = np.zeros(size, dtype=bool)
    border = (rows <= 1) | (rows >= height - 2) | (cols <= 1) | (cols >= width - 2)
    on_edge[component[border]] = True

    stars = (usable & ~on_edge & (minor > 1e-3)
             & (major < thresholds.star_max_length_full ** 2))
    if int(stars.sum()) < settings.sharpness_min_stars:
        return StarShape.none()

    index = np.nonzero(stars)[0]
    major, minor = major[index], minor[index]
    elongation = np.sqrt(major / minor)
    radius = np.sqrt(major + minor)

    # Orientation of each star, doubled so that a direction and its
    # opposite agree, and averaged with the elongation as the weight:
    # a round star has no meaningful orientation and should not vote.
    angle = 0.5 * np.arctan2(2.0 * cov_xy[index], cov_xx[index] - cov_yy[index])
    lean = np.maximum(elongation - 1.0, 0.0)
    coherence = (float(abs(np.sum(lean * np.exp(2j * angle))) / lean.sum())
                 if lean.sum() > 0 else 0.0)

    blobs = [{"cx": float(cx[i]), "cy": float(cy[i]), "size": float(flux[i]),
              "elongation": float(e), "angle": float(a)}
             for i, e, a in zip(index, elongation, angle)]
    return StarShape(count=len(index),
                     radius=float(np.median(radius)),
                     elongation=float(np.median(elongation)),
                     coherence=coherence,
                     blobs=blobs)


# ======================================================================
#  Straight tracks
# ======================================================================

@dataclass(frozen=True)
class Segment:
    """One elongated component: a whole trail, or a fragment of one."""

    label: int      # its component number in the labelled image
    cx: float
    cy: float
    length: float
    width: float
    ratio: float
    axis_x: float
    axis_y: float
    touches_edge: bool


@dataclass(frozen=True)
class TrackProfile:
    """How the light is spread along and across a track.

    Everything here is normalised by something the same frame provides --
    its noise, its stars -- so the numbers mean the same on any Seestar,
    at any gain, under any sky.
    """

    head_ratio: float = 1.0     # brightest slice over the typical one
    head_position: float = 0.5  # where that slice falls, 0 and 1 being the ends
    taper: float = 1.0          # brighter half over dimmer half
    glow: float = 0.0           # flux per pixel of span, in units of the noise
    peak: float = 0.0           # brightest pixel on the track, same units
    width: float = 0.0          # flux-weighted width across the line, working px
    green: float = 0.0          # green glow beside the line, in sigma; see measure_track_green


@dataclass(frozen=True)
class Track:
    """A straight line across the field, reassembled from its fragments.

    The line itself is kept -- midpoint and unit vector, in working
    pixels -- so the brightness along it can be sampled afterwards
    without repeating the search.
    """

    span: float          # working px, end to end
    coverage: float      # share of the span actually lit
    residual: float      # RMS distance of the pieces from the line
    fragments: int
    max_width: float     # working px
    touches_edge: bool
    tilt: float          # degrees, measured from the x axis
    centre_x: float
    centre_y: float
    unit_x: float
    unit_y: float
    members: tuple[int, ...] = ()   # component ids making it up
    profile: TrackProfile | None = None


def _line_direction(cx: np.ndarray, cy: np.ndarray,
                    weights: np.ndarray) -> tuple[float, float]:
    """Unit vector along the best-fit line through a set of centres."""
    total = weights.sum()
    mean_x = float((cx * weights).sum() / total)
    mean_y = float((cy * weights).sum() / total)
    dx, dy = cx - mean_x, cy - mean_y
    axis_x, axis_y = _principal_axes(np.array([float((weights * dx * dx).sum() / total)]),
                                     np.array([float((weights * dy * dy).sum() / total)]),
                                     np.array([float((weights * dx * dy).sum() / total)]))[2:]
    return float(axis_x[0]), float(axis_y[0])


def spatial_summary(small: np.ndarray, measures: dict,
                    settings: Settings) -> dict:
    """Where in the frame the stars and the sky are, not just how much.

    Everything a verdict currently rests on is a whole-frame aggregate: a
    median sky, one standard deviation, one star count. A cloud covering a
    fifth of the field barely moves any of them, and trailing confined to
    the corners is invisible to a measurement taken in the middle. Neither
    hypothesis can be tested from a bundle that carries aggregates only,
    and testing them from the frames means keeping four gigabytes a night.

    So this writes down the resolved primitives instead: a grid of tile
    statistics, the quantiles of the sky and of the component sizes, and a
    wider star map than the shift needs. None of it decides anything
    today. All of it is nearly free -- every number here comes out of
    arrays the pipeline has already built -- and it is what lets the next
    idea be scored without re-reading a single FITS.

    The tile background is a median over every fourth pixel: a background
    does not need eight thousand samples to be pinned down, and the full
    grid would otherwise cost as much as a second pass over the frame.
    """
    grid = settings.export_grid
    height, width = small.shape
    valid = measures["valid"]
    cx, cy = measures["cx"][valid], measures["cy"][valid]
    length, ratio = measures["length"][valid], measures["ratio"][valid]
    pixels = measures["pixels"][valid]

    tile_x = np.clip((cx / width * grid).astype(int), 0, grid - 1)
    tile_y = np.clip((cy / height * grid).astype(int), 0, grid - 1)
    tile_of = tile_y * grid + tile_x
    counts = np.bincount(tile_of, minlength=grid * grid)

    tiles = []
    for index in range(grid * grid):
        row, column = divmod(index, grid)
        patch = small[row * height // grid:(row + 1) * height // grid:4,
                      column * width // grid:(column + 1) * width // grid:4]
        here = tile_of == index
        tiles.append([int(counts[index]),
                      round(float(np.median(patch)), 1) if patch.size else 0.0,
                      round(float(np.std(patch)), 2) if patch.size else 0.0,
                      round(float(np.median(length[here])), 2) if here.any() else 0.0,
                      round(float(np.median(ratio[here])), 2) if here.any() else 0.0])

    deciles = np.arange(10, 100, 10)
    brightest = np.argsort(-pixels)[:settings.export_star_map]
    return {
        "grid": grid,
        # count, background, noise, median length, median ratio -- per tile,
        # row-major from the top-left of the working image.
        "tiles": tiles,
        "sky": [round(float(v), 1)
                for v in np.percentile(small[::4, ::4], [5, 25, 50, 75, 95])],
        "component_pixels": [round(float(v), 1)
                             for v in np.percentile(pixels, deciles)]
        if pixels.size else [],
        "component_ratio": [round(float(v), 2)
                            for v in np.percentile(ratio, deciles)]
        if ratio.size else [],
        # A star map wider than the shift vote uses. Kept separate on
        # purpose: shift_star_count feeds measure_field_shift, and moving
        # it would change a measurement rather than add one.
        "star_map": [[round(float(cx[i]), 1), round(float(cy[i]), 1),
                      int(pixels[i])] for i in brightest],
    }


def select_track_segments(segments: Sequence[Segment], settings: Settings,
                          thresholds: Thresholds) -> list[Segment]:
    """The components allowed to join a line: long enough, thin enough.

    ``analyse_frame`` applies the same two bars with numpy, while it
    still has the measurement arrays and before anything becomes a Python
    object; a test pins the two together. The rule is spelled out here
    because the case export keeps a wider net than the run uses, and
    whoever replays a bundle has to apply the bar of the day -- with this
    rule, not with a second copy of it.
    """
    return [segment for segment in segments
            if segment.length >= thresholds.segment_min_length
            and segment.ratio >= settings.segment_min_ratio]


def find_track(segments: Sequence[Segment], settings: Settings,
               thresholds: Thresholds) -> Track | None:
    """Find the longest straight line the fragments agree on, if any.

    The Seestar's field is a couple of degrees across. Nothing that flies
    through it -- satellite, meteor, aircraft -- curves noticeably at that
    scale, so a real crossing prints one straight line, and when the
    threshold breaks that line into pieces every piece stays on it *and*
    points along it.

    Both halves of that matter, and neither is enough on its own: trailed
    stars are all parallel too, so a line drawn along their common
    direction does gather a couple of them by luck. What luck does not
    produce is a *whole dotted line* of them, which is why a track has to
    be either lit end to end or made of many pieces.

    Only a convincingly thin fragment may start a line -- a merged pair
    of stars must never be able to -- but once the line exists any
    fragment may join it. That asymmetry is what recovers a faint trail:
    it arrives as a string of specks barely longer than wide, and at the
    strict bar only two or three of them survive.

    Each seed's own axis gives the first guess at the direction; the line
    is then refit through whatever it gathered, because a short fragment
    fixes its direction only to within a few degrees and a few degrees is
    a long way off at the other end of the frame.

    Works in working-image pixels. Returns the longest acceptable track,
    or None.
    """
    if not segments:
        return None

    cx = np.array([s.cx for s in segments])
    cy = np.array([s.cy for s in segments])
    length = np.array([s.length for s in segments])
    width = np.array([s.width for s in segments])
    axis_x = np.array([s.axis_x for s in segments])
    axis_y = np.array([s.axis_y for s in segments])
    ratio = np.array([s.ratio for s in segments])
    on_edge = np.array([s.touches_edge for s in segments])

    # The longest thin fragments carry the best-defined direction, and a
    # real track owns at least one of them. Anything past a few dozen is
    # a trailed frame, where the extra seeds only cost time.
    candidates = np.nonzero(ratio >= settings.seed_min_ratio)[0]
    seeds = candidates[np.argsort(-length[candidates])][:40]

    best: Track | None = None
    for seed in seeds:
        direction = (axis_x[seed], axis_y[seed])
        origin_x, origin_y = cx[seed], cy[seed]
        members = None

        # Gather wide, refit, gather tight. One refit is enough: the
        # second direction comes from centres spread over the whole
        # track, not from one fragment a few pixels long.
        for tolerance in (3.0 * thresholds.track_tolerance,
                          thresholds.track_tolerance):
            unit_x, unit_y = direction
            offset_x, offset_y = cx - origin_x, cy - origin_y
            across = np.abs(offset_x * -unit_y + offset_y * unit_x)
            aligned = (np.abs(axis_x * unit_x + axis_y * unit_y)
                       >= thresholds.track_cos_tilt)
            members = (across <= tolerance) & aligned
            if members.sum() >= 2:
                direction = _line_direction(cx[members], cy[members],
                                            length[members])
            elif not members[seed]:
                break

        if members is None or not members.any():
            continue

        unit_x, unit_y = direction
        along = ((cx[members] - origin_x) * unit_x
                 + (cy[members] - origin_y) * unit_y)
        half = length[members] / 2.0
        span = float((along + half).max() - (along - half).min())
        fragments = int(members.sum())
        coverage = min(float(length[members].sum()) / span, 1.0) if span else 0.0

        # How far the pieces stray from the line they were gathered on.
        # Two points always fit a line exactly, so it says nothing below
        # three -- which is why the coverage branch exists.
        across_line = ((cx[members] - cx[members].mean()) * -unit_y
                       + (cy[members] - cy[members].mean()) * unit_x)
        residual = float(np.sqrt((across_line ** 2).mean()))

        if span < thresholds.track_min_span:
            continue
        if fragments >= settings.track_min_fragments:
            # Enough pieces to tell whether they form a line. They must,
            # however well lit the result looks: a trailed frame once
            # produced a 60%-lit "track" out of ten star dashes scattered
            # 5 px either side of it.
            if residual > thresholds.track_max_residual:
                continue
        elif coverage < settings.track_min_coverage:
            # One or two pieces fit any line exactly, so the only
            # evidence left is that the trail is continuous.
            continue

        if best is None or span > best.span:
            middle = (float((along + half).max()) + float((along - half).min())) / 2.0
            best = Track(span=span,
                         coverage=coverage,
                         residual=residual,
                         fragments=fragments,
                         max_width=float(width[members].max()),
                         touches_edge=bool(on_edge[members].any()),
                         tilt=math.degrees(math.atan2(unit_y, unit_x)) % 180.0,
                         centre_x=float(origin_x + middle * unit_x),
                         centre_y=float(origin_y + middle * unit_y),
                         unit_x=float(unit_x),
                         unit_y=float(unit_y),
                         members=tuple(int(segments[k].label)
                                       for k in np.nonzero(members)[0]))

    return best


def measure_track_profile(cleaned: np.ndarray, labels: np.ndarray,
                          track: Track, noise: float) -> TrackProfile:
    """Measure how bright a track is, how wide, and how evenly it burns.

    Two things separate the objects that cross a frame, and only one of
    them is geometry.

    A satellite reflects sunlight for the whole exposure, so a fixed
    amount of light is smeared over the whole trail. A meteor deposits
    everything it has in well under a second: the same total lands on far
    fewer pixels, so the trail is brighter per unit length, and a bright
    one saturates and blooms wider than the seeing disc. ``glow``,
    ``peak`` and ``width`` are that difference.

    The shape of the profile -- ``head_ratio``, ``head_position``,
    ``taper`` -- is the other half of it, and the weaker half. It was
    tried on its own as a meteor test and failed: satellites glint in
    mid-passage, and a star sitting on the trail merges into the same
    connected component, so restricting the sum to the track's own pixels
    does not keep it out. (Summing a band around the line was worse: four
    of the first six peaks that way were stars.) Read it as supporting
    evidence for a decay from one end, never on its own.
    """
    if not track.members or noise <= 0:
        return TrackProfile()

    rows, cols = np.nonzero(np.isin(labels, track.members))
    if rows.size < 6:
        return TrackProfile()

    brightness = cleaned[rows, cols].astype(np.float64)
    total = float(brightness.sum())
    if total <= 0:
        return TrackProfile()

    offset_x = cols - track.centre_x
    offset_y = rows - track.centre_y
    along = offset_x * track.unit_x + offset_y * track.unit_y
    across = -offset_x * track.unit_y + offset_y * track.unit_x

    # Flux-weighted, so it is continuous rather than quantised the way a
    # bounding box across three pixels would be. Doubled to read as a
    # full width rather than a half-width.
    centre = float((brightness * across).sum() / total)
    spread = float((brightness * (across - centre) ** 2).sum() / total)
    width = 2.0 * math.sqrt(max(spread, 0.0))

    reach = track.span / 2.0
    slices = int(min(max(track.span / 10.0, 6), 40))
    position = np.clip(((along + reach) / track.span * slices).astype(int),
                       0, slices - 1)
    flux = np.bincount(position, weights=brightness, minlength=slices)

    lit = flux[flux > 0]
    if lit.size < 3:
        return TrackProfile(glow=total / track.span / noise,
                            peak=float(brightness.max()) / noise,
                            width=width)

    typical = float(np.median(lit))
    brightest = int(np.argmax(flux))
    first, second = float(flux[:slices // 2].sum()), float(flux[slices // 2:].sum())
    low, high = min(first, second), max(first, second)

    return TrackProfile(
        head_ratio=float(flux[brightest] / typical) if typical > 0 else 1.0,
        head_position=brightest / max(slices - 1, 1),
        taper=high / low if low > 0 else float(slices),
        glow=total / track.span / noise,
        peak=float(brightness.max()) / noise,
        width=width,
    )


def measure_track_green(data: np.ndarray, track: Track, step: int,
                        bayer: str) -> float:
    """How strongly a green glow runs alongside a track, in sigma.

    What gave both confirmed meteors away to the eye, before any number
    did: a diffuse green band beside the trail, on one side only, along
    its whole length -- the 557.7 nm oxygen line, which lingers for
    seconds after a meteor and drifts with the upper-air wind. A
    satellite only reflects sunlight and cannot make it. Measured on the
    raw mosaic, where green photosites stay apart from red and blue.

    For each third of the track and each side of it, the green excess --
    median of the green photosites minus median of the others, over a
    band 4 to 24 px from the line -- relative to the same excess 60-80 px
    away, which carries the sky's own colour. A side counts only by its
    weakest third, so a star or a glint on part of the trail cannot fake
    it. The best side is then divided by the scatter of the same quantity
    on empty bands 40-80 px out. Medians throughout, because stars sit on
    every band.

    Measured on M 45: 18.8 and 9.4 sigma for the two confirmed meteors,
    at most 1.3 over 31 satellites. Zero when there is no colour to read
    or too little sky around the track.
    """
    if data.ndim != 2 or not bayer or track.span <= 0:
        return 0.0
    half = track.span * step / 2.0 * 0.9
    if half < 30:
        return 0.0
    along = np.arange(-half, half, 1.0)
    across = np.arange(-80.0, 81.0, 1.0)
    grid_along, grid_across = np.meshgrid(along, across, indexing="ij")
    x = np.rint(track.centre_x * step + grid_along * track.unit_x
                - grid_across * track.unit_y).astype(int)
    y = np.rint(track.centre_y * step + grid_along * track.unit_y
                + grid_across * track.unit_x).astype(int)
    inside = (x >= 0) & (x < data.shape[1]) & (y >= 0) & (y < data.shape[0])
    values = np.zeros(x.shape, dtype=np.float64)
    values[inside] = data[y[inside], x[inside]]
    green_sites = np.array([c == "G" for c in bayer])[(y % 2) * 2 + (x % 2)]

    def excess(mask: np.ndarray) -> float | None:
        green, other = mask & green_sites, mask & ~green_sites
        if green.sum() < 20 or other.sum() < 20:
            return None
        return float(np.median(values[green]) - np.median(values[other]))

    thirds = np.minimum((np.arange(along.size) * 3) // along.size, 2)
    sides: dict[int, list[float]] = {1: [], -1: []}
    empty: list[float] = []
    for third in range(3):
        rows = (thirds == third)[:, None] & inside
        sky = excess(rows & (np.abs(grid_across) >= 60))
        if sky is None:
            return 0.0
        for sign in (1, -1):
            side = sign * grid_across
            band = excess(rows & (side >= 4) & (side < 24))
            if band is None:
                return 0.0
            sides[sign].append(band - sky)
            for low, high in ((40, 60), (60, 80)):
                null = excess(rows & (side >= low) & (side < high))
                if null is not None:
                    empty.append(null - sky)
    if len(empty) < 6:
        return 0.0
    scatter = float(np.std(empty))
    if scatter <= 0:
        return 0.0
    return max(min(values_) for values_ in sides.values()) / scatter


# ======================================================================
#  Star doubling
# ======================================================================

def points_are_collinear(points: np.ndarray, max_scatter: float) -> bool:
    """Do these centres fall on one straight line? A single point does."""
    if len(points) < 2:
        return True

    centred = points - points.mean(axis=0)
    covariance = np.cov(centred.T)
    if np.any(np.isnan(covariance)):
        return False

    _, eigenvectors = np.linalg.eigh(covariance)
    minor_axis = eigenvectors[:, 0]
    return float(np.std(centred @ minor_axis)) < max_scatter


def find_star_doubling(stars: Sequence[dict], settings: Settings,
                       thresholds: Thresholds) -> dict | None:
    """Detect stars printed twice by a jolt during the exposure.

    A sharp knock -- wind, mount backlash -- moves the telescope in one
    instant rather than dragging it. The stars stay round; each one
    simply appears twice, offset by the same vector everywhere in the
    frame. Nothing about that is elongated, so neither the shape test nor
    the track detector sees it.

    No registration and no star matching: we take the vector between
    every plausible pair of blobs, fix its sign so that A->B and B->A
    vote together, then look for the vector that gathers the most pairs.
    Real doubling shows up as a sharp peak shared by a large fraction of
    the stars; a crowded field without it produces vectors with no mode
    at all.

    All distances here are in full-resolution pixels.

    Returns ``{dx, dy, distance, pairs, fraction}`` or None.
    """
    min_pairs = settings.doubling_min_pairs
    if stars is None or len(stars) < 2 * min_pairs:
        return None

    # Brightest blobs first: their centroids are the best defined.
    brightest = sorted(stars, key=lambda s: s["size"],
                       reverse=True)[:settings.doubling_max_stars]
    points = np.array([(s["cx"], s["cy"]) for s in brightest], dtype=float)
    count = len(points)

    separation = points[:, None, :] - points[None, :, :]
    distance = np.hypot(separation[:, :, 0], separation[:, :, 1])
    left, right = np.triu_indices(count, k=1)
    usable = ((distance[left, right] >= thresholds.doubling_min_distance) &
              (distance[left, right] <= thresholds.doubling_max_distance))
    if usable.sum() < min_pairs:
        return None

    left, right = left[usable], right[usable]
    vectors = separation[right, left]

    # An offset and its opposite describe the same doubling; without
    # this the peak would split into two mirrored halves.
    flip = (vectors[:, 0] < 0) | ((vectors[:, 0] == 0) & (vectors[:, 1] < 0))
    vectors = np.where(flip[:, None], -vectors, vectors)

    # How many other vectors each one agrees with, within the tolerance.
    # A KD-tree gives the same count as a full pairwise distance matrix
    # and the same members for the winner, without the matrix: that was
    # pairs squared in float64 -- 80 MB at the 1,800 usable pairs a dense
    # field yields, growing quadratically from there -- and it was most
    # of what the vote used to cost. Checked vector by vector against the
    # matrix over 2,900 frames: identical results.
    tolerance = thresholds.doubling_tolerance
    tree = cKDTree(vectors)
    support = tree.query_ball_point(vectors, tolerance, return_length=True)
    winner = int(np.argmax(support))
    members = np.zeros(len(vectors), dtype=bool)
    members[tree.query_ball_point(vectors[winner], tolerance)] = True

    pair_count = int(members.sum())
    if pair_count < min_pairs:
        return None

    involved = np.unique(np.concatenate((left[members], right[members])))
    fraction = len(involved) / float(count)
    if fraction < settings.doubling_min_fraction:
        return None

    # A satellite trail chopped into segments by the threshold produces
    # blobs strung out at regular intervals along one line, so the vector
    # from one segment to the next repeats and looks exactly like a
    # doubling. Real doubling touches stars all over the frame; the
    # segments of a trail stay on the trail. Measured on real data: this
    # is what three of the four detections on a 230-frame session turned
    # out to be.
    if points_are_collinear(points[involved], thresholds.doubling_max_scatter):
        return None

    dx, dy = vectors[members].mean(axis=0)
    return {"dx": float(dx), "dy": float(dy),
            "distance": float(np.hypot(dx, dy)),
            "pairs": pair_count, "fraction": float(fraction)}


# ======================================================================
#  Field shift between consecutive frames
# ======================================================================

def measure_field_shift(previous: np.ndarray | None, current: np.ndarray | None,
                        thresholds: Thresholds) -> tuple[float, float, float] | None:
    """Measure how far the field moved between two consecutive frames.

    A gust of wind or play in the mount nudges the pointing from one
    exposure to the next. This is reported, never held against a frame:
    the stars stay round, stacking registers the shift away and the edges
    get cropped anyway. It earns its place in the CSV because it is the
    context that explains a trailed frame, and because it is nearly free.

    Not registration: nothing is transformed and no astrometry is solved.
    We take the vectors between every pair of stars across the two frames
    and find the dominant one with a 2D histogram, which is linear in the
    number of pairs and runs in a couple of milliseconds.

    Works in working-image pixels. Returns ``(dx, dy, confidence)`` or
    None; confidence is the share of stars following that vector, low on
    a large shift where most stars have left the frame.
    """
    if previous is None or current is None:
        return None
    if len(previous) < 8 or len(current) < 8:
        return None

    vectors = (current[None, :, :] - previous[:, None, :]).reshape(-1, 2)
    radius = thresholds.shift_search_radius
    in_range = (np.abs(vectors[:, 0]) <= radius) & (np.abs(vectors[:, 1]) <= radius)
    vectors = vectors[in_range]
    if len(vectors) < 8:
        return None

    step = thresholds.shift_vote_bin
    edges = np.arange(-radius, radius + step, step)
    histogram, _, _ = np.histogram2d(vectors[:, 0], vectors[:, 1], bins=[edges, edges])
    i, j = np.unravel_index(np.argmax(histogram), histogram.shape)

    # Sub-bin refinement: average the winning bin and its immediate
    # neighbourhood, so the result is not quantised to the vote step.
    centre_x = edges[i] + step / 2.0
    centre_y = edges[j] + step / 2.0
    near = ((np.abs(vectors[:, 0] - centre_x) <= step) &
            (np.abs(vectors[:, 1] - centre_y) <= step))
    if not near.any():
        return None

    dx, dy = vectors[near].mean(axis=0)
    confidence = float(near.sum()) / min(len(previous), len(current))
    return float(dx), float(dy), confidence


# ======================================================================
#  Per-frame analysis
# ======================================================================

@dataclass
class FrameStats:
    """Everything one frame contributes to the session-wide decision."""

    name: str
    path: str
    background: float          # median pixel value
    peak: float
    noise: float               # standard deviation
    stars: StarShape
    track: Track | None = None
    doubling: dict | None = None
    star_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    field_shift: tuple[float, float, float] | None = None
    # Set by mark_recurring_tracks: the same line is on a neighbouring
    # frame, so the track cannot be a meteor or an aircraft.
    track_recurs: bool = False
    segments: list[Segment] = field(default_factory=list)  # debug mode only
    # The same components under the wider export bars, kept only for
    # --export-cases so a replay can move the join bar downwards too.
    export_segments: list[Segment] = field(default_factory=list)
    # Resolved primitives, --export-cases only. Nothing reads them yet;
    # see spatial_summary for why they are written down anyway.
    spatial: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerContext:
    """What every worker process needs, sent once at pool start-up."""

    settings: Settings
    geometry: Geometry
    thresholds: Thresholds


_context: WorkerContext | None = None


def _init_worker(context: WorkerContext) -> None:
    global _context
    _context = context


def analyse_frame(path: str) -> FrameStats | None:
    """Measure one frame. Runs in a worker process; never raises."""
    name = os.path.basename(path)
    assert _context is not None
    settings = _context.settings
    thresholds = _context.thresholds
    try:
        data = load_frame(path)
        if data is None:
            return None

        small = working_image(data, _context.geometry.step)
        cleaned, mask = bright_structure(small, settings)
        labels, count = label(mask, structure=EIGHT_CONNECTED)
        measures = measure_components(labels, mask, small.shape, count,
                                      settings.min_component_pixels)

        # Selection stays vectorised; only the handful of components that
        # could belong to a track are turned into Python objects. The
        # landmark selection further down keys off this mask, so it holds
        # the run's own bars whatever the export widens them to.
        is_segment = (measures["valid"] &
                      (measures["length"] >= thresholds.segment_min_length) &
                      (measures["ratio"] >= settings.segment_min_ratio))

        def to_segments(chosen: np.ndarray) -> list[Segment]:
            return [Segment(label=int(i),
                            cx=float(measures["cx"][i]),
                            cy=float(measures["cy"][i]),
                            length=float(measures["length"][i]),
                            width=float(measures["width"][i]),
                            ratio=float(measures["ratio"][i]),
                            axis_x=float(measures["axis_x"][i]),
                            axis_y=float(measures["axis_y"][i]),
                            touches_edge=bool(measures["touches_edge"][i]))
                    for i in np.nonzero(chosen)[0]]

        if settings.export_cases:
            # A wider net for the bundle, then the bar of the day applied
            # to it -- which hands find_track the very list it would have
            # had anyway, since the two filters compare the same floats.
            export_segments = to_segments(
                measures["valid"] &
                (measures["length"] >= thresholds.segment_min_length
                 * settings.export_segment_length_factor) &
                (measures["ratio"] >= min(settings.segment_min_ratio,
                                          settings.export_segment_min_ratio)))
            segments = select_track_segments(export_segments, settings, thresholds)
            spatial = spatial_summary(small, measures, settings)
        else:
            export_segments = []
            spatial = {}
            segments = to_segments(is_segment)

        stars = measure_stars(crop_centre(data, settings), settings, thresholds)
        doubling = find_star_doubling(stars.blobs, settings, thresholds)
        # Several hundred blobs per frame are wanted here and nowhere
        # else; pickling them back to the parent would cost more than
        # measuring them did. Debug mode keeps them, to draw them.
        if not (settings.debug or settings.export_cases):
            stars = replace(stars, blobs=[])

        track = find_track(segments, settings, thresholds)
        if track is not None:
            profile = measure_track_profile(cleaned, labels, track,
                                            float(np.std(small)))
            track = replace(track, profile=replace(
                profile, green=measure_track_green(
                    data, track, _context.geometry.step,
                    _context.geometry.bayer)))

        # Keep the compact blobs as landmarks for the frame-to-frame
        # shift. Only this small array leaves the worker, not the
        # thousands of components behind it.
        is_landmark = (measures["valid"] & ~is_segment &
                       (measures["length"] <= thresholds.star_max_length))
        landmark_indices = np.nonzero(is_landmark)[0]
        if landmark_indices.size:
            area = (measures["length"][landmark_indices] *
                    measures["width"][landmark_indices])
            brightest = landmark_indices[np.argsort(-area)][:settings.shift_star_count]
            landmarks = np.column_stack((measures["cx"][brightest],
                                         measures["cy"][brightest]))
        else:
            landmarks = np.zeros((0, 2))

        return FrameStats(
            name=name,
            path=path,
            background=float(np.median(small)),
            peak=float(np.max(small)),
            noise=float(np.std(small)),
            stars=stars,
            track=track,
            doubling=doubling,
            star_positions=landmarks,
            segments=segments if settings.debug else [],
            export_segments=export_segments,
            spatial=spatial,
        )
    except MemoryError:
        # Told apart from a broken file because the remedy is different:
        # this frame is fine and the machine is not. Losing it silently
        # would leave a session short of frames while the summary still
        # counted them, so the caller reports the shortfall.
        print(f"⚠️ Out of memory on {name} — try --workers with a smaller "
              f"number, or close something.")
        return None
    except Exception as error:  # a bad file must not abort the session
        print(f"⚠️ Could not read {name}: {error}")
        return None


# ======================================================================
#  Session classification
# ======================================================================

@dataclass(frozen=True)
class SessionReference:
    """What a normal frame of this session looks like.

    Sky brightness and noise are taken at the median. The star shapes are
    taken at the first quartile instead -- the sharpest quarter of the
    night, near enough -- because they are the yardstick a fault is
    measured against, and a windy session where half the frames are
    smeared would otherwise quietly redefine smeared as normal.
    """

    background: float
    background_spread: float
    noise: float
    star_radius: float
    star_elongation: float
    # How many stars a good frame of this night shows. Taken at the third
    # quartile rather than the median so that a veiled night does not
    # quietly redefine "few stars" as normal -- the same reasoning as the
    # star shapes above. Defaulted so that a case bundle written before
    # this field existed still loads; cases_session fills it in.
    star_count: float = 0.0

    @classmethod
    def of(cls, frames: Sequence[FrameStats]) -> "SessionReference":
        shapes = [f.stars for f in frames if f.stars.measured]

        def sharpest_quarter(values: list[float]) -> float:
            return float(np.percentile(values, 25)) if values else 0.0

        return cls(
            background=float(np.median([f.background for f in frames])),
            background_spread=max(float(np.std([f.background for f in frames])), 1.0),
            noise=float(np.median([f.noise for f in frames])),
            star_radius=sharpest_quarter([s.radius for s in shapes]),
            star_elongation=sharpest_quarter([s.elongation for s in shapes]),
            star_count=float(np.percentile([f.stars.count for f in frames], 75))
            if frames else 0.0,
        )


@dataclass(frozen=True)
class Alert:
    """One thing worth knowing about a frame."""

    category: str
    message: str


def _ratio(value: float, reference: float) -> float:
    """How many times the reference, or 1.0 when there is nothing to compare."""
    return value / reference if reference > 0 and value > 0 else 1.0


def classify_frame(frame: FrameStats, reference: SessionReference,
                   settings: Settings,
                   thresholds: Thresholds) -> tuple[list[Alert], dict]:
    """Decide which alerts one frame raises.

    Each alert is independent: a frame can be hazy, trailed and crossed
    by a satellite at the same time, and saying so is more useful than
    forcing a single verdict.

    Returns the alerts and the row that ends up in the CSV report.
    """
    alerts: list[Alert] = []
    haze_sigma = ((frame.background - reference.background)
                  / reference.background_spread)
    elongation_ratio = _ratio(frame.stars.elongation, reference.star_elongation)
    radius_ratio = _ratio(frame.stars.radius, reference.star_radius)
    # Below 1 the stars carry less of the frame than they should, which is
    # what a veil looks like. _ratio answers 1.0 when there is nothing to
    # compare, which is the right no-alert answer for both of these.
    contrast = _ratio(frame.noise, reference.noise)
    star_share = _ratio(float(frame.stars.count), reference.star_count)

    # -- Cloud and haze -------------------------------------------------
    # Taken first because it is the most severe and the most certain.
    # measure_stars gives up below sharpness_min_stars and returns
    # StarShape.none(); _ratio then answers 1.0 for "nothing to compare",
    # so the veil test's AND quietly read a frame with no measurable star
    # as perfectly normal. Finding fewer than thirty stars in the centre
    # of a sub that usually shows several hundred is the strongest veil
    # there is. Worth 16 of the 222 ticked frames across nine sessions,
    # for no extra wrong flag at all.
    if not frame.stars.measured:
        alerts.append(Alert(CATEGORY_CLOUDS,
                            f"☁️ HEAVY VEIL (fewer than "
                            f"{settings.sharpness_min_stars} stars measurable "
                            f"in the centre of the frame)"))
    elif haze_sigma > settings.haze_sigma:
        alerts.append(Alert(CATEGORY_CLOUDS,
                            f"☁️ HAZE / CLOUD (deviation: {haze_sigma:.1f} sigma)"))
    elif frame.background > reference.background * settings.haze_background_rise:
        rise = frame.background / reference.background - 1.0
        alerts.append(Alert(CATEGORY_CLOUDS,
                            f"☁️ LIGHT HAZE (background up {rise:.0%})"))
    elif (contrast < settings.haze_contrast_drop
          and star_share < settings.haze_star_drop):
        # A veil the two tests above cannot see: it dims the stars instead
        # of brightening the sky, so the sky itself says nothing. Both
        # halves are required -- see the settings for what either one
        # alone costs.
        alerts.append(Alert(CATEGORY_CLOUDS,
                            f"☁️ VEIL ({star_share:.0%} of the stars a good "
                            f"frame of this session shows, contrast "
                            f"{contrast:.0%})"))

    # -- Trailing -------------------------------------------------------
    # Three signatures: the stars are drawn out into ovals, they have
    # swollen without leaning anywhere, or a jolt printed every one of
    # them twice. A field that moved but kept its stars sharp is not a
    # fault and is not reported -- the shift is in the CSV for whoever
    # wants it.
    smeared = (frame.stars.measured
               and elongation_ratio > settings.trailing_elongation_ratio)
    bloated = (frame.stars.measured
               and radius_ratio > settings.trailing_radius_ratio)
    if smeared or bloated or frame.doubling is not None:
        details = []
        if smeared:
            details.append(f"stars {elongation_ratio:.2f}x the session's "
                           f"sharpest, {frame.stars.elongation:.2f} long over wide")
            if radius_ratio > 1.10:
                details.append(f"{radius_ratio:.2f}x as wide")
        elif bloated:
            details.append(f"stars {radius_ratio:.2f}x as wide as the "
                           f"session's sharpest, but not leaning anywhere "
                           f"— focus or seeing rather than the mount")
        if frame.doubling is not None:
            doubling = frame.doubling
            details.append(
                f"doubled by {doubling['distance']:.0f}px "
                f"(dx={doubling['dx']:.0f}, dy={doubling['dy']:.0f}, "
                f"{doubling['fraction']:.0%} of stars)")
        alerts.append(Alert(CATEGORY_TRAILING, "🫨 TRAILING (" + ", ".join(details) + ")"))

    # -- Satellites, meteors, aircraft ----------------------------------
    track = frame.track
    # When the stars themselves are drawn into parallel lines, a sparse
    # dotted line of them is indistinguishable from a dotted trail. On
    # those frames a track has to be continuous, or made of enough pieces
    # that the alignment cannot be luck -- see track_fragments_if_soft.
    soft_stars = smeared or bloated
    if track is not None and (
            not soft_stars
            or track.coverage >= settings.track_min_coverage
            or track.fragments >= settings.track_fragments_if_soft):
        shape = track.profile or TrackProfile()
        edge_note = " · crosses the whole field" if track.touches_edge else ""
        pieces = (f", {track.fragments} fragments" if track.fragments > 1 else "")
        geometry_note = (f"span {track.span:.0f}px at {track.tilt:.0f}°, "
                         f"width {shape.width:.2f}px, {track.coverage:.0%} lit"
                         f"{pieces}")

        # Width first: it is the only one of these tests with a measured
        # ceiling under it. Then the light along the trail decides which
        # kind of wide thing it is -- evenly lit for the whole exposure,
        # or a bright head with a fading tail.
        is_wide = shape.width > thresholds.wide_track_width
        is_tapered = shape.taper >= settings.meteor_taper
        is_bright = frame.noise > reference.noise * settings.aircraft_brightness

        # Only a one-off track can be a meteor or an aircraft; see
        # track_recur_offset.
        is_wide = is_wide and not frame.track_recurs
        is_green = (shape.green >= settings.meteor_green_sigma
                    and not frame.track_recurs)

        if is_green:
            alerts.append(Alert(CATEGORY_METEORS,
                                f"☄️ METEOR (green glow beside the trail, "
                                f"{shape.green:.1f} sigma: {geometry_note})"
                                f"{edge_note}"))
        elif is_wide and is_tapered:
            alerts.append(Alert(CATEGORY_METEORS,
                                f"☄️ METEOR (wide and uneven: {geometry_note}, "
                                f"halves {shape.taper:.1f}:1, brightest at "
                                f"{shape.head_position:.0%} along){edge_note}"))
        elif is_wide and is_bright:
            alerts.append(Alert(CATEGORY_AIRCRAFT,
                                f"✈️ AIRCRAFT (wide and evenly lit: {geometry_note}, "
                                f"frame noise {frame.noise / reference.noise:.1f}x"
                                f"){edge_note}"))
        else:
            recur_note = (" · also on a neighbouring frame"
                          if frame.track_recurs else "")
            alerts.append(Alert(CATEGORY_SATELLITES,
                                f"🛰️ SATELLITE ({geometry_note}){edge_note}"
                                f"{recur_note}"))

    shift_dx, shift_dy, shift_confidence = frame.field_shift or (0.0, 0.0, 0.0)
    doubling = frame.doubling or {}
    profile = (frame.track.profile if frame.track and frame.track.profile
               else TrackProfile())
    metrics = {
        "frame": frame.name,
        "categories": " ".join(alert.category for alert in alerts),
        "background": round(frame.background, 1),
        "haze_sigma": round(haze_sigma, 2),
        "peak": round(frame.peak, 1),
        "noise": round(frame.noise, 2),
        "contrast": round(contrast, 3),
        "star_share": round(star_share, 3),
        "stars": frame.stars.count,
        "elongation": round(frame.stars.elongation, 3),
        "elongation_ratio": round(elongation_ratio, 3),
        "star_radius": round(frame.stars.radius, 3),
        "radius_ratio": round(radius_ratio, 3),
        "coherence": round(frame.stars.coherence, 2),
        "track_span": round(frame.track.span, 1) if frame.track else 0.0,
        "track_coverage": round(frame.track.coverage, 2) if frame.track else 0.0,
        "track_residual": round(frame.track.residual, 2) if frame.track else 0.0,
        "track_width": round(frame.track.max_width, 2) if frame.track else 0.0,
        "track_fragments": frame.track.fragments if frame.track else 0,
        "track_tilt": round(frame.track.tilt, 1) if frame.track else 0.0,
        "track_recurs": int(frame.track_recurs),
        "track_glow": round(profile.glow, 1),
        "track_peak": round(profile.peak, 1),
        "track_rms_width": round(profile.width, 2),
        "head_ratio": round(profile.head_ratio, 2),
        "head_position": round(profile.head_position, 2),
        "taper": round(profile.taper, 2),
        "track_green": round(profile.green, 1),
        "shift_px": round(float(np.hypot(shift_dx, shift_dy)), 1),
        "shift_dx": round(shift_dx, 1),
        "shift_dy": round(shift_dy, 1),
        "shift_confidence": round(shift_confidence, 2),
        "doubling_px": round(doubling.get("distance", 0.0), 1),
        "doubling_fraction": round(doubling.get("fraction", 0.0), 3),
    }
    return alerts, metrics


def same_line(a: Track, b: Track, settings: Settings,
              thresholds: Thresholds) -> bool:
    """Whether two tracks lie on one line: same angle, no offset across it.

    Not whether they overlap: a satellite moves along its own line, so on
    the next frame it is further down the same line, not on top of where
    it was.
    """
    turn = abs(a.tilt - b.tilt) % 180.0
    if min(turn, 180.0 - turn) > settings.track_recur_tilt:
        return False
    dx, dy = b.centre_x - a.centre_x, b.centre_y - a.centre_y
    across = max(abs(dx * a.unit_y - dy * a.unit_x),
                 abs(dx * b.unit_y - dy * b.unit_x))
    return across <= thresholds.track_recur_offset


def mark_recurring_tracks(frames: Sequence[FrameStats], settings: Settings,
                          thresholds: Thresholds) -> None:
    """Flag every track found again on the frame before or after it.

    The frames are in shooting order. Decision, not measurement: it reads
    the tracks already found, so a replay runs it too, after refitting.
    """
    for index, frame in enumerate(frames):
        frame.track_recurs = frame.track is not None and any(
            other.track is not None
            and same_line(frame.track, other.track, settings, thresholds)
            for other in frames[max(0, index - 1):index + 2]
            if other is not frame)


def _measure_shifts(frames: Sequence[FrameStats], thresholds: Thresholds) -> None:
    """Fill in each frame's shift relative to the previous one.

    Sequential by nature -- every measurement needs the frame before it --
    but it only ever handles short lists of star positions, never images,
    so the cost is negligible.
    """
    previous: np.ndarray | None = None
    for frame in frames:
        frame.field_shift = measure_field_shift(previous, frame.star_positions,
                                                thresholds)
        if frame.star_positions is not None and len(frame.star_positions) >= 8:
            previous = frame.star_positions


def classify_session(frames_folder: str | Session, mode: str | None = "copy",
                     settings: Settings = DEFAULT_SETTINGS,
                     progress: Callable[[int, int], None] | None = None
                     ) -> dict[str, list[str]]:
    """Analyse a session's light frames and file the flagged previews.

    ``frames_folder`` is a session folder, resolved with the configured
    layout, or a :class:`Session` already resolved. ``mode`` is "copy",
    "move", or None to skip the filing step entirely and only print the
    report. ``progress`` is called with (frames done, total) as the
    analysis goes, for a window to show.

    Returns the categories raised per frame, which is also what gets
    handed to :func:`sort_previews`.
    """
    session = (frames_folder if isinstance(frames_folder, Session)
               else resolve(session_root(frames_folder), configured_layout()))
    frames_folder = session.root
    # Sorted by name, which is chronological with Seestar's timestamps --
    # the frame-to-frame shift depends on it.
    paths = session.frames
    if not paths:
        print(f"⚠️ No FITS files in '{frames_folder}'.")
        return {}

    geometry = Geometry.of(paths[0], settings)
    refusal = unsupported_scale(geometry, settings)
    if refusal is not None:
        print(f"⚠️ '{frames_folder}' is outside what this tool can judge.")
        print(f"   {refusal}")
        return {}
    thresholds = Thresholds.derive(settings, geometry)
    if geometry.scale_is_assumed:
        print(f"⚠️ No FOCALLEN/XPIXSZ in the header — assuming "
              f"{geometry.pixel_scale:.2f} arcsec/px. Every threshold here is "
              f"angular, so if that guess is wrong the verdicts are too. Set "
              f"FALLBACK_PIXEL_SCALE to your own plate scale.")
    fov_x, fov_y = geometry.field_of_view
    print(f"Analysing {len(paths)} frames — {geometry.model}, "
          f"{geometry.width}x{geometry.height} at {geometry.pixel_scale:.2f}\"/px "
          f"({fov_x:.2f}° x {fov_y:.2f}°), working at 1/{geometry.step}...")
    started = time.time()

    context = WorkerContext(settings=settings, geometry=geometry,
                            thresholds=thresholds)
    with ProcessPoolExecutor(max_workers=settings.max_workers,
                             initializer=_init_worker,
                             initargs=(context,)) as pool:
        # Without a chunk size each frame is a separate task and the
        # scheduling costs more than the work on short runs.
        chunk = max(1, len(paths) // (4 * (os.cpu_count() or 4)))
        frames = []
        for done, frame in enumerate(pool.map(analyse_frame, paths,
                                              chunksize=chunk), start=1):
            if frame is not None:
                frames.append(frame)
            if progress is not None:
                progress(done, len(paths))

    if not frames:
        print("⚠️ No frame could be read.")
        return {}

    elapsed = time.time() - started
    _measure_shifts(frames, thresholds)
    mark_recurring_tracks(frames, settings, thresholds)
    reference = SessionReference.of(frames)

    print("\n" + "=" * 60)
    print("📋 SESSION REPORT")
    print("=" * 60 + "\n")

    categories_by_frame: dict[str, list[str]] = {}
    counts = {CATEGORY_CLOUDS: 0, CATEGORY_TRAILING: 0,
              CATEGORY_AIRCRAFT: 0, CATEGORY_METEORS: 0,
              CATEGORY_SATELLITES: 0}
    rows: list[dict] = []
    flagged: list[tuple[FrameStats, list[str], list[str]]] = []

    for frame in frames:
        alerts, metrics = classify_frame(frame, reference, settings,
                                         thresholds)
        rows.append(metrics)
        if not alerts:
            continue
        categories = [alert.category for alert in alerts]
        messages = [alert.message for alert in alerts]
        for category in categories:
            counts[category] += 1
        categories_by_frame[frame.path] = categories
        flagged.append((frame, categories, messages))
        print(f"📷 {frame.name}")
        for message in messages:
            print(f"   -> {message}")
        print("-" * 40)

    # Clear the last run out of the way first, so REVIEW/ always shows
    # the run you just did rather than the union of every run so far.
    # Only when previews are actually being filed: "report" mode is
    # documented as touching nothing, and throwing away the previous
    # run's picks without replacing them would be a nasty surprise.
    cleared = (clean_review_folder(frames_folder, settings, session.previews)
               if mode else 0)
    report_folder = write_reports(frames_folder, rows, counts, elapsed, settings)
    cases_path = (write_cases(frames_folder, frames, rows, reference, geometry,
                              settings, expected=len(paths))
                  if settings.export_cases else None)
    filed = (sort_previews(frames_folder, flagged, mode, geometry, settings,
                           session.previews)
             if mode else 0)

    print("\n" + "=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"📁 Frames            : {len(paths)}")
    if len(frames) < len(paths):
        # A partial run used to read as a complete one: the summary counted
        # the files on disk and called everything not flagged clean, so
        # frames lost to a memory spike or a bad file vanished without
        # trace. Whatever else this report says, it says this first.
        print(f"🚨 ANALYSED ONLY     : {len(frames)} — "
              f"{len(paths) - len(frames)} frame(s) could not be read, and "
              f"nothing below accounts for them")
    print(f"✅ Clean             : {len(frames) - len(flagged)}")
    print(f"⚠️ Flagged           : {len(flagged)}")
    print(f"   ☁️ haze / cloud   : {counts[CATEGORY_CLOUDS]}")
    print(f"   🫨 trailing       : {counts[CATEGORY_TRAILING]}")
    print(f"   ✈️ aircraft       : {counts[CATEGORY_AIRCRAFT]}")
    print(f"   ☄️ meteors        : {counts[CATEGORY_METEORS]}")
    print(f"   🛰️ satellites     : {counts[CATEGORY_SATELLITES]}")
    print(f"⏱️ {elapsed:.2f}s | ⚡ {len(paths) / elapsed:.1f} frames/s")
    if report_folder:
        print(f"📄 Reports written to '{report_folder}{os.sep}' "
              f"({settings.report_csv}, {settings.report_markdown})")
    if cases_path:
        size = os.path.getsize(cases_path) / 1e6
        print(f"🧳 Case bundle written to '{cases_path}' ({size:.1f} MB) — "
              f"every measurement, none of the pixels")
    if cleared:
        print(f"🧽 {cleared} file(s) from the previous run cleared out of "
              f"'{settings.review_folder}{os.sep}'")
    if mode and filed:
        verb = "moved" if mode == "move" else "copied"
        marks = "annotated and " if settings.debug else ""
        used = sorted({c for cs in categories_by_frame.values() for c in cs})
        print(f"🗂️  {filed} preview(s) {marks}{verb} into "
              f"'{os.path.join(frames_folder, settings.review_folder)}"
              f"{os.sep}<Category>{os.sep}' ({', '.join(used)})")
    print("=" * 60)

    return categories_by_frame


# ======================================================================
#  Debug overlay
# ======================================================================

# The colours the overlay draws in, one per thing the detector looked at.
DEBUG_COLOURS = {
    "track": (255, 60, 60),
    "fragment": (255, 170, 0),
    "star": (0, 220, 255),
    "crop": (60, 120, 255),
    "doubling": (255, 0, 220),
    "caption": (255, 255, 255),
}


def _debug_font(size: int):
    """A readable font at 4000 pixels wide, whatever the machine has."""
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_debug_overlay(frame: FrameStats, messages: Sequence[str],
                       preview_path: str, destinations: Sequence[str],
                       geometry: Geometry, settings: Settings) -> None:
    """Redraw one preview with everything the detector reacted to on it.

    A verdict on its own is hard to argue with. Seeing the exact
    fragments that were strung into a track, or the direction every star
    was measured to lean in, turns "why was this flagged?" into a glance.

    Reads the JPG preview, whose pixels line up one to one with the FITS,
    and writes the annotated version to each destination. The preview it
    read from is left as it is; only the copies filed under REVIEW carry
    the marks.
    """
    from PIL import Image, ImageDraw

    with Image.open(preview_path) as preview:
        canvas = preview.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    # Seestar previews match their FITS pixel for pixel, but a preview
    # that has been resized somewhere along the way would otherwise put
    # every mark in the wrong place.
    zoom = canvas.width / geometry.width if geometry.width else 1.0
    step = geometry.step * zoom
    thin = max(canvas.width // 900, 2)
    font = _debug_font(max(canvas.width // 70, 14))

    # -- the region the star shapes were measured in --------------------
    left, top, right, bottom = centre_crop_bounds((canvas.height, canvas.width),
                                                  settings)
    if frame.stars.measured:
        draw.rectangle((left, top, right, bottom),
                       outline=DEBUG_COLOURS["crop"], width=thin)

    # -- every star that voted on the shape, drawn as it was measured ---
    # The tick runs along the star's own major axis and grows with its
    # elongation, so a smeared frame shows a comb all leaning one way and
    # a clean one shows a field of dots.
    for blob in frame.stars.blobs:
        reach = 6.0 * thin * max(blob["elongation"] - 1.0, 0.15)
        x = left + blob["cx"] * zoom
        y = top + blob["cy"] * zoom
        dx = reach * math.cos(blob["angle"])
        dy = reach * math.sin(blob["angle"])
        draw.line((x - dx, y - dy, x + dx, y + dy),
                  fill=DEBUG_COLOURS["star"], width=max(thin // 2, 1))

    # -- the doubling vector, on the stars it was found from ------------
    if frame.doubling is not None:
        dx, dy = frame.doubling["dx"], frame.doubling["dy"]
        for blob in frame.stars.blobs[:60]:
            x, y = left + blob["cx"] * zoom, top + blob["cy"] * zoom
            draw.line((x, y, x + dx * zoom, y + dy * zoom),
                      fill=DEBUG_COLOURS["doubling"], width=thin)

    # -- the track and the fragments it was assembled from --------------
    track = frame.track
    if track is not None:
        for segment in frame.segments:
            if segment.label not in track.members:
                continue
            half_x = segment.axis_x * segment.length / 2.0
            half_y = segment.axis_y * segment.length / 2.0
            draw.line(((segment.cx - half_x) * step, (segment.cy - half_y) * step,
                       (segment.cx + half_x) * step, (segment.cy + half_y) * step),
                      fill=DEBUG_COLOURS["fragment"], width=thin * 3)

        reach = track.span / 2.0
        start = ((track.centre_x - reach * track.unit_x) * step,
                 (track.centre_y - reach * track.unit_y) * step)
        end = ((track.centre_x + reach * track.unit_x) * step,
               (track.centre_y + reach * track.unit_y) * step)
        draw.line((*start, *end), fill=DEBUG_COLOURS["track"], width=thin)
        for corner in (start, end):
            radius = thin * 12
            draw.ellipse((corner[0] - radius, corner[1] - radius,
                          corner[0] + radius, corner[1] + radius),
                         outline=DEBUG_COLOURS["track"], width=thin * 2)

    # -- what the verdict was, in words ---------------------------------
    caption = [frame.name] + list(messages)
    box = draw.multiline_textbbox((0, 0), "\n".join(caption), font=font, spacing=6)
    margin = thin * 6
    draw.rectangle((0, 0, box[2] + 2 * margin, box[3] + 2 * margin), fill=(0, 0, 0))
    draw.multiline_text((margin, margin), "\n".join(caption),
                        fill=DEBUG_COLOURS["caption"], font=font, spacing=6)

    for destination in destinations:
        canvas.save(destination, quality=88)


# ======================================================================
#  Output
# ======================================================================

@dataclass(frozen=True)
class FilingJob:
    """One preview, and everywhere it has to end up."""

    preview: str                  # the JPG in JPG/
    destinations: tuple[str, ...]  # one per category it was flagged under
    move: bool                    # remove the source once it is filed
    frame: FrameStats | None = None       # set only when annotating
    messages: tuple[str, ...] = ()


def _file_one(job: tuple) -> str | None:
    """Put one preview where it belongs. Runs in a worker; never raises."""
    filing, geometry, settings = job
    try:
        if filing.frame is None:
            source = filing.preview
            for index, destination in enumerate(filing.destinations):
                if filing.move and index == 0:
                    shutil.move(source, destination)
                    source = destination  # the file lives here now
                else:
                    shutil.copy2(source, destination)
        else:
            draw_debug_overlay(filing.frame, filing.messages, filing.preview,
                               filing.destinations, geometry, settings)
            if filing.move:
                os.remove(filing.preview)
        return None
    except Exception as error:  # one bad preview must not stop the rest
        return f"{os.path.basename(filing.preview)}: {error}"


def sort_previews(frames_folder: str,
                  flagged: Sequence[tuple[FrameStats, list[str], list[str]]],
                  mode: str, geometry: Geometry, settings: Settings,
                  previews: dict[str, str] | None = None) -> int:
    """File the JPG preview of every flagged frame under REVIEW/<category>/.

    The FITS files are never touched, whatever the mode -- a wrong
    threshold costs you a few copied thumbnails and nothing else. A frame
    that raised several alerts is filed under each of them, which makes
    it easy to review one failure mode at a time.

    In "move" mode the preview leaves ``JPG/`` for good; in "copy" mode it
    stays where it is. With ``debug`` set, what lands in REVIEW is the
    annotated version rather than a plain copy, so the folder you review
    already shows why each frame is in it.

    ``flagged`` carries, per frame, its categories and the messages to
    print on the overlay. ``previews`` maps a frame's name to its JPG,
    as the session layout found them; left out, the session folder is
    resolved again.
    """
    if not flagged:
        return 0

    if previews is None:
        previews = resolve(frames_folder, configured_layout()).previews
    review_root = os.path.join(frames_folder, settings.review_folder)

    if not previews:
        print("ℹ️ No JPG previews found — reports only, nothing to file.")
        return 0

    if settings.debug and mode == "move":
        print("ℹ️ --debug with --mode move: the preview leaves its folder and "
              "only the annotated copy survives.")

    jobs = []
    missing = 0
    for frame, categories, messages in flagged:
        stem = frame_stem(frame.name)
        preview = previews.get(stem)
        if preview is None:
            missing += 1
            continue

        destinations = []
        for category in categories:
            folder = os.path.join(review_root, category)
            os.makedirs(folder, exist_ok=True)
            destinations.append(os.path.join(folder, stem + ".jpg"))

        jobs.append((FilingJob(preview=preview,
                               destinations=tuple(destinations),
                               move=(mode == "move"),
                               frame=frame if settings.debug else None,
                               messages=tuple(messages)),
                     geometry, settings))

    if missing:
        print(f"ℹ️ {missing} flagged frame(s) had no JPG preview; "
              f"they are still listed in the reports.")
    if not jobs:
        return 0

    # Copying is trivial, but drawing on a 2160x3840 preview is not, and
    # both go through the same pool so the debug run stays bearable.
    if settings.debug:
        with ProcessPoolExecutor() as pool:
            problems = list(pool.map(_file_one, jobs))
    else:
        problems = list(map(_file_one, jobs))

    for problem in (p for p in problems if p):
        print(f"   ⚠️ Could not file {problem}")
    return sum(len(job[0].destinations)
               for job, problem in zip(jobs, problems) if problem is None)


def clean_review_folder(frames_folder: str, settings: Settings,
                        previews: dict[str, str] | None = None) -> int:
    """Clear out what the last run wrote, and nothing else.

    Without this, a threshold that used to flag a frame leaves its
    preview sitting in REVIEW for ever and the folder stops reflecting
    the run you just did.

    Deleting inside a folder the owner also uses calls for a rule that
    cannot go wrong, so there is one: **a file is only removed when the
    preview it copies still exists** -- among the previews the session
    layout finds, which is ``JPG/`` on the usual layout. Everything this
    script writes is a copy of a preview, so everything it wrote is
    covered; anything else -- notes, keepers, a preview a previous
    ``--mode move`` run took out of its folder and which now exists
    nowhere else -- is left alone.
    """
    review_root = os.path.join(frames_folder, settings.review_folder)
    if not os.path.isdir(review_root):
        return 0

    if previews is None:
        previews = resolve(frames_folder, configured_layout()).previews
    reproducible = {os.path.basename(path) for path in previews.values()}
    reproducible |= {stem + ".jpg" for stem in previews}

    removed = 0
    for name in (settings.report_csv, settings.report_markdown):
        report = os.path.join(review_root, name)
        if os.path.isfile(report):
            os.remove(report)
            removed += 1

    for folder_name in GENERATED_FOLDERS:
        folder = os.path.join(review_root, folder_name)
        if not os.path.isdir(folder):
            continue
        for entry in os.listdir(folder):
            if entry in reproducible and os.path.isfile(os.path.join(folder, entry)):
                os.remove(os.path.join(folder, entry))
                removed += 1
        if not os.listdir(folder):
            os.rmdir(folder)
    return removed


def _frame_notes(row: dict) -> str:
    """The measurements behind one frame's verdict, for the summary.

    Only what actually contributed: an elongation of 1.02x or a sub-pixel
    shift is noise, and printing it buries the frames that matter.
    """
    notes = []
    if row["elongation_ratio"] >= 1.10:
        notes.append(f"stars {row['elongation_ratio']:.2f}x")
    if row["doubling_px"]:
        notes.append(f"doubling {row['doubling_px']:.0f}px")
    if row["track_span"]:
        notes.append(f"track {row['track_span']:.0f}px")
    if row["shift_px"] >= 2.0:
        notes.append(f"shift {row['shift_px']:.0f}px")
    return f" — {', '.join(notes)}" if notes else ""


def write_reports(frames_folder: str, rows: list[dict],
                  counts: dict[str, int], elapsed: float,
                  settings: Settings) -> str | None:
    """Write the per-frame CSV and the human-readable summary.

    The CSV carries every frame, flagged or not, with the numbers each
    verdict was based on. That is what lets you re-try a threshold in a
    spreadsheet -- or spot that a cutoff sits in the middle of a cluster
    rather than in a gap -- without re-reading the whole session.

    Returns the folder the reports were written to, or None if writing
    was disabled or failed.
    """
    if not settings.write_reports or not rows:
        return None

    destination = os.path.normpath(os.path.join(frames_folder,
                                                settings.review_folder))
    try:
        os.makedirs(destination, exist_ok=True)

        csv_path = os.path.join(destination, settings.report_csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        flagged = [row for row in rows if row["categories"]]
        lines = [
            f"# Session report — {os.path.basename(os.path.abspath(frames_folder))}",
            "",
            f"{len(rows)} frames · {len(flagged)} flagged · "
            f"{len(rows) - len(flagged)} clean · {elapsed:.1f}s",
            "",
            "| Category | Frames |",
            "| --- | ---: |",
        ]
        lines += [f"| {name} | {count} |" for name, count in sorted(counts.items()) if count]
        lines += ["", "Per-frame measurements: "
                      f"[`{settings.report_csv}`](./{settings.report_csv})", ""]

        for category in sorted(counts):
            members = [r for r in flagged if category in r["categories"].split()]
            if not members:
                continue
            lines += [f"## {category} ({len(members)})", ""]
            for row in members:
                lines.append(f"- `{frame_label(row['frame'])}`"
                             f"{_frame_notes(row)}")
            lines.append("")

        markdown_path = os.path.join(destination, settings.report_markdown)
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return destination
    except OSError as error:
        print(f"⚠️ Could not write the reports: {error}")
        return None


# ======================================================================
#  Entry point
# ======================================================================

# ======================================================================
#  Case export
# ======================================================================

# One session's measurements, without its pixels: a couple of megabytes
# against the four gigabytes of FITS behind them. ``--export-cases``
# writes a bundle, ``replay_cases.py`` reads it back and re-runs the
# decisions on it, so any threshold below the measurement layer can be
# re-scored later on every session ever exported -- long after the frames
# have been stacked away, and across targets, because everything a
# verdict rests on is either a ratio to the frame's own session or an
# absolute in arcseconds.
#
# Two things make that work, and both are easy to get wrong.
#
# The session's yardstick travels *with* the frames, computed over the
# whole night at export time. So a bundle holding only the interesting
# frames still classifies them exactly as the full run did: you do not
# have to keep the clean majority in order to keep the cases.
#
# And a bundle cannot change how its numbers were obtained. The
# fingerprint below records the code that produced them, and the replay
# stops when it no longer matches. That guard is not theoretical: an
# earlier tool here kept its own copy of the star selection, drifted out
# of step with three fixes, and quietly validated a version that no
# longer existed. When the fingerprint moves, re-export from the FITS --
# they stay on the owner's disk, only the bundles travel.

# 1: the aggregates a verdict rests on. 2: adds spatial_summary, and one
# fingerprint per measurement group instead of one for all of them.
CASE_FORMAT_VERSION = 2
CASE_FORMATS_READ = (1, 2)

# The functions that turn pixels into numbers, grouped by what they
# produce. A bundle records one hash per group and a replay checks only
# the groups it actually reads, so a change to the track profile does not
# invalidate the star data and vice versa.
#
# ``analyse_frame`` is deliberately absent. It only assembles: it decides
# which of these to call and what to write down, and a bundle that gains
# a field or widens an export floor is not thereby describing different
# numbers. Everything that touches a pixel lives in one of the functions
# below -- keep it that way, or this guard stops meaning anything. The
# assembly is still hashed into the bundle as "assembly", for reading
# when something looks wrong, and never enforced.
MEASUREMENT_PARTS = {
    "frame": ("load_frame", "working_image", "bright_structure",
              "high_percentile", "_principal_axes", "measure_components"),
    "stars": ("centre_crop_bounds", "crop_centre", "measure_stars"),
    "profile": ("measure_track_profile",),
    # A group of its own, added after bundles already existed. A bundle
    # written before it has no hash for it: its tracks were simply never
    # measured for colour and carry green = 0, which replays exactly as
    # they were judged then. That is a measurement missing, not one gone
    # stale, so it does not condemn the rest of the bundle.
    "colour": ("measure_track_green",),
}

# ...and the settings those functions read. A bundle records every
# setting, but only a change to one of these invalidates it. The rest are
# decisions, which is the whole point of replaying.
MEASUREMENT_SETTINGS = ("working_scale", "opening_size", "bright_percentile",
                        "min_component_pixels", "sharpness_centre_fraction",
                        "sharpness_opening_size", "sharpness_percentile",
                        "sharpness_min_stars", "star_max_length",
                        "shift_star_count")

# Seestar stamps every frame with the moment it was shot. That is what
# the owner calls a frame by, and it says nothing about what the scope
# was pointed at.
FRAME_TIMESTAMP = re.compile(r"(\d{8}-\d{6})")


def _source_hash(names: Sequence[str]) -> str:
    module = sys.modules[__name__]
    source = "".join(inspect.getsource(getattr(module, name)) for name in names)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def measurement_fingerprints() -> dict[str, str]:
    """One short hash per measurement group, plus the assembly around them.

    Comments count, since getsource returns them, so a cosmetic edit to a
    primitive still invalidates the bundles that depended on it. That is
    the safe direction to err in -- a stale bundle scored as fresh is the
    failure that matters -- and replay_cases.py takes --stale-ok for the
    other one. What the split buys is that such an edit no longer
    invalidates the groups it did not touch.
    """
    parts = {name: _source_hash(names)
             for name, names in MEASUREMENT_PARTS.items()}
    parts["assembly"] = _source_hash(("analyse_frame",))
    return parts


def measurement_fingerprint() -> str:
    """The measurement groups as one hash, for a format-1 bundle."""
    parts = measurement_fingerprints()
    joined = "".join(parts[name] for name in sorted(MEASUREMENT_PARTS))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def case_frame_id(name: str, index: int) -> str:
    """What a frame is called inside a bundle: when, never what.

    Falls back to the position in the session for anything that is not a
    Seestar name, which keeps ids unique without inventing a timestamp.
    """
    found = FRAME_TIMESTAMP.search(name)
    return found.group(1) if found else f"frame-{index:04d}"


def _segment_row(segment: Segment) -> list:
    """A segment as a bare list -- a session holds tens of thousands."""
    return [segment.label, segment.cx, segment.cy, segment.length,
            segment.width, segment.ratio, segment.axis_x, segment.axis_y,
            int(segment.touches_edge)]


def _segment_of_row(row: Sequence) -> Segment:
    return Segment(label=int(row[0]), cx=row[1], cy=row[2], length=row[3],
                   width=row[4], ratio=row[5], axis_x=row[6], axis_y=row[7],
                   touches_edge=bool(row[8]))


_TRACK_FIELDS = ("span", "coverage", "residual", "fragments", "max_width",
                 "touches_edge", "tilt", "centre_x", "centre_y",
                 "unit_x", "unit_y")


def _track_record(track: Track | None) -> dict | None:
    if track is None:
        return None
    record = {name: getattr(track, name) for name in _TRACK_FIELDS}
    record["members"] = list(track.members)
    record["profile"] = asdict(track.profile) if track.profile else None
    return record


def _track_of_record(record: dict | None) -> Track | None:
    if record is None:
        return None
    profile = record.get("profile")
    return Track(members=tuple(record.get("members", ())),
                 profile=TrackProfile(**profile) if profile else None,
                 **{name: record[name] for name in _TRACK_FIELDS})


def _stars_record(stars: StarShape) -> dict:
    return {"count": stars.count, "radius": stars.radius,
            "elongation": stars.elongation, "coherence": stars.coherence,
            "blobs": [[b["cx"], b["cy"], b["size"], b["elongation"], b["angle"]]
                      for b in stars.blobs]}


def _stars_of_record(record: dict) -> StarShape:
    return StarShape(count=record["count"], radius=record["radius"],
                     elongation=record["elongation"],
                     coherence=record["coherence"],
                     blobs=[{"cx": b[0], "cy": b[1], "size": b[2],
                             "elongation": b[3], "angle": b[4]}
                            for b in record.get("blobs", [])])


def frame_case(frame: FrameStats, index: int, verdict: Sequence[str],
               label: dict | None) -> dict:
    """One frame's row in a bundle: every number, none of the pixels."""
    return {
        "id": case_frame_id(frame.name, index),
        "background": frame.background,
        "peak": frame.peak,
        "noise": frame.noise,
        "stars": _stars_record(frame.stars),
        "track": _track_record(frame.track),
        "doubling": dict(frame.doubling) if frame.doubling else None,
        "segments": [_segment_row(s) for s in frame.export_segments],
        "spatial": frame.spatial,
        "star_positions": [[float(x), float(y)] for x, y in frame.star_positions],
        "field_shift": list(frame.field_shift) if frame.field_shift else None,
        # What the run said at the time, so a replay can show what moved.
        "verdict": list(verdict),
        # What the owner saw, on the frames they have been through.
        "labels": list(label["labels"]) if label else [],
        "note": label["note"] if label else "",
    }


def frame_of_case(case: dict) -> FrameStats:
    """Rebuild the frame a bundle describes, pixels excepted.

    The shift is restored rather than re-measured: it is already the
    answer ``_measure_shifts`` would give, and a bundle holding only some
    of the night's frames has no previous frame to measure against.
    """
    shift = case.get("field_shift")
    return FrameStats(
        name=case["id"],
        path="",
        background=case["background"],
        peak=case["peak"],
        noise=case["noise"],
        stars=_stars_of_record(case["stars"]),
        track=_track_of_record(case.get("track")),
        doubling=case.get("doubling"),
        star_positions=np.array(case.get("star_positions") or [],
                                dtype=float).reshape(-1, 2),
        field_shift=tuple(shift) if shift else None,
        export_segments=[_segment_of_row(row) for row in case.get("segments", [])],
        spatial=case.get("spatial") or {},
    )


def read_frame_labels(path: str) -> dict[str, dict]:
    """Read a labels.csv, keyed the way a bundle names its frames.

    Keyed on the timestamp rather than on the file name, so that a .fit
    row, a .jpg row and a bundle id all land on the same frame -- a
    Seestar name carries its exposure as '60.0s', which defeats splitext.
    Takes the file rather than its folder so that a replay can be handed
    a labels.csv written after the bundle was.
    """
    if not os.path.isfile(path):
        return {}
    found: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = FRAME_TIMESTAMP.search((row.get("frame") or "").strip())
            if not match:
                continue
            found[match.group(1)] = {"labels": (row.get("labels") or "").split(),
                                     "note": (row.get("note") or "").strip()}
    return found


def session_label(frames_folder: str) -> str:
    """The target, taken from the folder, for the bundle's file name.

    This is the one place a bundle says what it looked at, and it says it
    on the outside: the owner has to be able to find the session a bundle
    came from among a dozen of them. Nothing inside reads it, and nothing
    scored depends on it -- which is what keeps cases from different
    targets comparable. Renaming the folder renames the next bundle and
    changes no identity, because that comes from the frames.
    """
    stem = os.path.basename(os.path.abspath(frames_folder))
    if stem.lower().endswith("_sub"):
        stem = stem[:-4]
    cleaned = "".join("_" if c in ' \t<>:"/\\|?*' else c for c in stem)
    return "_".join(filter(None, cleaned.split("_")))


def session_identifier(frames: Sequence[FrameStats]) -> str:
    """A name for a night that says when it was, and nothing else.

    The moment the first frame was taken. It comes from the data rather
    than from the folder, so renaming the folder -- or exporting the same
    night twice -- does not produce a second identity for one session;
    and two sessions cannot start in the same second. Nothing in it says
    what the scope was pointed at, which is what lets bundles from
    different targets be scored as one set.
    """
    return case_frame_id(frames[0].name, 0) if frames else "unknown"


def build_case_bundle(frames: Sequence[FrameStats], rows: Sequence[dict],
                      reference: SessionReference, geometry: Geometry,
                      settings: Settings, labels: dict[str, dict],
                      session_id: str, target: str = "") -> dict:
    """Everything a replay needs, and nothing that names the target."""
    verdicts = [(row.get("categories") or "").split() for row in rows]
    cases = []
    for index, (frame, verdict) in enumerate(zip(frames, verdicts)):
        identifier = case_frame_id(frame.name, index)
        cases.append(frame_case(frame, index, verdict, labels.get(identifier)))
    return {
        "format": CASE_FORMAT_VERSION,
        "fingerprints": measurement_fingerprints(),
        # Kept so that a reader written for format 1 still finds it.
        "measurement_fingerprint": measurement_fingerprint(),
        "session": {
            "id": session_id,
            # A label for the reader, so a bundle can be told from its
            # neighbours. Nothing decides anything on it.
            "target": target,
            "frames": len(cases),
            "geometry": asdict(geometry),
            "settings": asdict(settings),
            "reference": asdict(reference),
            # The floor the segments were kept at. A replay may move the
            # join bar anywhere above this, and nowhere below it.
            "segment_floor": {
                "length": (Thresholds.derive(settings, geometry).segment_min_length
                           * settings.export_segment_length_factor),
                "ratio": min(settings.segment_min_ratio,
                             settings.export_segment_min_ratio),
            },
        },
        "frames": cases,
    }


def cases_folder() -> str:
    """Where the bundles collect: beside the script, not inside a session.

    A bundle is meant to outlive its frames -- that is the whole reason
    it exists -- so it must not sit in the folder that gets emptied,
    archived or deleted once the night has been stacked. Beside the
    script they also accumulate in one place, which is what lets a replay
    score every night ever exported without being told where they are.
    """
    return os.path.dirname(os.path.abspath(__file__))


def write_cases(frames_folder: str, frames: Sequence[FrameStats],
                rows: Sequence[dict], reference: SessionReference,
                geometry: Geometry, settings: Settings,
                expected: int = 0) -> str | None:
    """Write the session's bundle into the folder the bundles collect in.

    Refuses a partial session. The format rests on the yardstick being
    measured over the whole night and travelling with the frames; if some
    frames were lost -- a memory spike, an unreadable file -- the
    reference describes the survivors instead, and every ratio in the
    bundle is quietly wrong. The identity goes too, since it is the first
    frame that was actually read.
    """
    if expected and len(frames) < expected:
        print(f"⚠️ No case bundle written: {expected - len(frames)} of "
              f"{expected} frames could not be read, and a bundle built on "
              f"the rest would carry a session reference measured on the "
              f"wrong frames. Fix the reading first — see --workers.")
        return None
    session_id = session_identifier(frames)
    target = session_label(frames_folder)
    bundle = build_case_bundle(frames, rows, reference, geometry, settings,
                               read_frame_labels(os.path.join(frames_folder,
                                                              "labels.csv")),
                               session_id, target)
    name = "_".join(filter(None, (settings.cases_prefix, target, session_id)))
    path = os.path.join(cases_folder(), f"{name}.json.gz")
    try:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(bundle, handle)
    except OSError as error:
        print(f"⚠️ Could not write the case bundle: {error}")
        return None
    return path


def read_cases(path: str) -> dict:
    """Load a bundle. Plain JSON when it is not gzipped, for eyeballing."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if bundle.get("format") not in CASE_FORMATS_READ:
        raise ValueError(f"{path}: case format {bundle.get('format')}, "
                         f"this build reads {CASE_FORMATS_READ}")
    return bundle


def cases_settings(bundle: dict) -> Settings:
    """The settings the bundle was measured with.

    Tolerates a bundle from an older build: a setting that no longer
    exists is dropped, and one that did not exist yet keeps today's
    default. Tuples come back from JSON as lists and have to be put back,
    or a frozen dataclass quietly holds the wrong type.
    """
    known = {f.name for f in fields(Settings)}
    kept = {name: tuple(value) if isinstance(value, list) else value
            for name, value in bundle["session"]["settings"].items()
            if name in known}
    return Settings(**kept)


def cases_session(bundle: dict) -> tuple[list[FrameStats], SessionReference,
                                         Settings, Geometry]:
    """Unpack a bundle into the objects the classifier already speaks.

    A bundle written before ``star_count`` joined the reference has to
    have it derived here instead, from the frames it carries. That is
    exact for a bundle holding a whole session and wrong for a partial
    one, which is why it is only ever a fallback -- and why a bundle now
    records the number rather than leaving it to be guessed.
    """
    frames = [frame_of_case(case) for case in bundle["frames"]]
    reference = SessionReference(**bundle["session"]["reference"])
    if not reference.star_count and frames:
        reference = replace(reference, star_count=float(
            np.percentile([f.stars.count for f in frames], 75)))
    geometry = Geometry(**bundle["session"]["geometry"])
    return frames, reference, cases_settings(bundle), geometry


def frame_label(filename: str) -> str:
    """The part of a frame's name that tells it from its neighbours.

    Seestar names every sub in a session with the same prefix and a
    timestamp; printing the prefix eighty times buries the one thing that
    identifies the frame.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def select_for_deletion(categories_by_frame: dict[str, list[str]],
                        wanted: set[str] | None) -> dict[str, list[str]]:
    """The flagged frames matching the requested categories, in order.

    ``wanted`` of None means every flagged frame, whatever put it there.
    """
    return {path: categories
            for path, categories in sorted(categories_by_frame.items())
            if wanted is None or wanted & set(categories)}


def confirm_deletion(selection: dict[str, list[str]], session_size: int,
                     assume_yes: bool) -> bool:
    """Show exactly what is about to go, and ask.

    A count on its own is not enough to consent to something
    irreversible: the trailing threshold is a severity dial, not a
    constant, and the honest failure mode of this tool is a dial set one
    notch too tight quietly selecting half the night. So the frames are
    listed, and the share of the session is spelled out when it is large.
    """
    print(f"\n⚠️ {len(selection)} .fit file(s) would be permanently deleted "
          f"— the FITS themselves, not their previews:")
    for path, categories in selection.items():
        print(f"     {frame_label(path):20s} {' '.join(categories)}")

    share = len(selection) / session_size if session_size else 0.0
    if share >= 0.5:
        print(f"   ❗ That is {share:.0%} of the session. If that is not what "
              f"you meant, the threshold is the thing to change, not the "
              f"answer to this question.")

    if assume_yes:
        print("   --yes given, going ahead.")
        return True

    try:
        answer = input("   Type 'delete' to confirm: ").strip().lower()
    except EOFError:          # piped stdin, no one there to answer
        answer = ""
    if answer != "delete":
        print("   Deletion cancelled — nothing was removed.")
        return False
    return True


def delete_flagged_frames(selection: dict[str, list[str]]) -> tuple[int, int]:
    """Delete the .fit files themselves. Returns ``(deleted, failed)``.

    The only place in this script that removes a FITS file rather than
    filing its JPG preview. Everywhere else deliberately never touches
    the data, so that a wrong threshold costs nothing; this runs only
    when asked for by name on the command line, and only after
    :func:`confirm_deletion` has shown what it covers.
    """
    deleted = 0
    failed = 0
    for path in selection:
        try:
            os.remove(path)
            deleted += 1
        except OSError as error:
            failed += 1
            print(f"   ⚠️ Could not delete {os.path.basename(path)}: {error}")
    return deleted, failed


def find_sub_folders(root: str = ".") -> list[str]:
    """Every session folder under ``root``, with the configured layout."""
    return [session.root for session in find_sessions(root, configured_layout())]


def prompt_for_choice(items: Sequence[str], prompt: str = "Your choice") -> str:
    """Show a numbered list and ask the user to pick one entry."""
    for index, item in enumerate(items, start=1):
        print(f"   [{index}] {item}")
    while True:
        answer = input(f"{prompt} (1-{len(items)}): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return items[int(answer) - 1]
        print("   ⚠️ Not a valid choice, try again.")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line. Everything is optional; see the epilog."""
    parser = argparse.ArgumentParser(
        description="Triage Seestar light frames before stacking.",
        epilog="The folder may be a session or a folder holding several; "
               "with several -- or none given, which searches the current "
               "folder -- they are offered as a numbered list. For a window "
               "instead of this, run subframe_inspector_app.py.")
    parser.add_argument("folder", nargs="?",
                        help="a session folder, or a folder holding sessions")
    parser.add_argument("--mode", choices=("copy", "move", "report", "none"),
                        default="copy",
                        help="what to do with the JPG previews of flagged "
                             "frames: copy them into REVIEW/ (default), move "
                             "them, or write the reports only ('report', or "
                             "'none' for the same thing)")
    parser.add_argument("--debug", action="store_true",
                        help="draw on the previews as they are filed into "
                             "REVIEW/, showing the track, its fragments and "
                             "the measured star shapes")
    parser.add_argument("--trailing", type=float, metavar="RATIO",
                        default=DEFAULT_SETTINGS.trailing_elongation_ratio,
                        help="how oval a frame's stars must be, against the "
                             "session's sharpest quarter, to count as trailed "
                             f"(default {DEFAULT_SETTINGS.trailing_elongation_ratio}; "
                             "1.20 is strict, 1.45 lenient)")

    parser.add_argument("--workers", type=int, metavar="N",
                        help="how many worker processes to use (default: as "
                             "many as the machine offers). Lower it if a run "
                             "reports frames it could not read for want of "
                             "memory; it changes no measurement")
    parser.add_argument("--export-cases", action="store_true",
                        help="write cases_<target>_<date-time>.json.gz beside "
                             "this script: every "
                             "measurement this run made, without the pixels, "
                             "and the labels recorded so far, so the "
                             "thresholds can be re-scored later, on every "
                             "session ever exported, with replay_cases.py")

    deletion = parser.add_argument_group(
        "deleting the FITS",
        "The one thing here that cannot be undone. Off unless asked for by "
        "name, and it lists what it covers before doing anything.")
    deletion.add_argument("--delete-flagged", action="store_true",
                          help="delete the .fit files of the flagged frames, "
                               "once their previews have been filed")
    deletion.add_argument("--categories", nargs="+", metavar="CATEGORY",
                          choices=ALL_CATEGORIES,
                          help="restrict --delete-flagged to these categories "
                               f"({', '.join(ALL_CATEGORIES)}); "
                               "default: every flagged frame")
    deletion.add_argument("--yes", "-y", action="store_true",
                          help="skip the confirmation prompt")

    arguments = parser.parse_args(argv)
    if arguments.categories and not arguments.delete_flagged:
        parser.error("--categories only means something with --delete-flagged")
    if arguments.yes and not arguments.delete_flagged:
        parser.error("--yes only means something with --delete-flagged")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    folder = arguments.folder or "."
    if not os.path.isdir(folder):
        print(f"⚠️ '{folder}' is not a folder.")
        return 1

    layout = configured_layout()
    session = resolve(session_root(folder, layout), layout)
    if not session.frames:
        candidates = find_sessions(folder, layout)
        if not candidates:
            print(f"⚠️ No light frames found in or under '{folder}'.")
            print(f"   Looked for {layout.describe()}")
            return 1
        if len(candidates) == 1:
            session = candidates[0]
        else:
            print("🎯 Several sessions found:")
            chosen = prompt_for_choice([s.root for s in candidates],
                                       "Select the session to analyse")
            session = next(s for s in candidates if s.root == chosen)

    print(f"📂 Sorting frames in: {session.root}")
    print(f"   {len(session.frames)} frames ({session.frame_pattern}), "
          f"{len(session.previews)} JPG previews\n")

    # Everything else lives in Settings, and can be overridden the same
    # way, e.g. replace(DEFAULT_SETTINGS, haze_sigma=3.0).
    settings = replace(DEFAULT_SETTINGS,
                       debug=arguments.debug,
                       export_cases=arguments.export_cases,
                       max_workers=arguments.workers,
                       trailing_elongation_ratio=arguments.trailing)

    mode = None if arguments.mode in ("report", "none") else arguments.mode
    categories_by_frame = classify_session(session, mode=mode, settings=settings)

    if not arguments.delete_flagged:
        return 0

    wanted = set(arguments.categories) if arguments.categories else None
    selection = select_for_deletion(categories_by_frame, wanted)
    if not selection:
        scope = f" in {', '.join(sorted(wanted))}" if wanted else ""
        print(f"\nℹ️ No flagged frame{scope} — nothing to delete.")
        return 0

    if not confirm_deletion(selection, len(session.frames), arguments.yes):
        return 0

    deleted, failed = delete_flagged_frames(selection)
    print(f"🗑️  {deleted} .fit file(s) deleted"
          + (f", {failed} failed" if failed else "") + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
