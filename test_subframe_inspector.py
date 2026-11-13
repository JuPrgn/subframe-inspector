# -*- coding: utf-8 -*-
"""Tests for subframe_inspector.

Run them with::

    python -m unittest -v test_subframe_inspector

Nothing here needs a FITS file or a night's data. What it covers is the
machinery that fails *silently* -- an arcsecond converted to the wrong
pixel scale, a track detector that stops telling a line from a scatter,
a housekeeping pass that deletes one file too many. Whether a threshold
is set in the right place is a question for real frames and the CSV, and
no unit test can answer it.

The synthetic images are deliberately crude: a few Gaussian blobs on a
flat background. They prove the measurements mean what their names say,
not that the sky looks like that.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np

import replay_cases as rc
import session_layout as sl
import subframe_inspector as si


S = si.DEFAULT_SETTINGS


def geometry(pixel_scale: float = 3.744, step: int = 2,
             width: int = 2160, height: int = 3840) -> si.Geometry:
    """A geometry with no header behind it, for arithmetic."""
    return si.Geometry(width=width, height=height, pixel_scale=pixel_scale,
                       step=step, model="test")


def draw_blob(image: np.ndarray, x: float, y: float, flux: float = 4000.0,
              sigma_long: float = 1.3, sigma_short: float = 1.3,
              angle: float = 0.0) -> None:
    """Add one elliptical Gaussian to an image, in place."""
    reach = int(max(sigma_long, sigma_short) * 5) + 2
    left, right = max(int(x) - reach, 0), min(int(x) + reach + 1, image.shape[1])
    top, bottom = max(int(y) - reach, 0), min(int(y) + reach + 1, image.shape[0])
    if right <= left or bottom <= top:
        return
    ys, xs = np.mgrid[top:bottom, left:right]
    dx, dy = xs - x, ys - y
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    image[top:bottom, left:right] += flux * np.exp(
        -0.5 * ((along / sigma_long) ** 2 + (across / sigma_short) ** 2))


def star_field(shape: tuple[int, int] = (400, 400), count: int = 200,
               sigma_long: float = 1.3, sigma_short: float = 1.3,
               angle: float = 0.0, seed: int = 7) -> np.ndarray:
    """A flat sky with `count` identical stars scattered over it."""
    rng = np.random.default_rng(seed)
    image = np.full(shape, 100.0, dtype=np.float32)
    image += rng.normal(0.0, 2.0, shape).astype(np.float32)
    margin = 12
    for _ in range(count):
        draw_blob(image, rng.uniform(margin, shape[1] - margin),
                  rng.uniform(margin, shape[0] - margin),
                  flux=rng.uniform(2000, 6000),
                  sigma_long=sigma_long, sigma_short=sigma_short, angle=angle)
    return image


def segment(cx: float, cy: float, length: float, width: float,
            angle_deg: float, label: int = 0) -> si.Segment:
    """One fragment, described directly rather than measured."""
    angle = np.radians(angle_deg)
    return si.Segment(label=label, cx=cx, cy=cy, length=length, width=width,
                      ratio=length / width, axis_x=float(np.cos(angle)),
                      axis_y=float(np.sin(angle)), touches_edge=False)


# ======================================================================
#  Scale: the conversion everything else rests on
# ======================================================================

class TestGeometry(unittest.TestCase):

    def test_plate_scale_from_header(self):
        """206265 * pitch / focal length, and the S30 Pro's own numbers."""
        header = {"FOCALLEN": 160.0, "XPIXSZ": 2.9, "XBINNING": 1}
        geo = si.Geometry.from_header(header, (3840, 2160), S)
        self.assertAlmostEqual(geo.pixel_scale, 3.7385, places=3)
        self.assertEqual((geo.width, geo.height), (2160, 3840))
        self.assertFalse(geo.scale_is_assumed)

    def test_binning_widens_the_pixels(self):
        header = {"FOCALLEN": 160.0, "XPIXSZ": 2.9, "XBINNING": 2}
        binned = si.Geometry.from_header(header, (1920, 1080), S)
        header["XBINNING"] = 1
        native = si.Geometry.from_header(header, (3840, 2160), S)
        self.assertAlmostEqual(binned.pixel_scale, 2 * native.pixel_scale)

    def test_missing_header_is_flagged_not_hidden(self):
        """A guessed scale must announce itself; silence would be a lie."""
        geo = si.Geometry.from_header({}, (3840, 2160), S)
        self.assertTrue(geo.scale_is_assumed)
        self.assertEqual(geo.pixel_scale, si.FALLBACK_PIXEL_SCALE)

    def test_decimation_tracks_the_plate_scale(self):
        """A finer camera is decimated harder, to a constant working scale."""
        coarse = si.Geometry.from_header(
            {"FOCALLEN": 160.0, "XPIXSZ": 2.9}, (3840, 2160), S)
        fine = si.Geometry.from_header(
            {"FOCALLEN": 250.0, "XPIXSZ": 2.0}, (3840, 2160), S)
        self.assertLess(fine.pixel_scale, coarse.pixel_scale)
        self.assertGreaterEqual(fine.step, coarse.step)
        for geo in (coarse, fine):
            working = geo.pixel_scale * geo.step
            self.assertLess(abs(working - S.working_scale), S.working_scale / 2)

    def test_decimation_is_clamped(self):
        """Never below 1, never above 4, whatever the header claims."""
        huge = si.Geometry.from_header(
            {"FOCALLEN": 4000.0, "XPIXSZ": 1.0}, (1000, 1000), S)
        tiny = si.Geometry.from_header(
            {"FOCALLEN": 50.0, "XPIXSZ": 9.0}, (1000, 1000), S)
        self.assertEqual(huge.step, 4)
        self.assertEqual(tiny.step, 1)

    def test_arcseconds_reach_both_pixel_spaces(self):
        geo = geometry(pixel_scale=3.744, step=2)
        self.assertAlmostEqual(geo.px(37.44), 10.0)
        self.assertAlmostEqual(geo.working_px(37.44), 5.0)
        self.assertEqual(geo.odd_px(34.0) % 2, 1)


class TestSupportedScale(unittest.TestCase):
    """Which sessions the tool declines, and why it must decline them.

    Decimation only coarsens a scale, so a lens already coarser than
    ``working_scale`` cannot be brought back to it and every pixel-valued
    threshold comes to mean something else. Getting this wrong in the lax
    direction is not a near miss: on the Seestar wide-angle lens the tool
    reported 59 frames of 59 as satellites.
    """

    def test_the_main_lens_is_supported(self):
        for scale in (3.74, 3.99):
            geo = geometry(pixel_scale=scale, step=2)
            self.assertIsNone(si.unsupported_scale(geo, S), f"{scale}")

    def test_a_finer_scope_hitting_the_step_cap_is_supported(self):
        """A sharper instrument decimates to 6.4 arcsec/px once step caps
        at 4, which is inside the invariant and must stay welcome."""
        geo = geometry(pixel_scale=1.6, step=4)
        self.assertIsNone(si.unsupported_scale(geo, S))

    def test_the_wide_angle_lens_is_declined(self):
        """6 mm of focal length and 1.6 micron pixels: 55 arcsec/px, which
        cannot decimate at all."""
        geo = geometry(pixel_scale=55.0, step=1)
        reason = si.unsupported_scale(geo, S)
        self.assertIsNotNone(reason)
        self.assertIn("55", reason)

    def test_the_refusal_says_how_far_off_the_thresholds_would_be(self):
        reason = si.unsupported_scale(geometry(pixel_scale=55.0, step=1), S)
        self.assertIn("7.3x", reason)

    def test_the_working_scale_is_the_plate_scale_times_the_step(self):
        self.assertAlmostEqual(geometry(pixel_scale=3.74, step=2).working_scale,
                               7.48)


class TestThresholds(unittest.TestCase):

    def test_arcseconds_land_in_the_right_space(self):
        """Working-image fields decimated, full-resolution ones not."""
        geo = geometry(pixel_scale=3.744, step=2)
        th = si.Thresholds.derive(S, geo)
        self.assertAlmostEqual(th.segment_min_length,
                               S.segment_min_length / (3.744 * 2), places=6)
        self.assertAlmostEqual(th.doubling_min_distance,
                               S.doubling_min_distance / 3.744, places=6)
        self.assertAlmostEqual(th.star_max_length_full,
                               2 * th.star_max_length, places=6)

    def test_same_angle_means_same_sky_on_any_camera(self):
        """The point of the whole arcsecond business."""
        for scale, step in ((3.744, 2), (2.39, 3), (1.65, 4)):
            th = si.Thresholds.derive(S, geometry(pixel_scale=scale, step=step))
            self.assertAlmostEqual(th.track_min_span * scale * step,
                                   S.track_min_span, places=5)

    def test_tilt_becomes_a_cosine(self):
        th = si.Thresholds.derive(replace(S, track_max_tilt=60.0), geometry())
        self.assertAlmostEqual(th.track_cos_tilt, 0.5, places=6)


# ======================================================================
#  Component geometry
# ======================================================================

class TestPrincipalAxes(unittest.TestCase):

    def test_an_axis_aligned_blob_keeps_its_axis(self):
        major, minor, ax, ay = si._principal_axes(
            np.array([4.0]), np.array([1.0]), np.array([0.0]))
        self.assertAlmostEqual(major[0], 4.0)
        self.assertAlmostEqual(minor[0], 1.0)
        self.assertAlmostEqual(abs(ax[0]), 1.0)
        self.assertAlmostEqual(ay[0], 0.0)

    def test_a_symmetric_blob_leans_diagonally(self):
        _, _, ax, ay = si._principal_axes(
            np.array([2.0]), np.array([2.0]), np.array([1.0]))
        self.assertAlmostEqual(abs(ax[0]), abs(ay[0]), places=6)


class TestMeasureComponents(unittest.TestCase):

    def build(self, mask: np.ndarray):
        from scipy.ndimage import label
        labels, count = label(mask, structure=si.EIGHT_CONNECTED)
        return si.measure_components(labels, mask, mask.shape, count, 3)

    def test_a_bar_is_long_and_thin(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[20, 5:35] = True
        m = self.build(mask)
        i = np.nonzero(m["valid"])[0][0]
        self.assertAlmostEqual(m["length"][i], 29.0)
        self.assertGreater(m["ratio"][i], 20)
        self.assertAlmostEqual(abs(m["axis_x"][i]), 1.0, places=6)

    def test_a_diagonal_bar_is_still_thin(self):
        """The reason for principal axes: its bounding box is square."""
        mask = np.zeros((40, 40), dtype=bool)
        for k in range(5, 35):
            mask[k, k] = True
        m = self.build(mask)
        i = np.nonzero(m["valid"])[0][0]
        self.assertGreater(m["ratio"][i], 10)

    def test_a_square_is_not_elongated(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[18:23, 18:23] = True
        m = self.build(mask)
        i = np.nonzero(m["valid"])[0][0]
        self.assertLess(m["ratio"][i], 1.5)

    def test_edges_are_noticed(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[0, 0:10] = True
        m = self.build(mask)
        i = np.nonzero(m["valid"])[0][0]
        self.assertTrue(m["touches_edge"][i])

    def test_empty_mask_is_survivable(self):
        m = self.build(np.zeros((20, 20), dtype=bool))
        self.assertFalse(m["valid"].any())


# ======================================================================
#  Star shape, which the trailing verdict rests on
# ======================================================================

class TestMeasureStars(unittest.TestCase):

    def setUp(self):
        self.th = si.Thresholds.derive(S, geometry())
        self.settings = replace(S, sharpness_min_stars=20)

    def test_round_stars_are_barely_elongated(self):
        shape = si.measure_stars(star_field(), self.settings, self.th)
        self.assertTrue(shape.measured)
        self.assertGreater(shape.count, 50)
        self.assertLess(shape.elongation, 1.15)

    def test_smeared_stars_read_as_elongated(self):
        round_stars = si.measure_stars(star_field(sigma_long=1.3),
                                       self.settings, self.th)
        smeared = si.measure_stars(
            star_field(sigma_long=2.6, sigma_short=1.3, angle=0.4),
            self.settings, self.th)
        self.assertGreater(smeared.elongation, 1.5 * round_stars.elongation)

    def test_elongation_grows_with_the_smear(self):
        """It has to be monotonic, or the severity dial means nothing."""
        measured = [si.measure_stars(
            star_field(sigma_long=long, sigma_short=1.3, angle=0.4),
            self.settings, self.th).elongation
            for long in (1.3, 1.8, 2.4, 3.2)]
        self.assertEqual(measured, sorted(measured))

    def test_a_common_smear_direction_is_coherent(self):
        smeared = si.measure_stars(
            star_field(sigma_long=2.6, sigma_short=1.3, angle=0.4),
            self.settings, self.th)
        self.assertGreater(smeared.coherence, 0.8)

    def test_too_few_stars_means_no_measurement(self):
        shape = si.measure_stars(star_field(count=3), self.settings, self.th)
        self.assertFalse(shape.measured)


class TestDoubling(unittest.TestCase):

    def setUp(self):
        self.th = si.Thresholds.derive(S, geometry())

    def blobs(self, offset=None, count=60, seed=3):
        # Spread over an area like the real centre crop: packed too
        # tightly, chance pairs fall inside the matching tolerance and
        # drag the averaged offset off the true one.
        rng = np.random.default_rng(seed)
        stars = []
        for _ in range(count):
            x, y = rng.uniform(20, 1880), rng.uniform(20, 1040)
            stars.append({"cx": x, "cy": y, "size": 1000.0})
            if offset is not None:
                stars.append({"cx": x + offset[0], "cy": y + offset[1],
                              "size": 800.0})
        return stars

    def test_a_shared_offset_is_recovered(self):
        found = si.find_star_doubling(self.blobs(offset=(9.0, 7.0)), S, self.th)
        self.assertIsNotNone(found)
        # The offset is averaged over every pair agreeing to within the
        # matching tolerance, so a fraction of a pixel is all it claims.
        self.assertAlmostEqual(abs(found["dx"]), 9.0, delta=0.5)
        self.assertAlmostEqual(abs(found["dy"]), 7.0, delta=0.5)
        self.assertGreater(found["fraction"], S.doubling_min_fraction)

    def test_a_plain_field_has_no_dominant_offset(self):
        self.assertIsNone(si.find_star_doubling(self.blobs(), S, self.th))

    def test_a_line_of_blobs_is_not_doubling(self):
        """Segments of one trail repeat an offset; they must not count."""
        stars = [{"cx": 40.0 + 14 * k, "cy": 40.0 + 14 * k, "size": 900.0}
                 for k in range(20)]
        self.assertIsNone(si.find_star_doubling(stars, S, self.th))

    def test_a_large_knock_is_still_a_doubling(self):
        """A jolt is instantaneous, so its offset can be as big as it likes.

        A 300 arcsec ceiling once hid a real one at 719 arcsec, on a
        frame that looked clean to every other test.
        """
        found = si.find_star_doubling(self.blobs(offset=(180.0, 60.0)), S, self.th)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(abs(found["dx"]), 180.0, delta=2.0)

    def test_the_search_reaches_a_third_of_the_measured_region(self):
        """The bound belongs to the crop, not to any mount or night."""
        geo = geometry(width=2160, height=3840)
        th = si.Thresholds.derive(S, geo)
        expected = (S.doubling_max_fraction * S.sharpness_centre_fraction
                    * min(geo.width, geo.height))
        self.assertAlmostEqual(th.doubling_max_distance, expected)

    def test_the_bound_shrinks_with_a_smaller_sensor(self):
        wide = si.Thresholds.derive(S, geometry(width=2160, height=3840))
        small = si.Thresholds.derive(S, geometry(width=1080, height=1920))
        self.assertAlmostEqual(wide.doubling_max_distance,
                               2 * small.doubling_max_distance)

    def test_too_few_stars_is_declined(self):
        self.assertIsNone(si.find_star_doubling(self.blobs(count=2), S, self.th))


# ======================================================================
#  Tracks
# ======================================================================

class TestFindTrack(unittest.TestCase):

    def setUp(self):
        self.geo = geometry()
        self.th = si.Thresholds.derive(S, self.geo)

    def line_of(self, count, angle_deg=30.0, spacing=60.0, length=9.0,
                width=1.2, scatter=0.0, seed=1):
        """Fragments strung along one line, optionally jittered off it."""
        rng = np.random.default_rng(seed)
        angle = np.radians(angle_deg)
        pieces = []
        for k in range(count):
            along = k * spacing
            off = rng.normal(0.0, scatter) if scatter else 0.0
            pieces.append(segment(
                100 + along * np.cos(angle) - off * np.sin(angle),
                100 + along * np.sin(angle) + off * np.cos(angle),
                length, width, angle_deg, label=k))
        return pieces

    def test_a_dotted_line_is_a_track(self):
        track = si.find_track(self.line_of(6), S, self.th)
        self.assertIsNotNone(track)
        self.assertEqual(track.fragments, 6)
        self.assertLess(track.residual, self.th.track_max_residual)
        self.assertAlmostEqual(track.tilt, 30.0, places=0)

    def test_three_specks_are_enough_when_they_are_straight(self):
        """A trail can be faint enough to leave only three pieces."""
        track = si.find_track(self.line_of(3, spacing=120.0), S, self.th)
        self.assertIsNotNone(track)
        self.assertEqual(track.fragments, 3)

    def test_a_scattered_line_is_refused(self):
        """Same count, same span, but the pieces do not sit on the line."""
        track = si.find_track(self.line_of(6, scatter=4.0), S, self.th)
        self.assertIsNone(track)

    def test_one_long_fragment_is_a_track(self):
        track = si.find_track([segment(500, 500, 200.0, 2.0, 45.0)], S, self.th)
        self.assertIsNotNone(track)
        self.assertAlmostEqual(track.coverage, 1.0)

    def test_a_short_blob_is_not(self):
        """Too short to be anything, however thin."""
        self.assertIsNone(si.find_track([segment(500, 500, 5.0, 1.0, 45.0)],
                                        S, self.th))

    def test_fat_fragments_cannot_start_a_line(self):
        """Merged star pairs are short and fat; they must never seed."""
        fat = [replace(piece, ratio=3.0, width=piece.length / 3.0)
               for piece in self.line_of(6)]
        self.assertIsNone(si.find_track(fat, S, self.th))

    def test_a_weak_fragment_may_still_join(self):
        """One thin seed plus faint specks: the trail is recovered."""
        pieces = self.line_of(5)
        pieces = [pieces[0]] + [replace(p, ratio=4.5, width=p.length / 4.5)
                                for p in pieces[1:]]
        track = si.find_track(pieces, S, self.th)
        self.assertIsNotNone(track)
        self.assertEqual(track.fragments, 5)

    def test_a_cross_direction_fragment_is_left_out(self):
        pieces = self.line_of(5)
        pieces.append(segment(pieces[2].cx, pieces[2].cy, 9.0, 1.2, 110.0, 99))
        track = si.find_track(pieces, S, self.th)
        self.assertIsNotNone(track)
        self.assertEqual(track.fragments, 5)

    def test_no_fragments_no_track(self):
        self.assertIsNone(si.find_track([], S, self.th))


class TestTrackProfile(unittest.TestCase):

    def test_an_even_trail_reads_flat(self):
        image = np.zeros((200, 200), dtype=np.float32)
        labels = np.zeros((200, 200), dtype=np.int32)
        image[100, 20:180] = 500.0
        labels[100, 20:180] = 1
        track = si.Track(span=160.0, coverage=1.0, residual=0.0, fragments=1,
                         max_width=1.0, touches_edge=False, tilt=0.0,
                         centre_x=100.0, centre_y=100.0, unit_x=1.0,
                         unit_y=0.0, members=(1,))
        profile = si.measure_track_profile(image, labels, track, noise=10.0)
        self.assertLess(profile.head_ratio, 1.3)
        self.assertLess(profile.taper, 1.3)
        self.assertGreater(profile.glow, 0.0)

    def test_a_fading_trail_tapers(self):
        image = np.zeros((200, 200), dtype=np.float32)
        labels = np.zeros((200, 200), dtype=np.int32)
        image[100, 20:180] = np.linspace(1000.0, 50.0, 160)
        labels[100, 20:180] = 1
        track = si.Track(span=160.0, coverage=1.0, residual=0.0, fragments=1,
                         max_width=1.0, touches_edge=False, tilt=0.0,
                         centre_x=100.0, centre_y=100.0, unit_x=1.0,
                         unit_y=0.0, members=(1,))
        profile = si.measure_track_profile(image, labels, track, noise=10.0)
        self.assertGreater(profile.taper, 2.0)


class TestFieldShift(unittest.TestCase):

    def test_a_known_translation_is_recovered(self):
        th = si.Thresholds.derive(S, geometry())
        rng = np.random.default_rng(11)
        before = rng.uniform(0, 900, size=(80, 2))
        after = before + np.array([7.5, -4.0])
        shift = si.measure_field_shift(before, after, th)
        self.assertIsNotNone(shift)
        dx, dy, confidence = shift
        self.assertAlmostEqual(dx, 7.5, places=1)
        self.assertAlmostEqual(dy, -4.0, places=1)
        self.assertGreater(confidence, 0.8)

    def test_nothing_to_compare_against(self):
        th = si.Thresholds.derive(S, geometry())
        self.assertIsNone(si.measure_field_shift(None, np.zeros((20, 2)), th))


class TestCollinearity(unittest.TestCase):

    def test_points_on_a_line(self):
        points = np.column_stack((np.arange(10.0), np.arange(10.0) * 2))
        self.assertTrue(si.points_are_collinear(points, 1.0))

    def test_points_all_over_the_place(self):
        rng = np.random.default_rng(5)
        self.assertFalse(si.points_are_collinear(rng.uniform(0, 100, (30, 2)), 1.0))


# ======================================================================
#  The verdicts themselves
# ======================================================================

class TestClassification(unittest.TestCase):
    """The rules that turn measurements into alerts."""

    def setUp(self):
        self.geo = geometry()
        self.th = si.Thresholds.derive(S, self.geo)
        self.reference = si.SessionReference(
            background=1000.0, background_spread=20.0, noise=50.0,
            star_radius=1.0, star_elongation=1.20)

    def frame(self, elongation=1.20, radius=1.0, track=None, background=1000.0,
              noise=50.0, doubling=None):
        stars = si.StarShape(count=200, radius=radius, elongation=elongation,
                             coherence=0.5)
        return si.FrameStats(name="f.fit", path="f.fit", background=background,
                             peak=30000.0, noise=noise, stars=stars,
                             track=track, doubling=doubling)

    def categories(self, frame):
        alerts, _ = si.classify_frame(frame, self.reference, S, self.th)
        return {alert.category for alert in alerts}

    def track(self, fragments=10, coverage=0.9, width=1.0):
        return si.Track(span=400.0, coverage=coverage, residual=0.1,
                        fragments=fragments, max_width=2.0, touches_edge=False,
                        tilt=30.0, centre_x=500.0, centre_y=500.0,
                        unit_x=1.0, unit_y=0.0,
                        profile=si.TrackProfile(width=width))

    # -- trailing -------------------------------------------------------

    def test_round_sharp_stars_raise_nothing(self):
        self.assertEqual(self.categories(self.frame()), set())

    def test_a_frame_with_no_measurable_star_is_a_veil(self):
        """measure_stars gives up below sharpness_min_stars, and _ratio
        answers 1.0 when there is nothing to compare -- so this frame used
        to read as perfectly normal and raise nothing at all."""
        frame = self.frame()
        frame.stars = si.StarShape.none()
        self.assertIn(si.CATEGORY_CLOUDS, self.categories(frame))

    def test_a_starless_frame_says_why_rather_than_blaming_the_sky(self):
        frame = self.frame()
        frame.stars = si.StarShape.none()
        alerts, _ = si.classify_frame(frame, self.reference, S, self.th)
        self.assertIn("stars measurable", " ".join(a.message for a in alerts))

    def test_oval_stars_are_trailing(self):
        frame = self.frame(elongation=1.20 * 1.4)
        self.assertIn(si.CATEGORY_TRAILING, self.categories(frame))

    def test_bloated_stars_are_trailing_too(self):
        """Focus drift and seeing swell the stars without leaning them."""
        frame = self.frame(elongation=1.20, radius=1.5)
        self.assertIn(si.CATEGORY_TRAILING, self.categories(frame))

    def test_bloat_says_so_rather_than_claiming_a_smear(self):
        alerts, _ = si.classify_frame(self.frame(radius=1.5), self.reference,
                                      S, self.th)
        message = " ".join(a.message for a in alerts)
        self.assertIn("not leaning anywhere", message)

    def test_a_jolt_is_trailing_even_with_perfect_stars(self):
        frame = self.frame(doubling={"dx": 8.0, "dy": 6.0, "distance": 10.0,
                                     "pairs": 30, "fraction": 0.3})
        self.assertIn(si.CATEGORY_TRAILING, self.categories(frame))

    # -- the gate on frames whose own stars are trailed -----------------

    def test_a_sparse_track_stands_on_a_sharp_frame(self):
        frame = self.frame(track=self.track(fragments=3, coverage=0.1))
        self.assertIn(si.CATEGORY_SATELLITES, self.categories(frame))

    def test_a_sparse_track_is_dropped_on_a_trailed_frame(self):
        """Three dashes fall into line often enough to matter."""
        frame = self.frame(elongation=1.20 * 1.4,
                           track=self.track(fragments=3, coverage=0.1))
        self.assertNotIn(si.CATEGORY_SATELLITES, self.categories(frame))

    def test_a_well_broken_track_survives_a_trailed_frame(self):
        """Twenty pieces on one line is not luck, however smeared the stars."""
        frame = self.frame(elongation=1.20 * 1.4,
                           track=self.track(fragments=20, coverage=0.1))
        self.assertIn(si.CATEGORY_SATELLITES, self.categories(frame))

    def test_a_continuous_track_survives_a_trailed_frame(self):
        frame = self.frame(elongation=1.20 * 1.4,
                           track=self.track(fragments=2, coverage=0.9))
        self.assertIn(si.CATEGORY_SATELLITES, self.categories(frame))

    # -- what a wide track becomes --------------------------------------

    def test_a_thin_track_is_a_satellite(self):
        frame = self.frame(track=self.track(width=1.0))
        self.assertIn(si.CATEGORY_SATELLITES, self.categories(frame))

    def test_a_wide_even_track_on_a_bright_frame_is_an_aircraft(self):
        wide = self.track(width=self.th.wide_track_width * 2)
        frame = self.frame(track=wide, noise=self.reference.noise * 2)
        self.assertIn(si.CATEGORY_AIRCRAFT, self.categories(frame))

    def test_a_wide_tapered_track_is_a_meteor(self):
        wide = si.replace(self.track(width=self.th.wide_track_width * 2),
                          profile=si.TrackProfile(
                              width=self.th.wide_track_width * 2, taper=4.0))
        self.assertIn(si.CATEGORY_METEORS, self.categories(self.frame(track=wide)))

    # -- haze ------------------------------------------------------------

    def test_a_raised_background_is_haze(self):
        frame = self.frame(background=1000.0 + 4 * 20.0)
        self.assertIn(si.CATEGORY_CLOUDS, self.categories(frame))

    def test_alerts_are_independent(self):
        frame = self.frame(elongation=1.20 * 1.4, background=1000.0 + 4 * 20.0,
                           track=self.track())
        self.assertEqual(self.categories(frame),
                         {si.CATEGORY_CLOUDS, si.CATEGORY_TRAILING,
                          si.CATEGORY_SATELLITES})


# ======================================================================
#  Anything that removes a file
# ======================================================================

class TestHousekeeping(unittest.TestCase):
    """The rule: a file goes only if the same name is still in JPG/."""

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.folder = self.root.name
        os.makedirs(os.path.join(self.folder, "JPG"))
        self.review = os.path.join(self.folder, S.review_folder)
        os.makedirs(os.path.join(self.review, si.CATEGORY_TRAILING))
        os.makedirs(os.path.join(self.review, "Field_Shift"))   # a retired one
        os.makedirs(os.path.join(self.review, "My_keepers"))    # not ours
        self.write("JPG/frame_0001.jpg")
        self.write(f"{S.review_folder}/{si.CATEGORY_TRAILING}/frame_0001.jpg")
        self.write(f"{S.review_folder}/Field_Shift/frame_0001.jpg")
        self.write(f"{S.review_folder}/{si.CATEGORY_TRAILING}/moved_only.jpg")
        self.write(f"{S.review_folder}/My_keepers/frame_0001.jpg")
        self.write(f"{S.review_folder}/notes.txt")
        self.write(f"{S.review_folder}/{S.report_csv}")

    def tearDown(self):
        self.root.cleanup()

    def write(self, relative: str) -> None:
        with open(os.path.join(self.folder, relative), "w") as handle:
            handle.write("x")

    def exists(self, relative: str) -> bool:
        return os.path.exists(os.path.join(self.folder, relative))

    def test_it_clears_what_it_wrote(self):
        si.clean_review_folder(self.folder, S)
        self.assertFalse(self.exists(
            f"{S.review_folder}/{si.CATEGORY_TRAILING}/frame_0001.jpg"))
        self.assertFalse(self.exists(f"{S.review_folder}/{S.report_csv}"))

    def test_it_clears_folders_from_older_versions(self):
        si.clean_review_folder(self.folder, S)
        self.assertFalse(self.exists(f"{S.review_folder}/Field_Shift"))

    def test_it_never_removes_the_only_copy(self):
        """A preview an earlier --mode move took out of JPG/ must survive."""
        si.clean_review_folder(self.folder, S)
        self.assertTrue(self.exists(
            f"{S.review_folder}/{si.CATEGORY_TRAILING}/moved_only.jpg"))

    def test_it_leaves_your_own_files_alone(self):
        si.clean_review_folder(self.folder, S)
        self.assertTrue(self.exists(f"{S.review_folder}/notes.txt"))
        self.assertTrue(self.exists(f"{S.review_folder}/My_keepers/frame_0001.jpg"))

    def test_it_survives_a_missing_review_folder(self):
        empty = tempfile.TemporaryDirectory()
        self.assertEqual(si.clean_review_folder(empty.name, S), 0)
        empty.cleanup()


class TestDeletionSelection(unittest.TestCase):

    def setUp(self):
        self.flagged = {
            "a.fit": [si.CATEGORY_TRAILING],
            "b.fit": [si.CATEGORY_CLOUDS, si.CATEGORY_TRAILING],
            "c.fit": [si.CATEGORY_SATELLITES],
        }

    def test_no_filter_takes_everything_flagged(self):
        self.assertEqual(len(si.select_for_deletion(self.flagged, None)), 3)

    def test_a_filter_restricts_it(self):
        chosen = si.select_for_deletion(self.flagged, {si.CATEGORY_TRAILING})
        self.assertEqual(set(chosen), {"a.fit", "b.fit"})

    def test_a_filter_can_match_nothing(self):
        self.assertEqual(
            si.select_for_deletion(self.flagged, {si.CATEGORY_METEORS}), {})

    def test_clean_frames_are_never_in_the_selection(self):
        self.assertNotIn("clean.fit", si.select_for_deletion(self.flagged, None))


# ======================================================================
#  Odds and ends
# ======================================================================

class TestSmallHelpers(unittest.TestCase):

    def test_frame_label_keeps_the_timestamp(self):
        self.assertEqual(
            si.frame_label("Light_LBN 550_60.0s_IRCUT_20260830-000305.fit"),
            "20260830-000305")

    def test_frame_label_copes_without_a_prefix(self):
        self.assertEqual(si.frame_label("plain.fit"), "plain")

    def test_sampled_percentile_matches_the_full_one(self):
        rng = np.random.default_rng(2)
        image = rng.normal(1000, 50, (1200, 1200)).astype(np.float32)
        sampled = si.high_percentile(image, 98.5)
        exact = float(np.percentile(image, 98.5))
        self.assertLess(abs(sampled - exact) / exact, 0.01)

    def test_small_images_are_not_sampled(self):
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        self.assertEqual(si.high_percentile(image, 50.0),
                         float(np.percentile(image, 50.0)))

    def test_every_category_has_a_folder_to_be_cleaned_from(self):
        """Add a category and forget this list, and its folder is never
        cleared again -- stale previews would pile up in it for ever."""
        for category in si.ALL_CATEGORIES:
            self.assertIn(category, si.GENERATED_FOLDERS)


class TestCaseExport(unittest.TestCase):
    """Exporting a session's measurements, and replaying them.

    A bundle is only worth anything if replaying it gives back exactly
    what the run said. These check the machinery that would break that
    silently: a field dropped on the way out, a tuple that comes back
    from JSON as a list, a stale bundle that no longer says so.
    """

    def setUp(self):
        self.geo = geometry()
        self.th = si.Thresholds.derive(S, self.geo)
        self.reference = si.SessionReference(
            background=1000.0, background_spread=20.0, noise=50.0,
            star_radius=1.0, star_elongation=1.20, star_count=3.0)

    def frame(self, name="Light_M42_60.0s_IRCUT_20260830-004055.fit",
              elongation=1.20, track=None, doubling=None):
        stars = si.StarShape(
            count=3, radius=1.1, elongation=elongation, coherence=0.4,
            blobs=[{"cx": 10.0, "cy": 20.0, "size": 300.0,
                    "elongation": 1.1, "angle": 0.3},
                   {"cx": 40.0, "cy": 60.0, "size": 250.0,
                    "elongation": 1.2, "angle": -0.2},
                   {"cx": 80.0, "cy": 90.0, "size": 200.0,
                    "elongation": 1.0, "angle": 0.0}])
        return si.FrameStats(
            name=name, path=os.path.join("somewhere", name),
            background=1010.0, peak=31000.0, noise=52.0, stars=stars,
            track=track, doubling=doubling,
            star_positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
            field_shift=(2.5, -1.5, 0.8),
            export_segments=[self.segment(ratio=3.2), self.segment(ratio=9.0)])

    def segment(self, ratio=9.0, length=40.0):
        return si.Segment(label=7, cx=100.0, cy=200.0, length=length,
                          width=length / ratio, ratio=ratio, axis_x=1.0,
                          axis_y=0.0, touches_edge=False)

    def track(self):
        return si.Track(span=400.0, coverage=0.9, residual=0.12, fragments=8,
                        max_width=2.0, touches_edge=True, tilt=31.5,
                        centre_x=500.0, centre_y=520.0, unit_x=1.0, unit_y=0.0,
                        members=(3, 7, 11),
                        profile=si.TrackProfile(head_ratio=1.4, head_position=0.3,
                                                taper=1.1, glow=4.0, peak=9.0,
                                                width=1.05))

    def bundle(self, frames, rows=None):
        rows = rows or [{"categories": ""} for _ in frames]
        return si.build_case_bundle(frames, rows, self.reference, self.geo, S,
                                    labels={}, session_id="test")

    # -- what a case keeps ----------------------------------------------

    def test_a_case_round_trips_everything_a_verdict_rests_on(self):
        original = self.frame(track=self.track(),
                              doubling={"dx": 8.0, "dy": 6.0, "distance": 10.0,
                                        "pairs": 30, "fraction": 0.31})
        restored = si.frame_of_case(si.frame_case(original, 0, ["Trailing"], None))

        self.assertEqual(restored.background, original.background)
        self.assertEqual(restored.noise, original.noise)
        self.assertEqual(restored.stars, original.stars)
        self.assertEqual(restored.track, original.track)
        self.assertEqual(restored.doubling, original.doubling)
        self.assertEqual(restored.field_shift, original.field_shift)
        self.assertEqual(restored.export_segments, original.export_segments)
        np.testing.assert_allclose(restored.star_positions,
                                   original.star_positions)

    def test_a_frame_with_no_track_round_trips_as_one(self):
        restored = si.frame_of_case(si.frame_case(self.frame(), 0, [], None))
        self.assertIsNone(restored.track)
        self.assertIsNone(restored.doubling)

    def test_a_bundle_classifies_exactly_as_the_session_did(self):
        """The whole point: same numbers in, same verdict out."""
        frames = [self.frame(elongation=1.20),
                  self.frame(name="Light_M42_60.0s_IRCUT_20260830-004156.fit",
                             elongation=1.20 * 1.5),
                  self.frame(name="Light_M42_60.0s_IRCUT_20260830-004257.fit",
                             track=self.track())]
        before = [si.classify_frame(f, self.reference, S, self.th)[0]
                  for f in frames]

        replayed, reference, settings, geo = si.cases_session(
            self.bundle(frames))
        thresholds = si.Thresholds.derive(settings, geo)
        after = [si.classify_frame(f, reference, settings, thresholds)[0]
                 for f in replayed]

        self.assertEqual([[a.category for a in alerts] for alerts in after],
                         [[a.category for a in alerts] for alerts in before])

    def test_the_session_yardstick_travels_with_the_frames(self):
        """What makes a partial export legitimate: the reference is the
        whole night's, measured before any frame was dropped."""
        bundle = self.bundle([self.frame()])
        bundle["frames"] = bundle["frames"][:1]
        _, reference, _, _ = si.cases_session(bundle)
        self.assertEqual(reference, self.reference)

    def test_a_bundle_without_a_star_count_has_one_derived(self):
        """Bundles written before the veil test existed carry no star
        count. Deriving it from the frames is exact for a whole session,
        so an old bundle still scores -- rather than silently having the
        veil branch disabled, which is the failure that would look fine."""
        bundle = self.bundle([self.frame()])
        del bundle["session"]["reference"]["star_count"]
        _, reference, _, _ = si.cases_session(bundle)
        self.assertEqual(reference.star_count, 3.0)

    def test_a_partial_session_writes_no_bundle(self):
        """The format rests on the yardstick covering the whole night. Built
        on the survivors of a session that lost frames, every ratio in it
        would be measured against the wrong reference -- and a memory spike
        is enough to cause that."""
        frames = [self.frame()]
        written = si.write_cases(self.tmp(), frames, [{"categories": ""}],
                                 self.reference, self.geo, S, expected=5)
        self.assertIsNone(written)

    def test_a_complete_session_writes_one(self):
        folder = self.tmp()
        # cases_folder() is where the bundles collect, beside the script.
        # A test must not write there, so it is pointed at the scratch dir.
        original = si.cases_folder
        si.cases_folder = lambda: folder
        self.addCleanup(setattr, si, "cases_folder", original)
        written = si.write_cases(folder, [self.frame()],
                                 [{"categories": ""}], self.reference,
                                 self.geo, S, expected=1)
        self.assertTrue(written and os.path.isfile(written))
        self.assertTrue(written.startswith(folder))

    # -- names that do not name the target -------------------------------

    def test_a_frame_id_is_the_timestamp_and_never_the_target(self):
        identifier = si.case_frame_id("Light_LBN 550_60.0s_IRCUT_20260830-004055.fit", 3)
        self.assertEqual(identifier, "20260830-004055")

    def test_a_frame_with_no_timestamp_falls_back_on_its_position(self):
        self.assertEqual(si.case_frame_id("whatever.fit", 3), "frame-0003")

    def test_the_session_id_comes_from_the_data_not_the_folder(self):
        """So renaming the sub folder does not give one night two names."""
        self.assertEqual(si.session_identifier([self.frame()]),
                         "20260830-004055")

    def test_the_bundle_is_labelled_with_its_target(self):
        """The one place a bundle says what it looked at. It has to, or the
        owner cannot tell a dozen of them apart -- but only on the outside,
        and nothing scored may read it."""
        self.assertEqual(si.session_label(os.path.join("x", "LBN 550_sub")),
                         "LBN_550")

    def test_a_target_that_would_not_survive_a_file_name_is_cleaned_up(self):
        self.assertEqual(si.session_label("NGC 6910: IRCUT*_sub"),
                         "NGC_6910_IRCUT")

    def test_the_target_is_a_label_and_not_an_identity(self):
        frames = [self.frame()]
        renamed = si.build_case_bundle(frames, [{"categories": ""}],
                                       self.reference, self.geo, S, {},
                                       si.session_identifier(frames), "Elsewhere")
        self.assertEqual(renamed["session"]["id"], "20260830-004055")

    # -- settings across the round trip -----------------------------------

    def test_settings_come_back_with_their_tuples_intact(self):
        """JSON has no tuple, and a frozen dataclass will hold a list
        without complaining -- then compare unequal for ever after."""
        restored = si.cases_settings(self.bundle([self.frame()]))
        self.assertEqual(restored, S)

    def test_a_setting_this_build_no_longer_has_is_dropped(self):
        bundle = self.bundle([self.frame()])
        bundle["session"]["settings"]["retired_threshold"] = 3.0
        self.assertEqual(si.cases_settings(bundle), S)

    def test_a_bundle_from_another_format_is_refused(self):
        path = os.path.join(self.tmp(), "cases.json")
        bundle = self.bundle([self.frame()])
        bundle["format"] = si.CASE_FORMAT_VERSION + 1
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle)
        with self.assertRaises(ValueError):
            si.read_cases(path)

    # -- the spatial primitives -------------------------------------------

    def test_the_spatial_summary_covers_every_tile(self):
        small = np.zeros((80, 60), dtype=np.float64) + 100.0
        measures = {"valid": np.array([True, True, True]),
                    "cx": np.array([5.0, 55.0, 30.0]),
                    "cy": np.array([5.0, 75.0, 40.0]),
                    "length": np.array([3.0, 4.0, 5.0]),
                    "ratio": np.array([1.1, 1.2, 9.0]),
                    "pixels": np.array([9.0, 16.0, 25.0])}
        summary = si.spatial_summary(small, measures, S)
        self.assertEqual(len(summary["tiles"]), S.export_grid ** 2)
        self.assertEqual(sum(tile[0] for tile in summary["tiles"]), 3)

    def test_a_component_in_a_corner_lands_in_a_corner_tile(self):
        """The whole point of the grid: a cloud over one part of the frame
        has to show up in one part of the summary."""
        small = np.zeros((80, 60), dtype=np.float64)
        measures = {"valid": np.array([True]), "cx": np.array([59.0]),
                    "cy": np.array([79.0]), "length": np.array([3.0]),
                    "ratio": np.array([1.0]), "pixels": np.array([9.0])}
        tiles = si.spatial_summary(small, measures, S)["tiles"]
        self.assertEqual(tiles[-1][0], 1)
        self.assertEqual(sum(tile[0] for tile in tiles[:-1]), 0)

    def test_the_star_map_is_ordered_brightest_first(self):
        measures = {"valid": np.array([True, True, True]),
                    "cx": np.array([1.0, 2.0, 3.0]),
                    "cy": np.array([1.0, 2.0, 3.0]),
                    "length": np.ones(3), "ratio": np.ones(3),
                    "pixels": np.array([5.0, 50.0, 20.0])}
        star_map = si.spatial_summary(np.zeros((40, 40)), measures, S)["star_map"]
        self.assertEqual([row[2] for row in star_map], [50, 20, 5])

    # -- the staleness guard ----------------------------------------------

    def test_the_fingerprint_moves_when_a_measurement_function_does(self):
        """The guard that stops a bundle being scored by code that would
        no longer measure it the same way."""
        before = si.measurement_fingerprint()
        original = si.load_frame
        try:
            def load_frame(path):  # deliberately not the real one
                return None
            si.load_frame = load_frame
            self.assertNotEqual(si.measurement_fingerprint(), before)
        finally:
            si.load_frame = original
        self.assertEqual(si.measurement_fingerprint(), before)

    # -- the segment filter -------------------------------------------------

    def test_the_join_bar_keeps_only_long_thin_fragments(self):
        short = self.segment(ratio=9.0, length=self.th.segment_min_length / 2)
        fat = self.segment(ratio=S.segment_min_ratio - 0.5, length=40.0)
        good = self.segment(ratio=S.segment_min_ratio + 0.5, length=40.0)
        self.assertEqual(si.select_track_segments([short, fat, good], S, self.th),
                         [good])

    def test_the_export_net_is_wider_than_the_bar_it_is_filtered_to(self):
        """A bundle keeps fragments the run itself threw away, so the join
        bar can later be lowered as well as raised. If these two ever met,
        that headroom would be gone without anything failing."""
        self.assertLess(S.export_segment_min_ratio, S.segment_min_ratio)
        self.assertLess(S.export_segment_length_factor, 1.0)

    # -- labels ---------------------------------------------------------

    def test_labels_are_matched_on_the_timestamp_not_the_file_name(self):
        """A Seestar name carries '60.0s', so splitext mangles it and a
        .fit row, a .jpg row and a bundle id must still agree."""
        path = os.path.join(self.tmp(), "labels.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("frame,labels,note,detected,updated\n")
            handle.write("Light_LBN 550_60.0s_IRCUT_20260830-004055,"
                         "Trailing Doubling,knocked,Trailing,2026-09-05\n")
        labels = si.read_frame_labels(path)
        self.assertEqual(labels["20260830-004055"]["labels"],
                         ["Trailing", "Doubling"])

    def test_a_missing_labels_file_is_not_an_error(self):
        self.assertEqual(si.read_frame_labels(os.path.join(self.tmp(), "no.csv")),
                         {})

    def test_a_case_carries_the_label_and_the_verdict_of_the_day(self):
        case = si.frame_case(self.frame(), 0, ["Trailing"],
                             {"labels": ["Satellites"], "note": "faint"})
        self.assertEqual(case["verdict"], ["Trailing"])
        self.assertEqual(case["labels"], ["Satellites"])
        self.assertEqual(case["note"], "faint")

    def tmp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return folder.name


class TestStaleness(unittest.TestCase):
    """Which changes condemn a bundle, and which must not.

    This is what decides whether the owner has to re-run the detector
    over every night they have kept. Getting it wrong in the lax
    direction validates a version that no longer exists; getting it wrong
    in the strict direction throws the corpus away on a comment.
    """

    def bundle(self, **overrides):
        fingerprints = dict(si.measurement_fingerprints())
        fingerprints.update(overrides)
        return {"fingerprints": fingerprints}

    def test_an_untouched_build_is_not_stale(self):
        self.assertEqual(rc.stale_parts(self.bundle()), {})

    def test_a_changed_assembly_is_not_stale(self):
        """Adding a field to the export rewrites analyse_frame and changes
        no number. If this ever fails, every bundle dies on every idea."""
        self.assertEqual(rc.stale_parts(self.bundle(assembly="0000deadbeef")), {})

    def test_a_changed_measurement_group_is_named(self):
        stale = rc.stale_parts(self.bundle(frame="0000deadbeef"))
        self.assertEqual(list(stale), ["frame"])

    def test_one_changed_group_does_not_condemn_the_others(self):
        stale = rc.stale_parts(self.bundle(profile="0000deadbeef"))
        self.assertNotIn("stars", stale)
        self.assertNotIn("frame", stale)

    def test_a_format_1_bundle_falls_back_to_the_combined_hash(self):
        fresh = {"measurement_fingerprint": si.measurement_fingerprint()}
        self.assertEqual(rc.stale_parts(fresh), {})
        old = {"measurement_fingerprint": "0000deadbeef"}
        self.assertEqual(list(rc.stale_parts(old)), ["everything (format 1)"])


class TestReplayScoring(unittest.TestCase):
    """replay_cases, where a wrong answer would look like a real result."""

    def test_a_measurement_setting_is_refused_rather_than_ignored(self):
        """Silently accepting it would show a threshold sweep that never
        swept anything."""
        with self.assertRaises(SystemExit):
            rc.parse_override("bright_percentile=97.0")

    def test_a_decision_setting_is_cast_to_its_declared_type(self):
        self.assertEqual(rc.parse_override("trailing_elongation_ratio=1.25"),
                         ("trailing_elongation_ratio", 1.25))
        self.assertEqual(rc.parse_override("track_min_fragments=4"),
                         ("track_min_fragments", 4))

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(SystemExit):
            rc.parse_override("no_such_thing=1")

    def test_doubling_and_trailing_are_one_verdict(self):
        self.assertEqual(rc.normalise(["Doubling"]), (si.CATEGORY_TRAILING,))

    def test_the_three_track_labels_score_as_one(self):
        """A satellite reported as a meteor is the same detection wearing
        a different hat, not a miss and a false positive."""
        self.assertEqual(rc.normalise([si.CATEGORY_METEORS]),
                         (si.CATEGORY_SATELLITES,))
        self.assertEqual(rc.normalise([si.CATEGORY_AIRCRAFT]),
                         (si.CATEGORY_SATELLITES,))

    def test_a_join_bar_below_the_exported_net_is_refused(self):
        """Below it the fragments were never written down, so the replay
        would report fewer tracks than the frames themselves would."""
        bundle = {"session": {"segment_floor": {"length": 4.0, "ratio": 3.0}}}
        settings = replace(S, segment_min_ratio=2.0)
        thresholds = si.Thresholds.derive(settings, geometry())
        with self.assertRaises(SystemExit):
            rc.check_segment_floor(bundle, settings, thresholds)


class TestSessionLayout(unittest.TestCase):
    """Finding a session's frames and previews, whatever the folder holds."""

    NAME = "Light_M 45_10.0s_LP_20261014-215003"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = os.path.join(self.temp.name, "M 45_sub")

    def touch(self, relative: str) -> str:
        path = os.path.join(self.root, *relative.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write("x")
        return path

    def test_the_seestar_layout(self):
        self.touch(f"Lights/{self.NAME}.fit")
        self.touch(f"JPG/{self.NAME}.jpg")
        self.touch(f"JPG/{self.NAME}_thn.jpg")
        session = sl.resolve(self.root)
        self.assertEqual([os.path.basename(f) for f in session.frames],
                         [f"{self.NAME}.fit"])
        self.assertEqual(list(session.previews), [self.NAME])

    def test_a_stack_beside_lights_is_not_a_sub(self):
        """The subs in Lights/ win over whatever .fit sits at the root."""
        self.touch(f"Lights/{self.NAME}.fit")
        self.touch("NGC_7331_69x20sec.fit")
        self.assertEqual(len(sl.resolve(self.root).frames), 1)

    def test_seestar_stacks_and_thumbnails_are_never_frames(self):
        self.touch(f"{self.NAME}.fit")
        self.touch("Stacked_75_M 45_10.0s_LP_20261014-220000.fit")
        self.touch(f"{self.NAME}_thn.jpg")
        session = sl.resolve(self.root)
        self.assertEqual(len(session.frames), 1)
        self.assertEqual(session.previews, {})

    def test_stacks_are_recognised_by_name(self):
        self.touch(f"{self.NAME}.fit")
        for name in ("Video_Stacked_Lunar_20260726-232815.fit",
                     "NGC7635_stack_d3x_S50_140h.fit",
                     "SH2-132_301x60sec_T21degC_2026-09-19.fit",
                     "LP_NGC_7635_1019x30sec_T26degC_2026-08-18.fit"):
            self.touch(name)
        self.assertEqual(len(sl.resolve(self.root).frames), 1)

    def write_fits(self, relative: str, cards: dict) -> str:
        path = self.touch(relative)
        header = "".join(f"{k:<8}= {v:>20}".ljust(80) for k, v in cards.items())
        header += "END".ljust(80)
        with open(path, "wb") as handle:
            handle.write(header.ljust(2880).encode("ascii"))
        return path

    def test_a_stack_named_like_a_sub_is_read_from_its_header(self):
        """A Seestar stack renamed to its timestamp looks exactly like a sub."""
        for stamp in ("20260829-032001", "20260914-223949"):
            self.write_fits(f"{stamp}.fit", {"NAXIS": 3, "STACKCNT": 946})
        self.assertEqual(sl.resolve(self.root).frames, [])

    def test_a_sub_header_is_not_a_stack(self):
        path = self.write_fits(f"{self.NAME}.fit", {"NAXIS": 2, "EXPTIME": 10})
        self.assertFalse(sl.is_stacked(path))

    def test_everything_side_by_side(self):
        self.touch(f"{self.NAME}.fit")
        self.touch(f"{self.NAME}.jpg")
        session = sl.resolve(self.root)
        self.assertIsNotNone(session.preview_for(session.frames[0]))

    def test_case_does_not_matter(self):
        self.touch(f"lights/{self.NAME}.FIT")
        self.assertEqual(len(sl.resolve(self.root).frames), 1)

    def test_the_review_copies_are_not_previews(self):
        self.touch(f"{self.NAME}.fit")
        self.touch(f"REVIEW/Trailing/{self.NAME}.jpg")
        self.assertEqual(sl.resolve(self.root).previews, {})

    def test_picking_the_lights_folder_finds_the_session(self):
        self.touch(f"Lights/{self.NAME}.fit")
        self.assertEqual(sl.session_root(os.path.join(self.root, "Lights")),
                         os.path.abspath(self.root))

    def test_a_session_at_the_root_stays_where_it_is(self):
        self.touch(f"{self.NAME}.fit")
        self.assertEqual(sl.session_root(self.root), os.path.abspath(self.root))

    def test_sessions_are_found_under_a_parent_but_not_inside_one(self):
        self.touch(f"Lights/{self.NAME}.fit")
        other = os.path.join(self.temp.name, "nested", "M 81_sub")
        os.makedirs(other)
        open(os.path.join(other, "a.fit"), "w").close()
        found = [s.root for s in sl.find_sessions(self.temp.name)]
        self.assertEqual(found, sorted([os.path.abspath(self.root),
                                        os.path.abspath(other)], key=str.lower))

    def test_a_custom_layout(self):
        self.touch(f"raw/{self.NAME}.fits")
        self.touch(f"previews/{self.NAME}.jpg")
        layout = sl.Layout(frames=sl.Layout.parse("raw/*.fits"),
                           previews=sl.Layout.parse(r"previews\*.jpg"))
        session = sl.resolve(self.root, layout)
        self.assertEqual(len(session.frames), 1)
        self.assertIsNotNone(session.preview_for(session.frames[0]))

    def test_housekeeping_follows_the_previews_wherever_they_are(self):
        """A copy is cleared only when its preview exists, in any layout."""
        self.touch(f"Lights/{self.NAME}.fit")
        self.touch(f"Lights_JPG/{self.NAME}.jpg")
        self.touch(f"REVIEW/Trailing/{self.NAME}.jpg")
        self.touch("REVIEW/Trailing/moved_only.jpg")
        si.clean_review_folder(self.root, S)
        trailing = os.path.join(self.root, "REVIEW", "Trailing")
        self.assertEqual(os.listdir(trailing), ["moved_only.jpg"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
