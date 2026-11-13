# -*- coding: utf-8 -*-
"""Re-run the detector's decisions on exported cases, without the frames.

``subframe_inspector.py --export-cases`` writes one bundle per session,
beside the script rather than inside the session folder so that it
outlives the frames: every measurement it made, none of the pixels, a
couple of megabytes against the four gigabytes of FITS behind them. This
reads those bundles back, re-runs the classification on them with
whatever settings you ask for, and scores the result against the labels
the owner recorded.

What that buys is a growing corpus. A bundle keeps its own session's
yardstick -- the sharpest quarter, the median sky -- so cases from
different targets and different nights can be scored as one set, and a
threshold change can be judged on every session ever exported rather
than on the three that happen to be on this disk::

    python replay_cases.py                          # score as exported
    python replay_cases.py --set trailing_elongation_ratio=1.25
    python replay_cases.py --set seed_min_ratio=4.0 --set track_min_coverage=0.4

The limit is firm and worth stating: this replays *decisions*, never
*measurements*. Anything that would change how a number was obtained --
the decimation, the top-hat, the detection floor, the star selection --
needs the FITS. Those settings are refused rather than silently ignored,
and a bundle measured by different code is refused too, because a stale
bundle scored as fresh is the one failure that would quietly waste
weeks. Re-export from the frames when that happens.

Nothing here re-implements a rule. The classification, the track search,
the doubling vote and the segment filter are all imported from
``subframe_inspector`` and called as the tool calls them -- a second copy
of the star selection in another tool of this project once drifted out of
step with three fixes and validated a version that no longer existed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
from dataclasses import dataclass, fields, replace
from typing import Sequence

from subframe_inspector import (  # noqa: E402 -- grouped on purpose
    CATEGORY_CLOUDS, CATEGORY_METEORS, CATEGORY_SATELLITES, CATEGORY_TRAILING,
    CATEGORY_AIRCRAFT, DEFAULT_SETTINGS, MEASUREMENT_SETTINGS, FrameStats,
    MEASUREMENT_PARTS,
    Settings, Thresholds, cases_session, classify_frame, find_star_doubling,
    find_sub_folders, find_track, mark_recurring_tracks,
    measurement_fingerprint,
    measurement_fingerprints, read_cases, read_frame_labels, session_label,
    select_track_segments,
)
from session_layout import configured_layout, resolve  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# The owner labels a doubled frame "Doubling"; the detector files it
# under Trailing, with the offset spelled out in the message. The two
# agree, so scoring treats them as one word.
LABEL_ALIASES = {"Doubling": CATEGORY_TRAILING}

# Satellite, meteor and aircraft are one detection wearing three hats:
# the track is found the same way and only the width and brightness
# choose the label. Scored separately, a satellite reported as a meteor
# would read as a miss *and* a false positive, which says nothing useful.
TRACK_FAMILY = (CATEGORY_SATELLITES, CATEGORY_METEORS, CATEGORY_AIRCRAFT)

SCORED = (CATEGORY_TRAILING, CATEGORY_SATELLITES, CATEGORY_CLOUDS)

# Re-running the pairwise vote costs about 175 ms a frame -- nothing in
# the tool itself, where it is spread over a pool of workers, but a
# minute of a corpus replay that is otherwise instant. So it is only
# re-run when one of its own settings has moved; otherwise the exported
# answer is already the answer these settings give.
DOUBLING_SETTINGS = tuple(f.name for f in fields(Settings)
                          if f.name.startswith("doubling"))


@dataclass
class Replayed:
    """One frame, as it was exported and as it comes out now."""

    session: str
    frame: str
    labels: tuple[str, ...]
    before: tuple[str, ...]
    after: tuple[str, ...]
    # True when the track search found a different line from the exported
    # one. Its brightness and width cannot be re-measured without the
    # pixels, so the meteor and aircraft branches are blind on this frame.
    profile_lost: bool = False


def normalise(categories: Sequence[str]) -> tuple[str, ...]:
    """Fold the aliases and the track family, so two names can be compared."""
    folded = set()
    for name in categories:
        name = LABEL_ALIASES.get(name, name)
        folded.add(CATEGORY_SATELLITES if name in TRACK_FAMILY else name)
    return tuple(sorted(folded))


def parse_override(text: str) -> tuple[str, object]:
    """Read one --set name=value, cast to the type Settings declares."""
    name, _, raw = text.partition("=")
    name, raw = name.strip(), raw.strip()
    declared = {f.name: f.type for f in fields(Settings)}
    if name not in declared:
        raise SystemExit(f"⚠️ '{name}' is not a setting.")
    if name in MEASUREMENT_SETTINGS:
        raise SystemExit(
            f"⚠️ '{name}' decides how the pixels are measured, not how the "
            f"numbers are judged. A bundle cannot answer for it — re-run "
            f"subframe_inspector.py on the FITS instead.")
    kind = str(declared[name])
    try:
        if "int" in kind and "float" not in kind:
            return name, int(raw)
        if "float" in kind:
            return name, float(raw)
        if "bool" in kind:
            return name, raw.lower() in ("1", "true", "yes", "on")
    except ValueError:
        raise SystemExit(f"⚠️ '{raw}' is not a valid value for {name}.")
    return name, raw


def stale_parts(bundle: dict) -> dict[str, tuple[str, str]]:
    """Which measurement groups no longer describe what this build measures.

    A format-2 bundle records one hash per group, so a change to the
    track profile no longer casts doubt on the star data. A format-1
    bundle carries only the combined hash, so any change condemns all
    of it.

    The assembly hash is recorded and never compared. Adding a field to
    the export, widening a floor, writing down a new aggregate — all of
    that changes how ``analyse_frame`` reads and none of it changes a
    number, which is the re-export this split exists to avoid.
    """
    current = measurement_fingerprints()
    stored = bundle.get("fingerprints")
    if not stored:
        was = bundle.get("measurement_fingerprint")
        now = measurement_fingerprint()
        return {} if was == now else {"everything (format 1)": (was, now)}
    return {name: (stored.get(name), current[name])
            for name in MEASUREMENT_PARTS
            if name in stored and stored[name] != current[name]}


def unmeasured_parts(bundle: dict) -> list[str]:
    """Measurement groups added since the bundle was written.

    Not stale: those numbers were never taken, so the bundle carries the
    defaults and replays exactly as it was judged. What it cannot do is
    speak for a rule that reads them -- the green glow, for a bundle from
    before the colour measure -- and the replay says so.
    """
    stored = bundle.get("fingerprints")
    if not stored:
        return []
    return [name for name in MEASUREMENT_PARTS if name not in stored]


def replay_settings(exported: Settings, as_exported: bool) -> Settings:
    """Today's decisions, the bundle's measurements.

    A bundle records every setting it was written with. The ones that
    describe how the pixels were measured have to be honoured -- the
    numbers came out of them and no replay can undo that. The rest are
    decisions, and taking those from the bundle too would make this tool
    useless for the thing it exists for: after moving a default in
    Settings, a replay would faithfully reproduce the old verdict and
    show no change at all. So the split is the same one MEASUREMENT_
    SETTINGS already draws, and ``--as-exported`` is there for when the
    point is to reproduce the original run rather than to judge a change.
    """
    if as_exported:
        return exported
    frozen = {name: getattr(exported, name) for name in MEASUREMENT_SETTINGS}
    return replace(DEFAULT_SETTINGS, **frozen)


def check_segment_floor(bundle: dict, settings: Settings,
                        thresholds: Thresholds) -> None:
    """Refuse a join bar below the one the bundle was exported at.

    The export keeps a wider net than the run used, so the bar can be
    lowered -- but only down to that net. Below it the fragments were
    never written down, and the replay would quietly report fewer tracks
    than the same settings would find on the frames themselves.
    """
    floor = bundle["session"].get("segment_floor")
    if not floor:
        return
    if settings.segment_min_ratio < floor["ratio"] - 1e-9:
        raise SystemExit(
            f"⚠️ segment_min_ratio={settings.segment_min_ratio} is below the "
            f"{floor['ratio']} this bundle kept its fragments down to. "
            f"Lower export_segment_min_ratio and re-export.")
    if thresholds.segment_min_length < floor["length"] - 1e-9:
        raise SystemExit(
            f"⚠️ segment_min_length reaches {thresholds.segment_min_length:.2f} "
            f"working px, below the {floor['length']:.2f} this bundle kept. "
            f"Raise export_segment_length_factor and re-export.")


def refit_frame(frame: FrameStats, settings: Settings, thresholds: Thresholds,
                refit: bool, revote: bool) -> bool:
    """Re-run what can be re-run on one frame. Returns whether the profile held.

    The track search is re-run from the exported fragments and the
    doubling vote from the exported blobs, so that the settings behind
    both actually bite -- the vote only when it has to, since it is the
    one expensive thing here. What cannot be re-run is
    ``measure_track_profile``: it needs the image. When the search lands
    on the line it landed on before -- the same fragments, which is what
    happens whenever the track settings have not moved -- the exported
    profile still describes it and is carried over. When it lands
    somewhere else, the width and brightness are gone with the pixels,
    and the caller is told so rather than shown a confident verdict on a
    default profile.
    """
    profile_held = True
    if revote and frame.stars.blobs:
        frame.doubling = find_star_doubling(frame.stars.blobs, settings,
                                            thresholds)
    if refit and frame.export_segments:
        before = frame.track
        segments = select_track_segments(frame.export_segments, settings,
                                         thresholds)
        found = find_track(segments, settings, thresholds)
        if found is not None and before is not None:
            if found.members == before.members:
                found = replace(found, profile=before.profile)
            else:
                profile_held = False
        elif found is not None:
            profile_held = False
        frame.track = found
    return profile_held


def replay_bundle(path: str, overrides: dict, refit: bool, stale_ok: bool,
                  labels: dict[str, dict] | None = None,
                  as_exported: bool = False) -> list[Replayed]:
    """Re-run one session's decisions from its bundle.

    A bundle carries the labels that existed when it was written. Pass
    ``labels`` to score against a labels.csv written since -- keyed on the
    frame's timestamp, so it finds its frames whatever the folder has
    been renamed to and whichever session the file came from.
    """
    bundle = read_cases(path)
    stale = stale_parts(bundle)
    if stale and not stale_ok:
        detail = "\n".join(f"     {name}: bundle {was}, this build {now}"
                           for name, (was, now) in sorted(stale.items()))
        raise SystemExit(
            f"⚠️ {path} was measured by different code:\n{detail}\n"
            f"   Those numbers no longer describe what this build would "
            f"measure, so scoring them would be scoring a version that no "
            f"longer exists.\n"
            f"   Re-export that session, or pass --stale-ok if you know the "
            f"change was cosmetic.")

    frames, reference, exported, geometry = cases_session(bundle)
    settings = replay_settings(exported, as_exported)
    settings = replace(settings, **overrides) if overrides else settings
    revote = any(getattr(settings, name) != getattr(exported, name)
                 for name in DOUBLING_SETTINGS)
    thresholds = Thresholds.derive(settings, geometry)
    check_segment_floor(bundle, settings, thresholds)

    # The target when the bundle carries one -- it is only a label, but
    # a list of changed frames reads better with the night named.
    session = bundle["session"].get("target") or bundle["session"]["id"]
    # Every track is refitted before any frame is judged: whether a track
    # recurs on the next frame is part of the verdict, as in the real run.
    held = [refit_frame(frame, settings, thresholds, refit, revote)
            for frame in frames]
    mark_recurring_tracks(frames, settings, thresholds)
    results = []
    for frame, case, profile_held in zip(frames, bundle["frames"], held):
        alerts, _ = classify_frame(frame, reference, settings, thresholds)
        categories = [alert.category for alert in alerts]
        fresher = (labels or {}).get(case["id"])
        results.append(Replayed(
            session=session,
            frame=case["id"],
            labels=tuple(fresher["labels"] if fresher else case.get("labels", ())),
            before=normalise(case.get("verdict", ())),
            after=normalise(categories),
            profile_lost=not profile_held))
    return results


def score(results: Sequence[Replayed], category: str) -> tuple[int, int, int, int]:
    """Caught, ticked, and the two kinds of unwanted flag.

    The two are worth very different amounts and pooling them into one
    precision figure is how a change gets talked into looking good.

    A flag on a frame the owner *labelled with something else* is the
    solid negative: they opened that frame and did not tick this. A flag
    on a frame with no label at all says almost nothing -- they may never
    have looked. On the first three sessions blanks were weak evidence of
    clean by the owner's own statement; on the six that came later that
    statement does not carry, and there is no ``Clean`` tick anywhere in
    the corpus to put it back. So precision is reported over judged
    frames only, and the blanks are shown beside it uncounted.
    """
    ticked = caught = other = blank = 0
    for row in results:
        labels = normalise(row.labels)
        detected = category in row.after
        if category in labels:
            ticked += 1
            caught += detected
        elif row.labels:
            other += detected
        else:
            blank += detected
    return caught, ticked, other, blank


def report(results: Sequence[Replayed], show_changes: int) -> None:
    labelled = sum(1 for row in results if row.labels)
    sessions = len({row.session for row in results})
    print(f"\n🧳 {sessions} session(s), {len(results)} frames, "
          f"{labelled} labelled\n")

    print(f"{'':>14} {'caught':>7} {'ticked':>7} {'recall':>7} "
          f"{'wrong':>7} {'precision':>10} {'unjudged':>10}")
    for category in SCORED:
        caught, ticked, other, blank = score(results, category)
        recall = f"{caught / ticked:.0%}" if ticked else "—"
        judged = caught + other
        precision = f"{caught / judged:.0%}" if judged else "—"
        print(f"{category:>14} {caught:>7} {ticked:>7} {recall:>7} "
              f"{other:>7} {precision:>10} {blank:>10}")
    print("   recall and precision count only the frames the owner judged.")
    print("   'unjudged' are flags on frames with no label at all — neither")
    print("   right nor wrong until somebody opens the preview.")

    lost = [row for row in results if row.profile_lost]
    if lost:
        print(f"\n⚠️ {len(lost)} frame(s) found a different line from the "
              f"exported one, so their width and brightness are unknown and "
              f"the meteor / aircraft branches could not be tried on them.")

    changed = [row for row in results if row.before != row.after]
    if not changed:
        print("\n✅ Every frame comes out exactly as the bundle recorded it.")
        return
    print(f"\n🔀 {len(changed)} frame(s) changed verdict against the bundle")
    for row in changed[:show_changes]:
        before = " ".join(row.before) or "clean"
        after = " ".join(row.after) or "clean"
        seen = f"   [owner saw: {' '.join(row.labels)}]" if row.labels else ""
        print(f"   {row.session} {row.frame}  {before} -> {after}{seen}")
    if len(changed) > show_changes:
        print(f"   ... and {len(changed) - show_changes} more "
              f"(--changes to see them)")


def default_bundles() -> list[str]:
    """Every bundle to hand: the ones beside this script, and any strays.

    The tool writes them beside the script so that they outlive the
    session folders they came from. The second pattern only catches
    bundles from before it did that, or ones copied in from elsewhere.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    found = glob.glob(os.path.join(here, "cases_*.json.gz"))
    found += glob.glob(os.path.join(here, "*", "cases_*.json.gz"))
    return sorted(dict.fromkeys(found))


# ======================================================================
#  Collecting the previews worth looking at
# ======================================================================

# What each category is decided on, as a ratio to the session. Used only
# to spread the sample over the range instead of taking the first N,
# which would otherwise all be the same borderline case.
# Which categories are worth a picture when the detector missed one.
# Trailing is left out on measurement: its misses are a labelling-severity
# effect -- the median ticked frame runs from 1.05x to 1.56x between
# sessions -- so an eye on a 1.13x frame settles nothing that the numbers
# have not already settled. Wrong flags are collected for every category
# regardless, being the only solid evidence of a mistake in the corpus.
MISSES_WORTH_A_LOOK = (CATEGORY_SATELLITES, CATEGORY_CLOUDS)

SORT_KEY = {
    CATEGORY_TRAILING:
        lambda case, ref: (case["stars"]["elongation"]
                           / max(ref["star_elongation"], 1e-9)),
    CATEGORY_SATELLITES:
        lambda case, ref: max((row[5] for row in case.get("segments", [])),
                              default=0.0),
    CATEGORY_CLOUDS:
        lambda case, ref: case["noise"] / max(ref["noise"], 1e-9),
}


def spread(items: list, count: int) -> list:
    """Up to ``count`` items spaced evenly through a sorted list.

    Taking the first N would hand back N versions of the same borderline
    frame. The interesting question is usually what the range looks like:
    the ones that nearly fired, the ones nothing was measured on, and
    something in between.
    """
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    step = (len(items) - 1) / (count - 1) if count > 1 else 0
    return [items[int(round(i * step))] for i in range(count)]


def control_frame(bundle: dict, verdicts: Sequence[Replayed]) -> dict | None:
    """The frame of this session closest to its own norm.

    Without one of these a preview of a disagreement is unreadable. Round
    stars, a normal sky and a normal star count are all session-relative,
    and the eye has no absolute scale for any of them -- crops compared
    across nights settle nothing, which is a mistake worth making only
    once.

    Chosen among the frames the owner did not tick and the detector did
    not flag, as the smallest distance to the reference over the four
    quantities the verdicts rest on.
    """
    reference = bundle["session"]["reference"]
    best, best_distance = None, None
    for case, verdict in zip(bundle["frames"], verdicts):
        if case.get("labels") or verdict.after or not case["stars"]["count"]:
            continue
        ratios = (case["stars"]["elongation"] / max(reference["star_elongation"], 1e-9),
                  case["stars"]["radius"] / max(reference["star_radius"], 1e-9),
                  case["noise"] / max(reference["noise"], 1e-9),
                  case["background"] / max(reference["background"], 1e-9))
        distance = sum(abs(r - 1.0) for r in ratios)
        if best_distance is None or distance < best_distance:
            best, best_distance = case, distance
    return best


def session_folder_for(target: str) -> str | None:
    """The sub folder a bundle came from, found by its own label rule."""
    for folder in find_sub_folders("."):
        if session_label(folder) == target:
            return folder
    return None


def find_source(folder: str, frame_id: str) -> tuple[str, str] | None:
    """The best picture of one frame: its preview, else the frame itself.

    A session does not always keep its ``JPG/`` folder, and skipping
    those frames would quietly collect nothing at all from a whole night.
    The FITS is always there, and subframe_labeller already knows how to
    stretch one for the eye -- so that renderer is reused rather than
    written a second time.
    """
    session = resolve(folder, configured_layout())
    previews = sorted(path for stem, path in session.previews.items()
                      if frame_id in stem)
    if previews:
        return previews[0], "jpg"
    frames = [path for path in session.frames
              if frame_id in os.path.basename(path)]
    return (frames[0], "fits") if frames else None


def make_preview(folder: str, frame_id: str, destination: str,
                 width: int) -> str | None:
    """Write one preview at the asked-for width. Returns what it came from.

    Falls back to copying the JPG whole when Pillow is absent, and gives
    up on a frame it can only reach through the FITS, since stretching one
    without Pillow is not possible here.
    """
    found = find_source(folder, frame_id)
    if found is None:
        return None
    source, kind = found
    try:
        from PIL import Image
    except ImportError:
        if kind == "fits":
            return None
        shutil.copy2(source, destination)
        return kind

    if kind == "jpg":
        image = Image.open(source)
    else:
        # A Seestar frame is a Bayer mosaic, so this carries a fine
        # checkerboard the JPG does not. Star shapes and trails read the
        # same, which is what these pictures are for.
        from subframe_labeller import read_fits
        image = read_fits(source)
    if image.width > width:
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.LANCZOS)
    image.convert("RGB").save(destination, "JPEG", quality=72, optimize=True)
    return kind


def collect_disagreements(paths: Sequence[str], results_by_path: dict,
                          out_dir: str, per_category: int, width: int,
                          all_categories: bool = False,
                          fits_budget: float = 0.0) -> None:
    """Gather the previews that answer what a bundle cannot.

    A bundle says a track was found and how wide it was. It cannot say
    whether the thing in the frame is a satellite, and no threshold sweep
    will ever tell you. That question needs the picture -- but only for
    the frames where the detector and the owner disagree, plus one
    control per session to read them against. On the corpus as it stands
    that is a few dozen files rather than several thousand.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    missing_folders = []
    fits_candidates = []

    for path in paths:
        bundle = read_cases(path)
        target = bundle["session"].get("target") or bundle["session"]["id"]
        folder = session_folder_for(target)
        if folder is None:
            missing_folders.append(target)
            continue
        reference = bundle["session"]["reference"]
        verdicts = results_by_path[path]
        by_id = {v.frame: v for v in verdicts}

        wanted = []
        for category in SCORED:
            missed, wrong = [], []
            for case in bundle["frames"]:
                verdict = by_id.get(case["id"])
                if verdict is None:
                    continue
                labelled = category in normalise(verdict.labels)
                flagged = category in verdict.after
                if labelled and not flagged:
                    missed.append(case)
                elif flagged and not labelled and verdict.labels:
                    wrong.append(case)
            key = SORT_KEY[category]
            missed.sort(key=lambda c: key(c, reference))
            if category in MISSES_WORTH_A_LOOK or all_categories:
                wanted += [(category, "missed", c)
                           for c in spread(missed, per_category)]
            # Every wrong flag is worth an eye: there are few, and they are
            # the only solid evidence of a mistake in the whole corpus.
            wanted += [(category, "wrong", c) for c in wrong]

        control = control_frame(bundle, verdicts)
        if control is not None:
            wanted.append(("control", "norm", control))

        if fits_budget:
            # One control per session goes with them: a full-resolution
            # frame is as unreadable without its own night's baseline as a
            # preview is.
            picks = []
            for category, kind, case in wanted:
                rank = FITS_PRIORITY.get(category)
                if kind == "missed" and rank is not None:
                    picks.append((rank(case, reference), category, case))
                elif kind == "norm":
                    picks.append((-1e9, "control", case))
            picks.sort(key=lambda row: row[0])
            here = []
            for _, category, case in picks:
                found = [path for path in
                         resolve(folder, configured_layout()).frames
                         if case['id'] in os.path.basename(path)]
                if found:
                    here.append(
                        (found[0], f"{category}_{target}_{case['id']}"
                                   f"{os.path.splitext(found[0])[1]}"))
            fits_candidates.append(here)

        for category, kind, case in wanted:
            name = f"{category}_{kind}_{target}_{case['id']}.jpg"
            source = make_preview(folder, case["id"],
                                  os.path.join(out_dir, name), width)
            if source is None:
                continue
            manifest.append({
                "file": name, "session": target, "frame": case["id"],
                "category": category, "kind": kind, "rendered_from": source,
                "labels": " ".join(case.get("labels", ())),
                "detected": " ".join(by_id[case["id"]].after),
                "elongation_ratio": round(case["stars"]["elongation"]
                                          / max(reference["star_elongation"], 1e-9), 3),
                "radius_ratio": round(case["stars"]["radius"]
                                      / max(reference["star_radius"], 1e-9), 3),
                "contrast": round(case["noise"] / max(reference["noise"], 1e-9), 3),
                "stars": case["stars"]["count"],
                "best_segment_ratio": round(
                    max((row[5] for row in case.get("segments", [])), default=0.0), 2),
                "note": case.get("note", ""),
            })

    if manifest:
        with open(os.path.join(out_dir, "manifest.csv"), "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    total = sum(os.path.getsize(os.path.join(out_dir, row["file"]))
                for row in manifest)
    print(f"\n🖼️  {len(manifest)} preview(s) collected into "
          f"'{out_dir}{os.sep}' ({total / 1e6:.1f} MB), with manifest.csv "
          f"giving the measurements beside each one")
    if missing_folders:
        print(f"   no session folder found for: {', '.join(missing_folders)} "
              f"— run this where the '<Target>_sub' folders are")
    if fits_budget:
        collect_fits(interleave(fits_candidates), out_dir, fits_budget)


# How invisible a missed frame is to the bundle, per category: the lower
# this comes out, the less there is in the export to work with and the
# more a FITS would buy. Satellites first because that blind spot is
# entirely a measurement question -- a trail can be plain to the eye and
# still leave no component above the detection floor.
FITS_PRIORITY = {
    CATEGORY_SATELLITES:
        lambda case, ref: max((row[5] for row in case.get("segments", [])),
                              default=0.0),
    CATEGORY_CLOUDS:
        lambda case, ref: -abs(case["noise"] / max(ref["noise"], 1e-9) - 1.0),
}


def interleave(per_session: Sequence[Sequence]) -> list:
    """Round-robin across sessions, keeping each one's own order.

    A budget of a few hundred megabytes is a couple of dozen frames at
    sixteen each. Taken session by session the first two nights would eat
    all of it, and a night contributes nothing useful without its own
    control, so every session gets its first pick before any gets a
    second.
    """
    flat = []
    for rank in range(max((len(rows) for rows in per_session), default=0)):
        for rows in per_session:
            if rank < len(rows):
                flat.append(rows[rank])
    return flat


def collect_fits(chosen: Sequence[tuple], out_dir: str, budget_mb: float) -> None:
    """Copy the frames themselves, within a budget, for development only.

    The FITS are otherwise never touched: nothing in this project moves
    or alters one, and the only code that removes one is
    ``delete_flagged_frames`` behind its own confirmation. This copies,
    and copies a handful -- the frames where the bundle holds nothing to
    reason about, so that a measurement can be developed against real
    pixels instead of guessed at.

    Ranked so the budget goes to the most invisible frames rather than
    the first ones found, and stated out loud before it runs, because
    each file is around sixteen megabytes.
    """
    if not chosen:
        return
    folder = os.path.join(out_dir, "fits")
    os.makedirs(folder, exist_ok=True)
    budget = budget_mb * 1e6
    copied, used = [], 0.0
    for source, name in chosen:
        size = os.path.getsize(source)
        if used + size > budget:
            continue
        shutil.copy2(source, os.path.join(folder, name))
        copied.append(name)
        used += size
    print(f"🔬 {len(copied)} FITS copied into '{folder}{os.sep}' "
          f"({used / 1e6:.0f} MB of the {budget_mb:.0f} MB budget). "
          f"The originals are untouched.")
    if len(copied) < len(chosen):
        print(f"   {len(chosen) - len(copied)} more were selected and left "
              f"out of the budget — raise --fits-budget to take them.")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score exported cases without the frames.",
        epilog="With no bundle given, every 'cases_*.json.gz' beside this "
               "script is replayed as one set.")
    parser.add_argument("bundles", nargs="*",
                        help="the case bundles to replay")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        dest="overrides",
                        help="override a setting for the replay; repeatable. "
                             "Only settings that judge, never ones that measure")
    parser.add_argument("--as-exported", action="store_true",
                        help="judge with the settings the bundle was written "
                             "with, instead of today's defaults — for "
                             "reproducing the original run rather than "
                             "scoring a change against it")
    parser.add_argument("--no-refit", action="store_true",
                        help="keep each frame's exported track instead of "
                             "searching for it again (faster, and the only "
                             "way to leave the track settings out of it)")
    parser.add_argument("--labels", action="append", default=[], metavar="CSV",
                        help="score against this labels.csv instead of the "
                             "labels stored in the bundle; repeatable, for "
                             "when a session was labelled after it was "
                             "exported")
    parser.add_argument("--stale-ok", action="store_true",
                        help="replay bundles measured by different code — "
                             "only when you know the change was cosmetic")
    parser.add_argument("--disagreements", metavar="DIR",
                        help="collect into DIR the previews where the detector "
                             "and the labels disagree, plus one control frame "
                             "per session — the questions a bundle cannot "
                             "answer, at a few dozen files instead of "
                             "thousands")
    parser.add_argument("--per-category", type=int, default=6, metavar="N",
                        help="how many missed frames per category and session "
                             "to collect, spread over the range (default 6). "
                             "Every wrong flag is collected regardless")
    parser.add_argument("--preview-width", type=int, default=1200, metavar="PX",
                        help="shrink collected previews to this width "
                             "(default 1200, about 300 KB each; needs Pillow, "
                             "copies whole without it)")
    parser.add_argument("--all-categories", action="store_true",
                        help="collect missed frames for every category, "
                             "trailing included — its misses are a labelling "
                             "threshold rather than a detection failure, so "
                             "they are left out by default")
    parser.add_argument("--fits-budget", type=float, default=0.0, metavar="MB",
                        help="also copy the FITS of the frames the bundle "
                             "holds nothing to reason about, up to this many "
                             "megabytes (each is about 16). For development: "
                             "it is the only thing here that copies a frame, "
                             "and it copies — nothing is moved or altered")
    parser.add_argument("--changes", type=int, default=20, metavar="N",
                        help="how many changed frames to list (default 20)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    paths = arguments.bundles or default_bundles()
    if not paths:
        print("⚠️ No case bundle found. Run subframe_inspector.py "
              "--export-cases on a session first.")
        return 1

    overrides = dict(parse_override(text) for text in arguments.overrides)
    if overrides:
        print("⚙️ " + ", ".join(f"{k}={v}" for k, v in overrides.items()))

    labels: dict[str, dict] = {}
    for csv_path in arguments.labels:
        labels.update(read_frame_labels(csv_path))
    if labels:
        print(f"🏷️ {len(labels)} labelled frame(s) read from "
              f"{len(arguments.labels)} file(s)")

    results: list[Replayed] = []
    by_path: dict[str, list[Replayed]] = {}
    for path in paths:
        found = replay_bundle(path, overrides, not arguments.no_refit,
                              arguments.stale_ok, labels,
                              arguments.as_exported)
        print(f"📦 {path}: {len(found)} frames")
        missing = unmeasured_parts(read_cases(path))
        if missing:
            print(f"   ℹ️ written before the {', '.join(missing)} measurement: "
                  f"the rules that read it cannot fire on this bundle.")
        by_path[path] = found
        results.extend(found)

    report(results, arguments.changes)
    if arguments.disagreements:
        collect_disagreements(paths, by_path, arguments.disagreements,
                              arguments.per_category, arguments.preview_width,
                              arguments.all_categories, arguments.fits_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
