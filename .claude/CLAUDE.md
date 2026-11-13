# Working on this project

Context an AI assistant needs before touching `subframe_inspector.py`.
Most of what follows was learned the hard way, on real data — treat it as
findings rather than preferences.

## What this is

A fast triage tool for Seestar light frames, run before stacking. It
sorts; it does not process. The owner is an amateur astrophotographer
working on Windows with WinPython.

**`DETECTION.md` is the other half of this file.** It owns the
specification behind every verdict: the hypothesis, the quantity
measured, the reference it is compared against, the number, and the
measurement that justifies the number. This file owns how to work here
and what has already been tried and rejected. They do not restate each
other -- when something moves, move it rather than copying it.

## Hard constraints

These are settled. Do not reopen them.

- **No registration, no astrometry, no plate solving.** `astroalign` was
  tried and removed: too heavy, and beside the point.
- **CPU only, multiprocessing, single pass.** No GPU, no heavy deps
  beyond numpy / scipy / astropy. Pillow is allowed for `--debug` only,
  imported lazily, and the run must survive its absence.
- **Speed is a first-order requirement**, not a nicety. The owner
  notices and comments on run time. Order of magnitude to stay inside:
  a couple of minutes for a 230-frame session on a laptop, well under
  one on a desktop.
- **FITS files are never moved or modified, except by `--delete-flagged`.**
  Everything else in the script works on the JPG previews, so that a
  wrong threshold costs nothing. That one option is the deliberate
  exception the owner asked for: it is off by default, it runs after the
  previews are filed, it lists every frame it covers, and it requires the
  word `delete` to be typed. Keep all four of those; do not add a second
  path that removes a `.fit`.
- **Nothing outside `REVIEW/` is ever deleted, and inside it only what
  this script wrote.** The rule that makes that safe: a file is removed
  only when a file of the same name still exists in `JPG/`, so anything
  irreplaceable -- the owner's own notes and picks, a preview an earlier
  `--mode move` took out of `JPG/` -- survives. Do not relax it into
  "delete the category folders".
- **Must work across the Seestar range** — S30, S30 Pro, S50, S50 Pro.
  Only S30 Pro data exists so far, so nothing model-specific may be
  hard-coded: the plate scale comes from `FOCALLEN` and `XPIXSZ` in the
  header and every sky-referenced threshold is written in arcseconds.
- **The main lens only.** The invariant above holds while the working
  image lands near 7.5 arcsec/px, and decimation only ever coarsens a
  scale -- so the wide-angle lens, at 55 arcsec/px, is outside it and
  `unsupported_scale` declines the session. Three of the "angular"
  thresholds turn out to be pixel quantities in disguise; that and a
  chance-alignment test are what supporting it would take. Measured, and
  written up under the wide-angle note in `DETECTION.md`. The owner's
  ruling for now: *"on peut l'écarter pour le moment"*.

## Approaches already tried and rejected

Re-proposing any of these wastes a cycle and makes the classification
worse.

| Approach | Why it failed |
| --- | --- |
| Morphological bridging (`binary_dilation`) between components | Joined the noise around bright stars into fake streaks; flagged ~100% of frames on a dense field. |
| Aircraft criterion as OR (wide **or** bright) | Sent nearly every satellite to the aircraft bin. Must stay AND. |
| Satellite criterion "total streak length > 150 px" | Total length grows mechanically with the number of trailed stars. |
| Trailing from *counting* elongated components | Trailed stars produce hundreds of them; so does a dense field at a low threshold. Superseded by measuring the stars themselves. |
| Star shape from a **thresholded bounding box** | No separation whatsoever — see below, this one is subtle. |
| Summing a band around a track to get its brightness profile | A bright star sitting on the trail lands straight in the profile. Four of the first six peaks measured that way were stars. Restricting the sum to the track's own component pixels helps but does not fix it — see the meteor section. |
| Flagging a track as "meteor-like" from an uneven brightness profile alone | Fired three times on real data and was wrong all three: satellites glint mid-passage, and stars merge into the trail. The profile now only chooses between two already-wide tracks. |
| Lowering the detection floor to reach the faintest trails | Swept over all three sessions, scored on both halves. It gains **nothing** and costs everything: p97.5 is identical to p98.5; p96 is worse on both counts (25 confirmed against 27, and ten flags on frames the owner passed over); p95 recovers the same 27 while flagging 71 clean frames, precision 28%. With three hundred specks in a frame, three of them always line up somewhere. The floor stays at 98.5 — this is settled, do not re-run it. |
| Requiring a track to beat what chance would put on the line | The statistic is sound and separates well — real trails beat chance by 240 to 420 times, the alignments a littered frame throws up by 19 to 38 — but it gains no recall, and rejecting the best candidate simply lets `find_track` fall back to a worse one: on `015313` it swapped a three-dash chance line for a two-dash one, verified against the preview. Worth revisiting only together with ranking candidates by quality instead of by span. |
| A Radon transform to integrate faint sub-threshold lines | The top-hat residual is not white noise — it carries column structure — so the significance is meaningless: control frames peaked at 11.9 sigma against 9.5 for a frame with a real trail, both near 90 degrees. Would need the row/column systematics removed first. **Now settled, see the next row: removing them does not help either.** |
| Recovering a sub-threshold trail from the top-hat residual, by any means | Settled negative, on `034122` in LBN 550 — a trail plainly visible in the preview whose fragments never reach the seed bar. Its excess over the surrounding residual is about **3 counts against a per-pixel scatter of 200**, so the mean over the 1600 samples along it carries a standard error of 5 and the trail sits at **0.8 sigma**. Measured three ways, each with a within-frame null of wrong angles through the same midpoint and the same search effort: counting pixels above a lowered bar (0.93x to 1.13x the null at p90/p95/p98), integrating along the line (-0.1 sigma), and the same after removing the row and column medians (-0.2 sigma). Two of seventeen wrong directions beat the real one. The eye sees it because it looks at the stretched original and recognises a coherent shape, not because the residual holds a recoverable signal. Reducing that scatter would mean debayering or matched filtering — outside the dependency and speed budget. |
| Judging track width by the geometric extent across the line | A few outlying pixels put the widest satellites at 7.6 working px, above the old aircraft threshold of 4. The flux-weighted width caps at 1.52 over the same tracks. |
| Star *brightness* as a haze signal | Barely separates: 0.92 of the session median on the ticked frames against 1.06 on the ones judged and passed. The detection floor is a percentile, so what survives a veil is always the bright end. It is the *number* of stars that collapses (0.44 against 0.96), not their light. |
| A high-percentile reference for the star count | Gains nothing -- 0.41 against 0.44 for the median. Unlike the sky background, whose median a veiled night really does drag with it, the star count median holds up. |
| A low-percentile reference for the sky background | Sound reasoning, poor trade: p25 buys ten frames for fifteen flags on frames the owner had judged, where `haze_background_rise` at 1.10 buys the same ten for none. |
| Lowering `haze_sigma` to reach a veiled night | 2.0 buys seven frames and costs fourteen. The frames it misses have a background *at* the night's own median -- a third of some of those nights is veiled -- so no bar on that statistic reaches them. |
| A cloud covering part of the frame, from the 4x4 tile grid | The signal is real -- the missed frames' emptiest tile runs 0.72 of its own session median against 0.95 on frames judged clean -- but it buys only 7 frames beyond the starless branch, for 2 wrong flags and 6.6 ms a frame always on. Most of what looked local was `_ratio` reading a starless frame as normal. The tiles are exported anyway, so re-testing this needs no FITS. |
| A hot pixel mask | Removed, having been measured. It found no sensor defects: two masks from the same session overlapped 3.4%, from the two halves of one night 0.8%, and from two sessions of the same S30 Pro **0.4%** -- a defect map is a property of the camera, so a real one would be identical. Over 30 frames, *zero* pixels landed in the top 0.3% of at least fifteen of them; the Seestar corrects its own defects on board. Removing it changed one verdict over 540 frames, and that one was the documented spurious satellite on `014411`. Do not build another one without first re-running that overlap test on the camera in question. |

## Trailing: what actually works, and the trap in it

**Flux-weighted second moments at full resolution separate trailing
cleanly. Bounding boxes at half resolution do not.** Both are "star
elongation", which is why the second was once written off as useless —
do not repeat that inference from the name.

Measured on the 230-frame LBN 550 session, median over ~280 stars of
sqrt(major/minor) from flux-weighted moments, against the session's
sharpest quarter:

- every one of the 14 frames the owner labelled trailed: **1.38x or above**
- the median frame: 1.04x
- the sharpest quarter: 1.00x by construction

The same quantity from a thresholded bounding box on the `::2` image
takes about eight distinct values over the whole session and separates
nothing. A star spans two or three pixels there; there is nothing between
the steps. Any shape threshold measured that way is untunable.

The two populations are still a **continuum** — wind smears
progressively, there is no gap in the distribution — so
`trailing_elongation_ratio` is a user-facing severity dial, exposed as
`--trailing`, not a constant.

### Field shift is not a fault

Earlier versions reported it as its own category. The owner's ruling:
a field that moved but kept its stars round costs nothing — stacking
registers it away and the edges get cropped regardless. On one 249-frame
session that category alone was 47 of 59 alerts, pure noise.

The measurement stays (it is nearly free and explains a trailed frame)
but it is a CSV column, not a verdict. Do not wire it back into one.

## Straight tracks: satellites, meteors, aircraft

The Seestar's field is a couple of degrees across, so anything crossing
it flies dead straight. The detector looks for **one line**, and
reassembles it when the threshold breaks it into fragments.

Five findings hold this together, all measured against previews checked
one by one:

1. **Aspect ratio is the discriminant, not length.** Longest fragment of
   a real trail: 10 to 184 times as long as wide. A smeared star: 6 at
   most. The short fat blobs that used to be reported on dense fields —
   merged star pairs — sit between 2.7 and 4.3. Length does not separate
   at all: every false positive measured exactly at the detection floor.
2. **Two bars, not one.** Only a fragment at ratio ≥ 6 may *start* a
   line; anything at ratio ≥ 4 may *join* one. A faint trail arrives as a
   dotted line of specks barely longer than wide, and at the strict bar
   only two or three survive — that is how the old version lost three
   real satellites. A merged star pair can never start a line.
3. **Judge a line by how straight it is, never by how many pieces it
   has.** Trailed stars are all parallel, so a line along their common
   direction gathers a few by luck. Counting fragments does not separate
   those from a real trail — an "at least six fragments" rule sat in a
   gap it had no data for and missed a three-fragment satellite the owner
   spotted by eye (`051837`, NGC 6910 IRCUT), while letting through a
   ten-dash chance line at 60% coverage. The RMS residual about the
   fitted line does separate them, with a factor of three to spare:

   | | RMS residual, working px |
   | --- | --- |
   | nine confirmed trails | 0.07 – 0.30 |
   | chance lines on trailed frames | 1.02, 5.29, 7.12 |

   So: three fragments or more are accepted on the residual alone; one or
   two, which fit any line exactly, need coverage instead.
4. **The trailed-frame gate still earns its keep, but it counts pieces.**
   On a frame judged trailed or soft, a track below the coverage bar is
   dropped *unless* it has enough fragments. Removing the gate entirely
   was tried once the residual test was in: it let through three more
   tracks, and all three, checked against the previews, were triples of
   star dashes. But a flat gate cost a confirmed satellite (`040434`)
   the moment the trailing threshold moved and its frame became "soft".
   The residual does not separate those cases -- the chance triples
   measured 0.03 to 0.47 against the real trail's 0.15 -- and the
   fragment count does: three against twenty-two. Hence
   `track_fragments_if_soft`.
5. **The seed bar was guesswork and is now measured.** It sat at 6 from
   before the residual test existed. Merged star pairs, the thing it is
   there to stop, reach 4.3 at the very most, so it now sits ten percent
   clear of them at 4.75 -- which recovered a real trail whose thinnest
   piece was 4.97. Sweeping it from 6 down to 4.5 across all three
   sessions changed exactly one frame, and that frame was real.

The line direction is refit through the fragments it gathered: a short
fragment fixes its own direction only to a few degrees, and a few degrees
is a long way off at the other end of the frame.

**Known miss, and it is a hard one.** `044551` in NGC 6910 IRCUT holds a
faint real trail that never gets reported. The frame is badly trailed, so
its star dashes are themselves elongated enough to seed lines, while no
fragment of the real trail is thin enough to seed one. Lowering the seed
bar to reach it reopens the door to merged star pairs. The frame is
flagged Trailing regardless, so it still reaches review.

## Doubling: a knock is not a trail

A jolt during the exposure prints the whole star field **twice**, both
images sharp, offset by however far the knock moved the mount. The
owner's description is the detector's own model: *"le nombre d'étoiles
double et les patterns formés par les étoiles sont doublés."* Every star
has a partner at one shared offset, and the pairwise vote finds it.

**The search range was the bug, and it hid a real one for a long time.**
`doubling_max_distance` was an absolute 300 arcsec, justified in a
comment by "beyond that it is a trail, already covered elsewhere". That
reasoning does not hold: a trail needs continuous motion during the
exposure, while a knock is instantaneous and can be as large as it likes.
`020219` in NGC 6910 LP is doubled at **192 px -- 719 arcsec, 0.2
degrees -- with 74% of the stars agreeing**, three times the strength of
the reference case, and it was reported as clean.

It is worth knowing how invisible that frame is to every *other* test,
because the next such case will look the same. Its stars are neither
oval (elongation 1.04x the session) nor swollen past the bar (1.20x) nor
leaning coherently (0.5); resolved-pair statistics at 2x2, 3x3 and 4x4
tiles are indistinguishable from a clean control; there is no tangential
rotation pattern; the bright stars measure like the control's; and at 4x
magnification the stars look single and round. Nothing is wrong with the
shape of a doubled star. There are simply two of it.

**Verify a claimed doubling by autocorrelation, not by eye.** A dense
field duplicated 200 px over is not something you can read off a preview,
and squinting at one wastes an hour. Shift the top-hat crop by the
offset the detector claims and correlate it with the original, then do
the same for twenty other directions of the same length: a real doubling
spikes at 16x the level elsewhere, where the strongest wrong direction
reaches 2.4x. It is independent of the pair vote, so it is worth having.
It confirmed both knocks in NGC 6910 LP before the owner did.

**A doubled frame may also raise a spurious satellite, and that is
allowed.** On `014411` the track is the thin bridge between a bright star
and its own second image, 25 px long. The owner was asked and ruled
against suppressing it: a rule that discounted tracks near a doubled
bright star would cost real satellites, while a stray label on a frame
already condemned as Trailing costs nothing -- *"de toute façon cette
image devra être supprimée."* Do not add that suppression.

The bound is now **geometric**: a share of the shorter side of the
measured crop. Past that, too few stars keep a partner inside the crop
for the fraction test to mean anything -- a limit that belongs to the
measurement rather than to a mount or a night. Do not turn it back into
an angle, and do not tune it on the sessions in this repository: the
owner's point, and it is a fair one, is that his data holds the knocks he
happened to get, not the extreme ones.

## Meteors versus satellites

**Meteors are what the owner actually wants to catch.**

The owner's criterion, stated directly: *a satellite leaves a thin,
fairly regular trace; a meteor has a head and a tail, some glow around
its passage, and the head is far brighter than the tail.* Their second
point, after the first attempt failed: *a meteor's trace should also be
wider and brighter.*

### What was tried and failed

The **shape of the brightness profile on its own**. It was shipped as a
"meteor-like" note, fired three times, and was wrong all three. The owner
checked the two strongest:

- `025557` (halves 8.1:1) — a satellite. Thin at both ends, brighter in
  the middle. A glint in mid-passage, not a tail.
- `002717` (6.1x spike at 10% along) — a satellite, uniform all the way,
  with a star sitting on its path.

That second one is a limit of the measurement, not of the threshold:
restricting the sum to the track's own pixels does **not** keep stars
out, because a star on the trail merges into the same connected
component. (Summing a band around the line was worse: four of the first
six peaks that way were stars.)

### What works: width has a measured ceiling

A satellite trail is a straight line convolved with the seeing disc and
nothing else, so its width is set by the optics rather than by how bright
it is. Measured over the 26 confirmed satellites:

| | min | median | max |
| --- | ---: | ---: | ---: |
| flux-weighted width (working px) | 0.43 | 1.06 | **1.52** |
| peak brightness (x frame noise) | 1.3 | 9.7 | 89.6 |

**69x in brightness, and the width never passes 1.52.** That is a real
ceiling with a physical reason under it, and `wide_track_width` (18
arcsec ≈ 2.4 working px) sits 60% above it. Use the **flux-weighted**
width for this and never the geometric one: a few outlying pixels push
the widest satellites to 7.6 working px on that measure, straight through
the old aircraft threshold, which only the brightness AND was saving.

Do not read the width *ratio to the stars* instead — it looks like the
better-normalised quantity but it spans 0.62 to 2.30 over the same 26
tracks and correlates 0.6 with brightness. The absolute width is tighter.

### What is still guesswork

Which kind of wide thing it is. Aircraft take the evenly-lit branch,
meteors the tapered one. No aircraft has been seen in this data. **Two
meteors now have** -- `M 45`, `20261105-010712` and `20261103-045310`,
confirmed by the owner -- and the detector filed both as satellites:
they measure 1.84 and 2.36 working px, under `wide_track_width`. The
same session holds a slow satellite that measures 3.16 on one frame and
was filed as a meteor. So width does not separate them there, and two
examples are no reason to move the threshold. What the owner actually
recognised them by was a **green glow along one side of the trail**, and
a single-frame track; a satellite recurring at the same tilt on
consecutive frames is the cue that would have cleared the false one.
Details and numbers in `DETECTION.md`, under "First confirmed meteors".
Do not tune `wide_track_width` on these two.
The owner reports the old aircraft rule firing correctly on a sub of
their own that is not in this repository — so if you touch the width
measure, ask them to re-run that frame. That precedent also settles the
policy question: a physically motivated rule with a measured margin under
it is worth shipping unvalidated, because the whole tool only ever copies
JPGs and a misfiled preview costs nothing.

One idea already ruled out: *"a satellite crosses the whole frame, a
meteor starts and ends inside it."* The field is small enough that a
meteor crosses it completely, so `touches_edge` separates nothing here.
The owner said so directly.

## Where the run time actually goes

Measured, so nobody has to guess again. Shares of one frame's work,
which travel between machines far better than seconds do:

| | |
| --- | --- |
| two morphological openings, 2 Mpx each | **~45%** |
| the two `label` passes and the moments | ~16% |
| reading and rescaling the FITS | ~9% |
| everything else | the remainder |

Four things follow, and each was tried before being written down.

- **scipy's opening is already optimal.** Its cost is flat in the kernel
  size (3, 5, 9, 15 and 25 all measure the same), so it is running the
  van Herk algorithm. Rewriting it separably, with preallocated output,
  or with a transpose for the vertical pass all give bit-identical
  results and no speed-up. Don't.
- **Parallelism is not the lever it looks like.** Morphological filters
  over multi-megapixel arrays are memory bound, so the pool stops
  scaling well before it runs out of cores: measured on one machine, the
  compute plateaued at barely over 2x and got *worse* with a worker per
  logical core. The full pipeline does better than that only because
  disk I/O overlaps with compute, which is why the pool is left at the
  default worker count rather than pinned. Re-measure before capping it
  on any particular machine; the answer will not be the same on two.

  It can also cost *correctness*, which is the reason `--workers` exists.
  Every worker holds a full-resolution frame and several working copies,
  so a dozen at once can ask for two gigabytes; on a machine with other
  things running, `load_frame` starts raising `MemoryError` and frames
  drop out of the session. Measured on the 61-frame IRCUT session with
  11 GB of the 16 in use elsewhere: the default pool lost 11, then 17,
  then 42 frames over three consecutive runs, while `--workers 4`
  completed every time **and ran slightly faster**. Lowering it changes
  no measurement.
- **Measure twice before believing a slowdown.** On one laptop the same
  run drifted from 20 s to 40 s over an hour, with no code change at
  all: power profile and clock throttling. Timings taken minutes apart
  on a busy machine are not comparable, and a "regression" measured that
  way is usually the machine.
- **`np.minimum.at` is not the bottleneck it used to be.** Replacing the
  four extent reductions in `measure_components` with sort + `reduceat`
  made it 16x *slower* (0.41 ms against 6.6 ms). numpy has optimised
  `ufunc.at`; leave it alone.

So the only remaining lever is processing fewer pixels, and that means
`sharpness_centre_fraction` or `working_scale` -- both of which change
what is measured.

**`sharpness_centre_fraction` was measured and refused.** Shrinking the
full-resolution window from 0.50 to 0.40, 0.35 and 0.30 across all three
sessions:

| | 0.40 | 0.35 | 0.30 |
| --- | ---: | ---: | ---: |
| LBN 550 (230) | +7%, **7 verdicts** | +7%, **4** | +31%, **5** |
| NGC 6910 LP (249) | +17%, **2** | +22%, **2** | +26%, **4** |
| NGC 6910 IRCUT (61) | +9%, 0 | +7%, 0 | +14%, 0 |

The frames that flip include `022119` and `030115`, two of the three
labelled satellites, plus `040434` which was confirmed by eye. The
coupling is not obvious and worth knowing: the window only measures star
*shapes*, but the trailing verdict gates the track test -- on a frame
judged trailed, only a continuously lit track counts -- so a nudge to the
elongation silently changes which satellites are reported. A tenth of the
run time is not worth that.

`working_scale` has not been measured yet. Measure it the same way.

## The one trap left

**Two pixel scales.** The pipeline works on a decimated image, but one
pass runs at full resolution over the centre of the frame. This used to
be the main source of bugs — a missed scaling on the doubling guard once
broke it silently.

It is now structural rather than a matter of care: `Settings` holds
arcseconds, `Geometry` reads the plate scale from the header, and
`Thresholds.derive` converts once, into fields that each state which
image they belong to. **Nothing else in the file converts anything.** Keep
it that way; do not reintroduce a bare pixel constant.

The decimation factor itself is chosen from the plate scale so the
working image lands near a constant 7.5 arcsec/px whatever the model —
which is what lets the few genuinely pixel-valued settings
(`opening_size`, `bright_percentile`) stay model-independent.

## Ground truth available

Session `LBN 550_sub` (230 frames, 2026-08-30). Frames labelled by the
owner:

- **Trailing**: `000305` (doubling), `003133`, `004055`, `004611`,
  `004927`, `005443`, `005959`, `010516`, `011138`, `011800`, `013044`,
  `014224`, `022232`, `035155` (trailing + doubling + haze)
- **Satellites**: `022119`, `030115`, `031350`

Current state over **nine** sessions, 2484 frames, 648 judged by eye --
read it from the case bundles with `python replay_cases.py`:

| | caught | ticked | recall | wrong | precision | unjudged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trailing | 230 | 344 | 67% | 6 | 97% | 36 |
| Satellites | 58 | 97 | 60% | 2 | 97% | 0 |
| Clouds_Haze | 178 | 222 | 80% | 11 | 94% | 20 |

Precision that high is the thing to protect. A change that trades several
points of it for one more frame is a bad change, however good the recall
column looks.

**The recall figures are far below what the first three sessions
suggested, and nothing regressed to cause it.** On `LBN 550` the numbers
are still exactly the documented ones and the code is unchanged; the six
sessions added later are simply harder in one specific way. A third of
some of those nights is veiled and a fifth of another is trailed, and
*every* threshold here is relative to the night's own normal -- so
"unusually hazy for this night" stops meaning anything when the night
itself is hazy. Where the reference could be moved to fix that it was
measured and mostly could not (see the table above); where a second,
independent signature existed -- the stars a veil dims -- it was worth 31
points of cloud recall.

The twelve trailing misses are all in NGC 6910 LP and all borderline: the
owner is unsure about them and expects the stacker to absorb them. Do not
chase them. The three satellite misses (`003335`, `034122`, `035155` in
LBN) have **no fragment thin enough to seed a line at all** -- not one
component reaches the seed bar -- so they are below the detection floor
rather than behind a badly-set threshold. The two haze misses have a
background at or below the session median: nothing measurable is there.

The older hand-picked list still passes too: **14/14 trailing and 3/3
satellites**. Re-check both sets after any change to detection. Note `000305` is caught by the doubling
detector, not by elongation — it sits at 1.26x, below the 1.30 dial,
because a doubled star stays round. Do not "fix" that by lowering the
dial.

**Both label lists are incomplete.** They are frames the owner happened
to note, not an exhaustive audit. Fifteen further detections were checked
against their previews during this work and every one was real —
including trailing on `003650`, and satellites on `002503`, `033908`,
`033347`, `040434`, `024331`, `020739`, `020840`, `025557`. So a
detection outside the labelled set is not a false positive by default;
open the preview.

Two more sessions are available and unlabelled: `NGC 6910_sub` (249
frames) and `NGC 6910_IRCUT_sub` (61), both dense Milky Way fields, which
is where the old detector's false positives lived.

### `<session>/labels.csv` — the growing set

`subframe_labeller.py` writes one there per session, and it supersedes the
lists above wherever the two overlap: those were noted in passing, this
one is a frame gone through deliberately. Columns: `frame`, `labels`
(what the owner saw, space-separated), `note` (free text), `detected`
(what the detector said at the time it was labelled), `updated`.

Read it before tuning anything, and mind what it does **not** say. The
owner's own words: *"il est possible que je n'ai pas tout vu -- considère
que d'autres choses peuvent apparaître que je n'ai pas coché, mais ce que
j'ai coché je l'ai vu."*

**What is ticked is real** -- the owner is unequivocal about that, and
about satellites in particular: *"si je l'ai mis en label c'est que c'est
le cas."*

**On these three sessions, and only these, an absent frame is weak
negative evidence.** Their words: *"toutes les trames sans rien me
paraissent clean mais j'ai pu passer à côté de quelques unes."* So for
this data both halves are measurable, precision with a pinch of salt:
count a flag on an unlabelled frame as probably wrong, but open the
preview before believing it. Earlier, under tighter thresholds, fifteen
such "extras" were checked one by one and every one turned out real.

**Do not carry that assumption to a new session.** Also their words:
*"ma phrase c'est uniquement sur les données actuelles, sur de futurs sub
je n'aurai peut-être pas le temps de tout labéliser."* A `labels.csv`
that arrives later is positive evidence and nothing more, unless the
owner says otherwise for that session. If you need precision on new data,
ask -- do not infer it from the blanks.

**There is no `Clean` tick anywhere in the corpus.** The vocabulary
actually used is `Trailing` 344, `Clouds_Haze` 222, `Satellites` 96,
`Aircraft` 1 -- so a frame is either ticked for a fault or blank, and
there is no solid negative at all. Precision is therefore reported over
*judged* frames only: a flag on a frame ticked for something else is real
evidence of a mistake, a flag on an unlabelled frame is evidence of
nothing. `replay_cases.py` keeps the two columns apart and they must
never be pooled -- pooling them once made a change look 15 points worse
than it was. If precision on new data matters, the cheapest fix by far is
for the owner to tick thirty frames `Clean` per session.

That asymmetry -- ticks certain, blanks probable -- is what the scoring
should reflect. It immediately paid for itself: a threshold change that
looked like +8 recall turned out to be +1 real frame and +13 flags on
frames the owner had looked at and passed over.

The vocabulary is the detector's own plus `Clean`. A `Doubling` button
existed briefly and was removed on the owner's ruling -- *"c'est du
trailing"* -- having never been ticked once in 648 labelled frames. Both
readers keep the alias anyway (`LABEL_ALIASES` in `replay_cases.py`,
`LABEL_TO_DETECTOR` in `subframe_labeller.py`), so a file written while it
existed still counts as agreeing with a `Trailing` verdict. Do not add
the button back without asking: the distinction is real, but it belongs
to `doubling_px` in the CSV, not to the labels.

**Trailing recall measures the labelling, not the detector -- do not
chase it.** Per session, the median `elongation_ratio` of the frames
ticked Trailing runs 1.05, 1.13, 1.18, 1.23, 1.35, 1.45, 1.50, 1.56, and
the recall on them runs 0%, 23%, 33%, 66%, 100%, 81%, 100%, 100%. Two of
those sessions are the same target one night apart. The severity being
ticked moves by a factor of ten between nights, so no dial value
satisfies both ends and the pooled figure says nothing about detection.

Two explanations for the 43 unreachable ticks were tested and refuted:
corner trailing (a 3x3 full-resolution grid puts the corners at 0.94-1.00
of the centre, and unticked controls at 0.91-0.96) and a globally soft
night (the absolute reference spans only 1.21-1.34 over nine sessions
against a ticked share of 1%-21%). Neither is worth re-running.

**The dial has no correct value, and this is now measured.** Over 344
ticked frames across nine sessions, the 139 the detector misses are
spread almost perfectly flat between 1.00x and 1.30x -- 24, 24, 22, 24,
20, 25 per 0.05 band. There is no cluster just under the bar and no gap
to sit in, so every step down buys a constant handful of frames for a
growing number of flags. What fixed the default at **1.25** is two of the
owner's own rulings on frames they went and opened: `023406` at 1.26x is
real trailing, `034739` at 1.22x has sound stars. That brackets it, and
nothing else does.

**Mild trailing is a judgement call, and the owner says so.** On the
softer frames: *"c'est un peu visible mais c'est pas non plus excessif,
pas facile à l'oeil de trouver le juste milieu, et probablement cela est
géré par l'outil de stacking."* Do not chase the last few borderline
ticks by dragging the threshold down -- the owner does not consider them
lost frames. Catch the clear cases and leave `--trailing` as the way to
go stricter on a particular night.

It is not under `REVIEW/`, deliberately: every run of the detector
empties that folder.

## Case bundles: replaying a session without its frames

`--export-cases` writes `cases_<target>_<date>-<time>.json.gz` **beside
the script**, not in the session folder: every number the run measured,
none of the pixels. Three megabytes against the four gigabytes of FITS
behind them. A bundle is meant to outlive its frames, which is exactly
why it must not live in the folder that gets archived or deleted once
the night has been stacked -- and beside the script they collect in one
place, which is what lets a replay score every night ever exported
without being told where they are. `replay_cases.py` reads those bundles
back, re-runs the classification and scores it against the labels:

```
python subframe_inspector.py "LBN 550_sub" --mode report --export-cases
python replay_cases.py --set trailing_elongation_ratio=1.25
```

The whole corpus -- 540 frames over three sessions -- replays in under
four seconds, so a threshold sweep is a loop rather than an afternoon.
More to the point, it keeps working once the frames are gone: the owner
cannot keep every night on disk, and a session that is interesting for
two of its frames still costs four gigabytes to hold on to.

**Why bundles from different targets score as one set.** Everything a
verdict rests on is either a ratio to the session's own yardstick or an
absolute in arcseconds -- and that yardstick, the sharpest quarter and
the median sky, is computed over the whole night and travels *inside*
the bundle. Two consequences:

- a bundle holding only the interesting frames classifies them exactly
  as the full run did, so there is no need to keep the clean majority in
  order to keep the cases;
- nothing a verdict touches names the target. A frame is called by the
  moment it was shot, a session by the moment its first frame was shot,
  so renaming the sub folder cannot give one night two identities.

The target does appear, in the file name and in `session.target`, and
that is deliberate: with a dozen bundles side by side the owner has to be
able to tell which night is which. It is a label. Nothing reads it, and
nothing scored may start to.

**What replays and what does not.** Everything downstream of
`analyse_frame`: the whole of `classify_frame`, `find_track` from the
exported fragments, `find_star_doubling` from the exported blobs,
`measure_field_shift` from the exported landmarks. Nothing upstream of
it: `working_scale`, `opening_size`, the 98.5 floor, the star selection,
`sharpness_centre_fraction`. Those are listed in `MEASUREMENT_SETTINGS`
and `--set` refuses them, rather than accepting an override that would
silently do nothing.

One thing genuinely cannot be replayed: `measure_track_profile` needs the
image. When a refit lands on the same fragments -- which it does whenever
the track settings have not moved -- the exported profile still describes
that line and is carried over. When it lands elsewhere, the width and
brightness went with the pixels, and the replay reports how many frames
that happened to instead of judging them on a default profile. So a sweep
of the track settings is honest about recall and blind to the
meteor/aircraft split.

**Two guards, and both matter more than they look.**

`measurement_fingerprint()` hashes the source of the ten functions that
turn pixels into numbers. A bundle records it and the replay refuses to
score one measured by different code. Comments count, so a cosmetic edit
invalidates the corpus and `--stale-ok` is the way past it -- that is the
right direction to err in. This is the `calibrate_doubling.py` failure
with a lock on it: a stale corpus scored as fresh would validate a
version that no longer exists, and nothing would look wrong.

The export keeps fragments below the bars the run itself uses
(`export_segment_min_ratio`, `export_segment_length_factor`), so the join
bar can later be lowered as well as raised. Below that floor the
fragments were never written down and a replay would quietly find fewer
tracks than the frames would, so `check_segment_floor` refuses it.

**One fingerprint per measurement group, and the assembly is excluded.**
`MEASUREMENT_PARTS` splits the pixel-touching functions into `frame`,
`stars` and `profile`; a bundle records a hash of each, plus one of
`analyse_frame` under `assembly` which is recorded and never compared.

That exclusion is the point. `analyse_frame` decides *which* primitives
to call and *what* to write down -- so adding an exported field, widening
an export floor or recording a new aggregate rewrites it while changing
no number. Under the old single hash every such edit condemned every
bundle, which meant re-running the detector over every night kept on
disk. Now only a real change to a primitive does, and only for the group
it belongs to.

The rule that keeps this honest: **every operation on a pixel lives in
one of the functions listed in `MEASUREMENT_PARTS`.** `working_image` and
`bright_structure` were extracted from `analyse_frame` for exactly that
reason. Put a `grey_opening` or a percentile back inline and the guard
silently stops meaning anything.

**Bundles carry resolved primitives, not only aggregates.** `spatial_summary`
writes a 4x4 tile grid, sky and component quantiles, and a 400-star map
per frame -- none of it read by any verdict, all of it nearly free, and it
is what lets the next spatial idea be scored without the FITS. The
measurement that forced it: the 120 landmarks already exported give 13
stars per tile on a 3x3 grid, and at 13 the counting noise alone produces
a density hole of 0.62 +/- 0.15, wider than the gap between cloudy and
clear frames. Anything spatial measured on those landmarks is noise.

**A replay judges with today's defaults, not the bundle's.** A bundle
records every setting it was written with; the measurement ones are
honoured, because the numbers came out of them, and the decision ones are
taken from `Settings` as it stands now. Otherwise the tool would be
useless for the one thing it exists for: after moving a default, a replay
would faithfully reproduce the old verdict and report no change at all.
`--as-exported` is there for reproducing the original run instead.

**The pictures a bundle cannot replace, and how few are needed.**
`replay_cases.py --disagreements DIR` collects the previews of the frames
where the detector and the labels disagree, plus **one control frame per
session** -- the frame closest to that night's own norm among those
neither ticked nor flagged. The control is not optional: round stars, a
normal sky and a normal star count are all session-relative, and crops
compared across nights settle nothing. Rendered at 1200 px they run
about 290 KB each, roughly seventy files for nine sessions.

Two defaults set on measurement rather than taste. Trailing misses are
left out (`--all-categories` takes them): that gap is a labelling
severity effect, so an eye on a 1.13x frame settles nothing the numbers
have not. Every *wrong* flag is collected whatever its category, being
the only solid negative evidence in the corpus.

It does not need `JPG/`. A session does not always keep its previews, so
a frame without one is rendered from the FITS through
``subframe_labeller.read_fits`` -- the same stretch, not a second copy of it.
What comes out is a Bayer mosaic and carries a checkerboard the JPG does
not; star shapes and trails read the same.

`--fits-budget MB` also copies the frames themselves, ranked by how
little the bundle holds on them -- a satellite miss with no fragment
above the floor first, since that blind spot is purely a measurement
question. It **copies**; nothing in this project moves or alters a FITS,
and `delete_flagged_frames` remains the only code that removes one.
Sixteen megabytes each, so the budget is in megabytes rather than files
and it says what it took.

**Re-labelling does not mean re-exporting.** A bundle carries the labels
that existed when it was written. When a session is labelled afterwards,
pass `replay_cases.py --labels <session>/labels.csv` instead of re-running
on the FITS; frames are matched on their timestamp, so it works whatever
the folder is called and whichever session the file came from.

**One cost worth knowing.** The doubling vote is 175 ms a frame -- nothing
in the tool, where it is spread over a pool, but a minute of an otherwise
instant corpus replay. It is therefore only re-run when one of the
`doubling_*` settings has actually moved.

**The validation to repeat after any change to the export.** Run a
session twice, with `--export-cases` and without: the two `report.csv`
must be byte-identical, because the export widens the segment net and a
mistake there would change what `find_track` sees. Then replay the
bundle: every frame must come out exactly as recorded. Both held on all
three sessions when this was written, and the replayed corpus reproduces
the score table above.

## The CSV report is a diagnostic tool

`REVIEW/report.csv` holds one row per frame, flagged or not, with every
measurement behind each verdict. Read it before writing an evaluation
script: it often answers the question already, and it shows whether a
threshold sits in a gap between two populations or in the middle of one.

It once exposed a blind spot that the labels have now closed: stars can
bloat without being smeared in any direction — focus drift or seeing —
and the oval test cannot see it. On the frames the owner ticked for it,
`elongation_ratio` sat near the session median while `radius_ratio` ran
1.36 to 1.54. `trailing_radius_ratio` tests that in its own right and the
report names which of the two it found. `coherence` is still only a
column: bloated-but-incoherent would be the sharper diagnosis, and
nobody has needed it yet.

## Debug mode

`--debug` draws on the previews as they are filed into `REVIEW/`, rather
than into a folder of its own: the track and its fragments, the region
the stars were measured in, a tick along each star's own axis scaled by
its elongation, the doubling offset, and the verdict in words. It is the fastest way to
settle "is this a false positive?" — and the fastest way for you to check
your own work, since the previews line up with the FITS pixel for pixel.

## How to work here

- **Measure before proposing.** Several plausible hypotheses were killed
  by evidence, and one recorded failure ("elongation does not separate")
  turned out to be true only of the way it had been measured. Write a
  throwaway evaluation script, compare labelled frames against a control
  group, look at whether the distributions actually separate — then
  decide.
- **Look at the previews.** They are JPGs whose pixels match the FITS one
  to one, so a crop around a detection settles in seconds what a
  distribution argues about for an hour. Most of the thresholds here were
  fixed that way.
- **Score a threshold against the case bundles before touching the
  FITS.** `python replay_cases.py --set <name>=<value>` re-runs every
  decision over every exported session in under four seconds and prints
  both columns, which is the measurement this file keeps asking for. Only
  a change to how the pixels are *measured* needs a full run.
- **Check that a run analysed every frame before believing any number
  off it.** A shortfall now says so in the summary (`🚨 ANALYSED ONLY`),
  because it used to be invisible: the report counted the files on disk
  and called everything unflagged clean, so a session short of eleven
  frames looked complete. That cost a wrong answer during this work -- an
  A/B of two settings showed eleven "changed verdicts" that were nothing
  but frames missing from one of the two runs. If you see that line, the
  session is partial and every rate computed from it is wrong.
- **Run the tests, then prove behaviour is unchanged.**
  `python -m unittest test_subframe_inspector` takes about a second and
  covers the machinery that fails silently: the arcsecond-to-pixel
  conversions, the component and star measurements, the track detector,
  the case round-trip, and every path that removes a file. It says
  nothing about whether a threshold sits in the right place -- for that,
  capture a per-frame, per-category reference over the three sessions
  first, and compare after.
- **Do not duplicate pipeline logic in tools.** `calibrate_doubling.py`
  once kept its own copy of the star selection rules, drifted out of step
  with three fixes, and quietly validated a version that no longer
  existed. It calls `crop_centre()` and `measure_stars()` directly.
- **Run in `--mode report` while iterating.** It reports without touching
  a single file. Ask before switching to `copy` or `move`.
- **Prefer vectorisation for speed.** Replacing the per-component PCA
  loop with `bincount` moments gave a 12x speedup on that step, with
  numerically identical results. Gains like that cost nothing in
  behaviour; threshold changes always cost something.

## Code style

- English throughout: identifiers, comments, docstrings, console output.
- Comments explain **why**, and especially what was tried and failed. A
  comment that restates the code is noise; one that says "the OR version
  sent every satellite here" saves the next person a day.
- Docstrings written as prose, not as parameter lists. Say what the
  function is for and what makes it non-obvious.
- In the tests, the *name* says what is being checked and a docstring is
  added only when there is a reason the name cannot carry -- why the
  case matters, or a fact that is invisible from the assertion. About a
  quarter of them have one. Do not "finish the job" by documenting the
  rest: `test_a_square_is_not_elongated` needs no gloss, and when
  everything is commented nothing stands out.
- Type hints throughout. Dataclasses over ad-hoc dicts for anything
  crossing a function boundary.
- All thresholds in `Settings`, in arcseconds, documented in place with
  the measurement that justifies them and the direction to move them.
- Console output is user-facing and uses emoji; keep stdout forced to
  UTF-8 or Windows cp1252 consoles abort on the first print.

## Files

| | |
| --- | --- |
| `subframe_inspector.py` | the detector, and its command line. Every measurement and every verdict lives here |
| `session_layout.py` | where a session's frames and previews are: relative patterns, tried from the session folder, plus the saved preferences. Standard library only, because the labeller must run without numpy |
| `subframe_inspector_app.py` | the analysis window. Calls `classify_session` exactly as `main()` does, with stdout redirected into its log pane; decides nothing itself |
| `session_browser.py` | the session list both windows open on, the layout dialog, and the error hooks that make a traceback visible under pythonw |
| `subframe_labeller.py` | the labelling window: frames one at a time with a close-up beside them, recording what the eye saw into `<session>/labels.csv`. Filters both halves -- what the detector flagged and what the owner ticked, `Clean for me` included, which is the view to point them at when the corpus needs solid negatives. Shows the JPG preview, the `REVIEW/` copy or the FITS itself, so a session with no previews still works. Reads `report.csv` rather than re-deriving any verdict; tkinter + Pillow, with numpy/astropy needed only for the FITS source |
| `setup.bat` | one-time install for other users: a `.venv` beside the tools, and two `.lnk` shortcuts that start the windows through `pythonw` |
| `DETECTION.md` | the specification behind every verdict: hypothesis, quantity, reference, number, and the measurement justifying it, per category -- plus the blind spots each one has |
| `test_subframe_inspector.py` | stdlib `unittest`, no fixtures on disk, about a second |
| `replay_cases.py` | re-runs the decisions on exported case bundles, with any settings, and scores them against the labels. Imports every rule from `subframe_inspector` and re-implements none |
| `calibrate_doubling.py` | a tuning aid for the doubling detector, against real frames |

## Sharing the tool

The owner now shares it with other Seestar users, who will not have a
command line habit, a WinPython install, or the owner's folder layout.
Their choices, from when this was set up:

- **Python plus a launcher, not a frozen exe.** `setup.bat` builds a
  venv and shortcuts; users install Python themselves.
- **Two windows, not one.** The inspector and the labeller stay
  separate programs; the inspector only launches the labeller.
- **English** throughout the interface.
- **Any layout.** Sessions are found by `session_layout` patterns,
  never by a `_sub` suffix or a fixed `JPG/` folder, and users can type
  their own patterns. Nothing may go back to assuming the owner's disk.

Two things that follow and are easy to break:

- **Nothing may require a console.** Under pythonw there is no stdin and
  no visible stdout: an `input()` hangs forever and a traceback vanishes.
  The command line keeps its prompts; the windows use dialogs, and
  `install_error_hooks` writes uncaught errors to
  `subframe_inspector_errors.log`.
- **The deletion guarantees hold in the window too.** `DeleteDialog`
  is reachable only after a run made in that window that filed the
  previews, lists every frame, warns at half the session, and stays
  disabled until `delete` is typed. Do not add a shortcut around it.

`REVIEW/` housekeeping follows the previews wherever the layout found
them: a copy is removed only if a preview of that name still exists.
That is the same rule as before, with `JPG/` generalised -- keep it.

## Environment

- The owner develops on more than one Windows machine, so nothing here
  should assume a particular core count, clock or interpreter path.
- Use the `.venv` that `setup.bat` creates: `.venv\Scripts\python.exe`.
  The owner also has a WinPython install (look for `C:\WPy64-*`), but the
  `python` on PATH usually does **not** have astropy — check before
  concluding the code is broken. A session's `report.csv` came out
  byte-identical under WinPython (numpy 2.x, astropy 6.1) and the venv
  (numpy 2.5, astropy 8).
- Set `PYTHONIOENCODING=utf-8` when running from a console that is not
  already UTF-8.
- Any helper script that uses multiprocessing needs an
  `if __name__ == "__main__":` guard — Windows spawns rather than forks,
  and without it the children re-import and respawn forever.
