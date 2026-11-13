# Detection specification

What each verdict measures, against what, and at which threshold.

`CLAUDE.md` holds the approaches already tried and rejected, and how to
work in this repository. Nothing here is repeated there.

---

## Invariants

- Sky-referenced settings are in arcseconds. `Geometry` reads the plate
  scale from `FOCALLEN` and `XPIXSZ`; `Thresholds.derive` converts once.
  Nothing else converts anything.
- Two pixel scales: a decimated working image at ~7.5 arcsec/px, and one
  full-resolution pass over the central `sharpness_centre_fraction` (0.5)
  of the frame. Every threshold field names the one it belongs to.
- No threshold is absolute. Each compares a frame against a reference
  built from its own session.
- `analyse_frame` is the last function to touch a pixel. Everything after
  it works on numbers, which is what case bundles replay.

## Session references

| quantity | taken at | reason |
| --- | --- | --- |
| `background`, `noise` | median | — |
| `star_radius`, `star_elongation` | 1st quartile | a night half smeared must not redefine smeared as normal |
| `star_count` | 3rd quartile | a night half veiled must not redefine sparse as normal |

`background_spread` is the standard deviation of the session's
backgrounds, floored at 1.0.

---

## Clouds_Haze

Four independent branches, evaluated in order; the first to fire wins.

| # | condition | setting | value |
| --- | --- | --- | ---: |
| 0 | no star measurable at all | `sharpness_min_stars` | 30 |
| 1 | `(background − ref) / spread` above | `haze_sigma` | 2.5 |
| 2 | `background / ref` above | `haze_background_rise` | 1.10 |
| 3 | `noise / ref` below **and** `stars / ref` below | `haze_contrast_drop`, `haze_star_drop` | 0.90, 0.85 |

Branch 0 comes first because it is the most severe and the most certain.
`measure_stars` gives up below `sharpness_min_stars` and returns
`StarShape.none()`; `_ratio` then answers 1.0 for "nothing to compare",
so branch 3's AND read a frame with no measurable star as normal and the
frame raised nothing at all. Worth **16 of the 222 ticked frames for no
extra wrong flag**.

Branch 3 exists because a veil dims stars without raising the sky.
`noise` is `np.std` over the working image, so it is carried by stars and
structure rather than by read noise. Separation over 222 ticked frames,
nine sessions:

| | ticked | judged, not ticked |
| --- | ---: | ---: |
| `stars / ref` | 0.44 | 0.96 |
| `noise / ref` | 0.69 | 1.01 |

The two are ANDed: either alone reaches the same recall for 2x to 12x the
wrong flags. Branch 3 is worth 31 points of recall.

**Blind spot.** 44 of 222 ticks are missed, with a background at the
session median, a normal star count and normal contrast.

*A local cloud was measured and set aside.* Comparing each of the 4x4
tiles against its own session median, the missed frames' emptiest tile
runs 0.72 against 0.95 on frames judged clean — a real signal. But over
the four sessions carrying spatial data it adds only 7 frames beyond
branch 0, for 2 wrong flags and 6.6 ms a frame always on, and most of what
looked local was branch 0's bug. The tiles stay in the bundles, so
re-testing this costs no FITS.

---

## Trailing

Three independent signatures. Any one raises the verdict.

| # | signature | quantity | setting | value |
| --- | --- | --- | --- | ---: |
| 1 | stars drawn into ovals | median sqrt(major/minor), flux-weighted moments, full resolution | `trailing_elongation_ratio` | 1.25 |
| 2 | stars swollen, no common lean | median star radius | `trailing_radius_ratio` | 1.35 |
| 3 | every star printed twice | pairwise offset vote | `doubling_min_fraction` | 0.20 |

**Branch 1 is a severity dial, not a constant.** The 139 missed ticks are
spread flat between 1.00x and 1.30x — 24, 24, 22, 24, 20, 25 per 0.05
band. No gap exists to place it in. Bracketed by two frames checked
against their previews: `023406` at 1.26x is trailed, `034739` at 1.22x
is not. `--trailing` overrides it per session.

Measure branch 1 from flux-weighted second moments at full resolution. A
thresholded bounding box on the decimated image takes ~8 distinct values
over a session and separates nothing.

**Branch 2** catches focus drift and seeing. On frames ticked for it,
`elongation_ratio` sits at the session median while `radius_ratio` runs
1.36 to 1.54. Lowering it is less efficient than lowering branch 1: 16
frames against 25 for the same cost.

**Branch 3** takes the vector between every plausible pair of stars,
signed so A→B and B→A agree, and looks for the mode. Bounds:

| | setting | value |
| --- | --- | ---: |
| shortest offset | `doubling_min_distance` | 22 arcsec |
| longest offset | `doubling_max_fraction` x crop | 0.33 |
| offsets counted as one | `doubling_tolerance` | 19 arcsec |
| minimum pairs | `doubling_min_pairs` | 3 |
| stars used | `doubling_max_stars` | 150 brightest |

The upper bound is geometric, not angular: a knock is instantaneous and
unbounded in size, but past a third of the crop too few stars keep a
partner inside it for the fraction to mean anything. Verify a claimed
doubling by autocorrelation, not by eye.

**The recall on this branch measures the labelling, not the detector.**
Per session, the median `elongation_ratio` of the frames ticked Trailing
against the recall on them:

| ticked at | 1.56 | 1.50 | 1.45 | 1.35 | 1.23 | 1.18 | 1.13 | 1.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recall | 100% | 100% | 81% | 100% | 66% | 33% | 23% | 0% |

Two of those are the same target one night apart. No dial value satisfies
both ends, because the severity being ticked moves by a factor of ten
between nights. Pooled recall on this category is therefore not a
measure of the detector; read it per session, and use `--trailing` for a
night judged strictly.

**Blind spots, both closed.** 43 of the 139 missed ticks have elongation
below 1.10 and radius below 1.20 — stars as round as the sharpest quarter
of their own night. Two explanations were tested and both fail:

- *Trailing outside the measured centre.* `measure_stars` run over a 3x3
  grid of five such frames at full resolution: corner elongation is
  0.94–1.00 of the centre, and 0.91–0.96 on unticked controls from the
  same night. The corners are no worse than the middle.
- *A night soft everywhere, absorbed by its own reference.* The absolute
  reference elongation spans 1.21 to 1.34 across nine sessions while the
  ticked share spans 1% to 21%. It carries no information, so an absolute
  floor cannot work.

---

## Satellites, meteors, aircraft

One detection, three labels. The field spans a couple of degrees, so
anything crossing it prints a straight line; the detector finds one line
and reassembles it from fragments.

| stage | setting | value |
| --- | --- | ---: |
| a fragment may **start** a line at ratio | `seed_min_ratio` | 4.75 |
| a fragment may **join** one at ratio | `segment_min_ratio` | 4.0 |
| minimum fragment length | `segment_min_length` | 30 arcsec |
| gathering distance from the line | `track_tolerance` | 75 arcsec |
| maximum angle between fragment and line | `track_max_tilt` | 25° |
| minimum end-to-end span | `track_min_span` | 150 arcsec |
| RMS residual about the line, at ≥3 fragments | `track_max_residual` | 4.5 arcsec |
| lit share of the span, at 1–2 fragments | `track_min_coverage` | 0.5 |
| fragments needed on a trailed frame below that coverage | `track_fragments_if_soft` | 6 |

**Aspect ratio discriminates, length does not.** Longest fragment of a
real trail: 10x to 184x. Smeared star: 6x at most. Merged star pair:
2.7x to 4.3x. Every false positive ever measured sat exactly at the
detection floor in length.

**Straightness discriminates, fragment count does not.** RMS residual:
0.07–0.30 working px over nine confirmed trails, against 1.02, 5.29 and
7.12 for chance alignments. Two fragments fit any line exactly, hence
the coverage branch below three.

**The soft-frame gate counts pieces**, because the residual cannot
separate that case: chance triples on trailed frames measured 0.03–0.47
against a real trail's 0.15, while the fragment count read 3 against 22.

**Width decides which label.** A satellite trail is a line convolved with
the seeing disc, so its width is set by the optics, not by brightness.
Over 26 confirmed satellites:

| | min | median | max |
| --- | ---: | ---: | ---: |
| flux-weighted width (working px) | 0.43 | 1.06 | 1.52 |
| peak brightness (x frame noise) | 1.3 | 9.7 | 89.6 |

| | setting | value |
| --- | --- | ---: |
| wide enough to leave the satellite branch | `wide_track_width` | 18 arcsec (~2.4 working px) |
| tapered → meteor | `meteor_taper` | 2.0 |
| evenly lit and bright → aircraft | `aircraft_brightness` | 1.5 x session noise |

Use the flux-weighted width, not the geometric extent: outlying pixels
put the widest satellites at 7.6 working px on that measure. Do not
normalise it by the stars — it spans 0.62 to 2.30 over the same tracks.

The meteor/aircraft split is **unvalidated**: neither has been confirmed
in this data.

**Blind spot, and it is closed.** On `NGC 7380_Wizard` (1080x1920, 20 s
subs) all 22 ticked satellites are missed and no fragment exists — not
one component reaches ratio 3.0. Sweeping the seed bar from 4.75 to 3.0
over nine sessions recovers one frame. Three frames in `LBN 550` fail the
same way.

These trails are visible in the previews and are not recoverable from the
top-hat residual. On `034122`, whose trail was located by search rather
than by eye, the excess is about 3 counts against a per-pixel scatter of
200: over the 1600 samples along the line that is 0.8 sigma, and 2 of 17
wrong directions through the same point score higher. Counting pixels
above a lowered bar, integrating along the line, and doing so after
removing the row and column medians all give the same answer — see the
rejected-approaches table in `CLAUDE.md` for the numbers. Do not re-run
these.

---

## The wide-angle lens: declined, not supported

The Seestar's second lens is 6 mm at 1.6 micron pixels: **55 arcsec/px**,
a 33° x 59° field. Decimation only coarsens a scale, so `step` clamps at
1 and the working image stays seven times coarser than the 7.5 arcsec/px
every pixel-valued threshold assumes.

| threshold | main lens | wide angle |
| --- | ---: | ---: |
| `segment_min_length` (30") | 4.01 working px | **0.55** |
| `track_min_span` (150") | 20.1 | **2.7** |
| `star_max_length` (90") | 12.04 | **1.64** |

One pixel is longer than the minimum trail; three in a line cross the
field; any star over 1.64 px is disqualified from being a star. Measured
result: **59 frames of 59 flagged, 54 as satellites.**

`unsupported_scale` now declines such a session before reading anything,
`working_scale_tolerance` being 1.5. What it would take to support it,
both measured on those 59 frames:

- **Three thresholds are pixel quantities written as angles.**
  `segment_min_length`, `track_min_span` and `star_max_length` only ever
  worked in arcseconds because the working scale never moved. Given the
  same *working pixel* counts as the main lens, the session drops from
  59/59 to 6/59.
- **The remaining 6 are chance alignments and need the test that was
  shelved.** Their coverage is 0.02 — three specks over 700 px, no trail
  in the frame. The rule "three fragments and a straight line" was
  calibrated on a 2° field; a 33° field holds ten times the specks and
  three collinear ones are certain. The chance-alignment statistic
  recorded in `CLAUDE.md` as sound but useless on narrow fields is what
  separates this regime. A coverage floor does not: the confirmed
  satellite `051837` also sits at 2%.

## Field shift

Dominant vector between star positions of consecutive frames, by 2D
histogram. Not registration: nothing is transformed, no astrometry.

Reported in the CSV, never a verdict. A field that moved with round stars
stacks away and gets cropped. As a category it produced 47 of 59 alerts
on one session.

---

## Not measured

**Hot pixels.** There are none to find; the Seestar corrects its sensor
on board. A per-session mask was built and removed. Two masks of it
overlapped 3.4% across seeds of one session, 0.8% across the halves of
one night, and 0.4% across two sessions of the same camera — a defect map
would be identical. Over 30 frames, zero pixels reached the top 0.3% of
at least 15 of them.

Removing it changed one verdict over 540 frames: `014411` lost a
satellite that was the bridge between a bright star and its own second
image on a doubled frame. The frame stays flagged Trailing.

**Registration, astrometry, plate solving.** Out of scope by design.

---

## Exported, not yet used

`spatial_summary` writes these into every case bundle. No verdict reads
them; they exist so a spatial or photometric hypothesis can be scored
without the FITS.

| field | shape | for |
| --- | --- | --- |
| `tiles` | 4x4, each: component count, background, noise, median length, median ratio | a cloud over part of the frame; trailing away from the centre |
| `sky` | p5, p25, p50, p75, p95 of the working image | sky structure and gradient |
| `component_pixels`, `component_ratio` | deciles | the faint end, and the shape distribution |
| `star_map` | brightest 400 components: x, y, pixel count | star density anywhere in the frame |

The grid exists because 120 landmarks over 9 tiles is 13 per tile, and at
13 the counting noise alone produces a density hole of 0.62 ± 0.15 —
wider than the difference between cloudy and clear frames. A 4x4 grid
over every valid component puts hundreds in each tile.

Two hypotheses are waiting on this data, both listed as blind spots
above: a local cloud, and trailing outside the measured centre.

## Scoring a change

```
python subframe_inspector.py "<session>_sub" --mode report --export-cases
python replay_cases.py --set <name>=<value>
```

Precision counts only frames carrying a label. A flag on a frame ticked
for something else is an error; a flag on an unlabelled frame is
unknown — there is no `Clean` tick in the corpus. Never pool the two.

Nine sessions, 2484 frames, 648 labelled:

| | caught | ticked | recall | wrong | precision | unjudged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trailing | 230 | 344 | 67% | 6 | 97% | 36 |
| Satellites | 58 | 97 | 60% | 3 | 95% | 0 |
| Clouds_Haze | 162 | 222 | 73% | 11 | 94% | 20 |

`LBN 550` hand-picked reference: 14/14 trailing, 3/3 satellites. Re-check
after any change to detection.
