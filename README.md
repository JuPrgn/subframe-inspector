# subframe_inspector

Triage for Seestar light frames, run before stacking. It reads a night's
subs, works out which ones are worth a second look, and files their JPG
previews into one folder per problem — trailed stars, cloud, a satellite
across the field.

It is a sorting tool, not an astrometry package. Nothing is registered,
transformed or resampled, and a whole session goes through in well under
a minute.

**What it touches.** Everything it produces lives under `REVIEW/`, and
the only files it deletes there are the ones it wrote itself. Two options
reach outside that folder, both off by default: `--mode move` takes the
JPG *preview* out of `JPG/` rather than copying it, and
`--delete-flagged` removes the `.fit` files of flagged frames. Without
those two, your FITS are read and nothing more.

## Requirements

Python 3.10 or newer:

```
pip install -r requirements.txt
```

`numpy`, `scipy` and `astropy` are needed; `pillow` only for `--debug`,
and the run says so rather than failing if it is missing.

**Instruments.** It is built for the whole Seestar range: the focal
length and pixel pitch come out of the FITS header and every threshold is
written in arcseconds, so nothing is tied to one sensor. In practice it
has only been run on an **S30 Pro** — the S50 and S50 Pro paths are
sound in principle and untested in fact. Other cameras should work too as
long as their headers carry `FOCALLEN` and `XPIXSZ`; without those the
run warns you that it is guessing the plate scale, and the thresholds
will need moving.

## Layout

Put the script next to your capture folders:

```
subframe_inspector.py
LBN 550_sub/
├── Light_....fit          the frames to inspect
├── JPG/                   one .jpg preview per .fit, same base name
└── REVIEW/                created by the script
    ├── Trailing/
    ├── Clouds_Haze/
    ├── Satellites/
    ├── Meteors/
    └── Aircraft/
```

`JPG/` is optional. Without it the run reports normally, says so, and
copies nothing.

## Usage

```
python subframe_inspector.py                        # pick a folder, copy previews
python subframe_inspector.py "LBN 550_sub"
python subframe_inspector.py "LBN 550_sub" --mode report
python subframe_inspector.py "LBN 550_sub" --debug
python subframe_inspector.py "LBN 550_sub" --trailing 1.45
```

With no folder it finds the `*_sub` folders beside it and asks which one
to use. A frame that raised several alerts is filed under each of them.

| Option | Effect |
| --- | --- |
| `--mode copy` | copy previews into `REVIEW/` (default) |
| `--mode move` | move them out of `JPG/` |
| `--mode report` | write the two reports and nothing else |
| `--debug` | draw on the previews as they are filed, showing what triggered each alert |
| `--trailing RATIO` | how oval the stars must be to count as trailed (default 1.30) |
| `--export-cases` | also write a case bundle: every measurement, none of the pixels — see below |
| `--workers N` | cap the worker processes. Lower it if a run reports frames it could not read for want of memory; it changes no measurement |
| `--delete-flagged` | delete the `.fit` files of flagged frames — irreversible, see below |
| `--categories ...` | restrict `--delete-flagged` to some categories |
| `--yes` | skip the confirmation prompt |

## What it flags

| Alert | Meaning |
| --- | --- |
| ☁️ Haze / cloud | the sky background is well above the session's own median |
| 🫨 Trailing | the stars have lost their shape — drawn out into ovals, swollen without leaning anywhere, or printed twice by a jolt |
| 🛰️ Satellite | a straight track, no wider than the seeing disc |
| ☄️ Meteor | a wide track with a bright head and a fading tail |
| ✈️ Aircraft | a wide track lit evenly, on an unusually bright frame |

Each alert is decided on its own, so one frame can raise several — a hazy
frame crossed by a satellite is reported as both.

**A field that merely moved is not a fault, and is not flagged.** Wind
and play in the mount shift the pointing between exposures. As long as
the stars stay round, stacking registers that away and the edges get
cropped anyway. What makes a frame bad is the stars themselves being
smeared. The shift is still measured — it is in the CSV, and it is often
what explains a trailed frame — but it never raises an alert.

## How it works

### Two passes over each frame

A frame is read once and looked at twice.

The **working pass** covers the whole frame, decimated to roughly
7.5 arcsec per pixel — a factor of two on an S30 Pro, more on a longer
focal length. This is where trails are found: they are large, faint and
spread across the field, so resolution buys nothing and costs a lot.

The **detail pass** covers the middle quarter of the frame at full
resolution, where the shape of the stars and their exact positions are
measured. Keeping it to the centre costs about as much as the working
pass, and avoids the corner aberrations that would stretch stars for
optical reasons rather than mechanical ones.

### Finding structures

Both passes start the same way. A morphological **opening** — an erosion
followed by a dilation — erases anything smaller than its kernel, which
on a star field means the stars themselves, leaving an estimate of the
local background. Subtracting it (a *top-hat* filter) flattens gradients,
nebulosity and vignetting in one step, so a single threshold works
everywhere in the frame. Pixels above a high percentile are kept and
grouped into **connected components**, diagonals included — without that,
a trail at 45° breaks into a line of isolated pixels.

Each component is described by its **principal axis** rather than by a
bounding box: a diagonal trail has an almost square bounding box, so
height over width misses it entirely, while the eigenvectors of its pixel
coordinates do not care about orientation. A dense field yields some
twenty thousand components, so their moments are accumulated for all of
them at once rather than one at a time.

### Trailing

The verdict rests on one number: the median **elongation** of a few
hundred stars, taken as √(λ₁/λ₂) of their flux-weighted second moments
and compared against the sharpest quarter of the session.

Weighting by flux, at full resolution, is what makes this measurable. A
star covers only a few pixels, so counting them gives a handful of
possible answers and no way to tell 1.2 from 1.3; the light distribution
inside those pixels is continuous. For scale, on one 230-frame session
the median frame came out at 1.04× the reference and every frame its
owner had picked out by eye at 1.38× or more. The two populations shade
into each other — wind smears progressively — which is why `--trailing`
is a severity dial rather than a fixed line.

Stars can also lose their shape without leaning anywhere: focus drift,
or a spell of bad seeing, swells them evenly. The oval test cannot see
that — on frames where it happens the elongation sits near the session
median while the star **radius** runs 1.4 to 1.5 times the session's
sharpest — so radius is tested in its own right, and the report says
which of the two it found.

A third and quite different fault is caught here. A sharp knock during
the exposure prints every star **twice**, offset by one shared vector,
both images perfectly round; no shape measurement sees that. Instead the
detector takes the vector between every plausible pair of star centres,
folds each onto a single sign so that A→B and B→A agree, and looks for
the offset gathering the most pairs. Real doubling gives a sharp peak
shared by a quarter or more of the stars, where a merely crowded field
has no dominant offset at all — its best one rarely reaches 8%.

### Straight tracks

The Seestar's field is a couple of degrees across, so anything crossing
it — satellite, meteor, aircraft — flies dead straight. The detector
looks for **one line**, and reassembles it when noise breaks the trail
into fragments.

**A fragment is recognised by being thin, not by being long.** The
longest piece of a real trail runs from ten to nearly two hundred times
as long as it is wide. A smeared star reaches six at most, and a pair of
stars merged by the threshold sits between three and four.

**Only a convincingly thin fragment may start a line; any fragment may
join one.** That asymmetry is what finds a faint trail, which arrives as
a dotted line of specks barely longer than wide — too weak to start
anything, but perfectly able to confirm a line that already exists. It
also means a merged pair of stars can never start one.

**A line is judged by how straight it is.** On a trailed frame every star
is smeared the same way, so a line drawn along that common direction
picks up a few of them by chance. What separates those from a real trail
is not how many pieces there are but how well they sit on the line: a
trail *is* a straight line, so its fragments land within a fraction of a
pixel, while dashes gathered by luck spread through the whole search
band. Three pieces or more are judged on that residual, which is what
lets a satellite faint enough to leave only three specks still be
reported. One or two pieces fit any line exactly, so there the trail has
to be continuous instead — and the same applies on a frame whose own
stars are trailed, where three dashes can fall into line by accident.

Because a short fragment fixes its own angle only to within a few
degrees — and a few degrees is a long way off at the other end of the
frame — the line is refit through whatever it gathered before being
measured.

### Telling the three apart

By **width**, which has a natural ceiling. A satellite trail is a
straight line convolved with the seeing disc and nothing more, so its
width is set by the optics rather than by brightness: across two dozen
confirmed satellites spanning a factor of 69 in peak brightness, none
exceeded 11 arcsec. Anything clearly wider is not an ordinary satellite —
a meteor burns a plasma trail with a glow around it, an aircraft carries
several lights a wingspan apart.

The width is measured from the flux distribution across the trail rather
than from its outline, which a few stray pixels can inflate fivefold.

Which kind of wide thing it is then comes from the light *along* the
trail: evenly lit for the whole exposure is an aircraft, a bright head
with a fading tail is a meteor. That last distinction is the one part of
this with no confirmed example behind it — see
[Known limits](#known-limits).

### Field shift and cloud

The **shift** between consecutive frames is found without registering
anything: take the vectors between every pair of star positions across
the two frames and drop them into a 2D histogram; the dominant bin is the
translation. It is reported, never held against the frame.

**Cloud and haze** are read three ways, all against the session's own
frames: the sky brightening well past the session median, a smaller flat
rise, and — the one that catches the most — a veil that dims the stars
without raising the sky at all, seen as the star count and the frame
contrast dropping together. An absolute sky brightness would mean nothing
across different targets, gains and skies.

### One scale for every model

Every threshold that measures the sky is written in **arcseconds**. The
plate scale is computed once per session from `FOCALLEN` and `XPIXSZ` in
the header, and turned into pixels in a single place. The decimation
factor follows from it too, so the working image lands near a constant
angular resolution whatever the instrument is.

## Debug mode

`--debug` draws the detector's own view onto the previews as they are
filed, so the folder you review already shows why each frame is in it:

| Colour | What it is |
| --- | --- |
| 🔴 red line, circled ends | the track that was found, end to end |
| 🟠 orange | each fragment that was strung into it |
| 🔵 blue rectangle | the region the star shapes were measured in |
| 🔷 cyan ticks | every measured star, along its own axis, longer the more elongated |
| 🟣 magenta | the doubling offset, drawn on the stars it was found from |

A trailed frame shows a comb of cyan ticks all leaning the same way; a
clean one shows a field of dots. The verdict and its numbers are printed
across the top.

The preview in `JPG/` is read and left alone — only the copies under
`REVIEW/` carry the marks. Needs Pillow, and adds a few seconds. With
`--mode move` the preview leaves `JPG/` and only the annotated copy
survives, which the run warns you about.

## Reports and housekeeping

Two files land in `REVIEW/` on every run, whether or not the previews
exist:

- **`report.csv`** — one row per frame, flagged or not, with every
  measurement the verdicts were based on. This is the useful one: sort it
  in a spreadsheet, or re-try a threshold on it, without re-reading
  gigabytes of FITS.
- **`report.md`** — a readable summary: counts per category, then the
  frames in each with the numbers that put them there.

| CSV columns | What they hold |
| --- | --- |
| `background`, `haze_sigma`, `peak`, `noise` | the sky and the frame |
| `stars`, `elongation`, `elongation_ratio` | the trailing verdict |
| `star_radius`, `radius_ratio`, `coherence` | star size, and whether they all lean one way |
| `track_span`, `track_coverage`, `track_fragments`, `track_tilt`, `track_width` | the track's geometry |
| `track_rms_width`, `track_glow`, `track_peak` | its flux-weighted width, and its brightness per unit length and at its brightest, in units of the frame noise |
| `head_ratio`, `head_position`, `taper` | how the light sits along it |
| `shift_px`, `shift_dx`, `shift_dy`, `shift_confidence` | the field shift |
| `doubling_px`, `doubling_fraction` | the jolt detector |

Every run clears the previous one out of `REVIEW/` first, so the folder
shows the run you just did rather than the union of every run so far —
including category folders left behind by older versions of the script.

Deleting inside a folder you also use calls for a rule that cannot go
wrong, so there is one: **a file is removed only if a file of the same
name still exists in `JPG/`.** Everything the script writes is a copy of
a preview, so everything it wrote is covered, and anything else is left
where it is — your own notes, sub-folders and picks, or a preview that a
previous `--mode move` run took out of `JPG/` and which therefore exists
nowhere else. `--mode report` does no housekeeping at all.

## Deleting the FITS

The one thing here that cannot be undone, and the only part that touches
your data rather than a copy of a preview. It is off unless asked for by
name:

```
python subframe_inspector.py "LBN 550_sub" --delete-flagged
python subframe_inspector.py "LBN 550_sub" --delete-flagged --categories Trailing Clouds_Haze
python subframe_inspector.py "LBN 550_sub" --delete-flagged --categories Trailing --yes
```

Deletion happens after the previews are filed, so the JPG of everything
removed is still in `REVIEW/` for you to look at afterwards. Before
anything goes, the run lists every frame it covers with the categories
that put it there, and waits for you to type `delete` — not `y`, which is
too easy to hit by reflex. `--yes` skips the prompt for scripted runs.

If the selection covers more than half the session, the run says so
loudly. That is the honest failure mode of this tool: `--trailing` is a
severity dial, and one notch too tight quietly selects half the night.
When that warning appears, the threshold is the thing to change, not the
answer to the question.

## Known limits

- **The meteor and aircraft categories are unproven.** Neither has fired
  on any session tested so far. The gate they share — a track wider than
  the seeing disc — is on firm ground; the split between them, which goes
  on whether the light fades along the trail, is not. Getting it wrong
  files one preview in the wrong folder, which is why it ships anyway.
- **A trail's brightness profile alone would not identify a meteor.** A
  satellite can glint in mid-passage, and a star sitting on the trail
  inflates the peak, so `head_ratio` and `taper` vary widely among
  ordinary satellites. Width is what decides; the profile only chooses
  between two tracks that are already wide.
- **Mild trailing is a judgement call.** The defaults catch the clear
  cases. Frames where the smear is just perceptible sit below them on
  purpose: they are hard to call by eye, and a stacker that registers
  each sub will absorb them. Lower `--trailing` for a night you want
  judged strictly, and sort the CSV on `elongation_ratio` to see where
  the line falls on your own data.

## Keeping a session after the frames are gone

A night of subs is four gigabytes. What the detector actually reasons
about is a few hundred numbers per frame, and those fit in three
megabytes:

```
python subframe_inspector.py "LBN 550_sub" --mode report --export-cases
```

writes `cases_LBN_550_20260830-000205.json.gz` **next to the script** —
every measurement the run made, the labels recorded so far, and what it
concluded at the time. It lands there rather than in the session folder
because it is meant to outlive the frames: delete the subs once they are
stacked and the cases are still on hand. `replay_cases.py` reads those bundles back, re-runs the whole
classification with whatever settings you give it, and scores the result
against your labels:

```
python replay_cases.py                                   # as exported
python replay_cases.py --set trailing_elongation_ratio=1.25
```

```
🧳 3 session(s), 540 frames, 132 labelled

                caught  ticked  recall  flagged  precision
      Trailing      58      70     83%       60        97%
    Satellites      27      30     90%       28        96%
   Clouds_Haze      38      40     95%       38       100%

✅ Every frame comes out exactly as the bundle recorded it.
```

Every bundle next to the script is replayed as one set, and the whole
corpus takes about four seconds — so trying a threshold is a loop, not an
afternoon. Bundles pile up: each night you export adds its cases, and the
next threshold gets judged on all of them.

**Bundles from different targets mix freely.** Everything a verdict rests
on is a ratio to that night's own stars or an absolute in arcseconds, and
the night's yardstick is stored inside the bundle. The target appears in
the file name so you can find the session again; nothing the detector
looks at knows about it, and renaming the sub folder changes no identity
— that comes from the frames themselves.

**What a bundle cannot do** is change how the numbers were measured. The
decimation, the top-hat, the detection floor and the star selection all
need the frames; `--set` refuses those settings rather than pretending.
A bundle also records which code measured it, and the replay stops if
that code has changed since — re-export from the FITS when it does, which
is why the FITS are worth keeping even when the previews are not.

If you label a session *after* exporting it, there is no need to
re-export:

```
python replay_cases.py --labels "LBN 550_sub/labels.csv"
```

**The pictures worth keeping.** A bundle can say a track was found and how
wide it was; it cannot say whether the thing in the frame is a satellite.
For that:

```
python replay_cases.py --disagreements previews_out
```

collects the previews of the frames where the detector and your labels
disagree, plus one control frame per session — the frame closest to that
night's own norm. About 290 KB each and a few dozen files, not the
several thousand a session holds. It works without a `JPG/` folder:
frames with no preview are rendered from the FITS.

Add `--fits-budget 400` to also copy the frames themselves for the cases
where only real pixels can answer — ranked so the budget goes to the
frames the bundle holds nothing on. It copies; nothing moves or changes.

## Where the thresholds come from

`DETECTION.md` is the specification behind every verdict: for each
category, the physical hypothesis, the quantity measured, what it is
compared against, the number, and the measurement that justifies the
number — plus the blind spots each one still has. Read it before moving
a threshold.

Three things worth knowing from it up front:

- **Nothing is judged in absolute terms.** A frame is compared against a
  normal frame of its own session, which is what lets one set of numbers
  work under a suburban sky and a dark one. The flip side is that when
  *most* of a night is bad, "unusual for this night" stops meaning much.
- **Cloud wears three faces.** The sky brightening, a smaller flat rise,
  and — the one that catches the most — a veil that dims the stars
  without brightening the sky at all. That last one is caught by the
  star count collapsing *and* the frame's contrast dropping together.
- **The trailing dial has no correct value.** The frames that sit near it
  are spread evenly, with no gap to settle in, so `--trailing` is a
  severity slider rather than a constant to get right.

## Tests

```
python -m unittest -v test_subframe_inspector
```

No FITS file or night's data needed — the fixtures are Gaussian blobs on
a flat background. What they cover is the machinery that fails silently:
arcseconds converted to the wrong pixel scale, a track detector that
stops telling a line from a scatter, and every path that removes a file.
Whether a threshold sits in the right place is a question for real frames
and the CSV, and no unit test can answer it.

## Labelling by eye

Every threshold here was fixed by looking at previews and writing down
what was actually on them. `label_frames.py` is that job made bearable:

```
python label_frames.py "LBN 550_sub"
python label_frames.py                  # asks which session
```

It shows the previews one at a time, in the order they were shot, with
what the detector made of each one beside the image, and records what you
saw instead.

Beside the whole frame sits a **close-up**, a strip of it at full
resolution or better. That is the quick read on trailing: the same patch
of sky shows a field of dots on a good frame and a field of dashes all
leaning the same way on a trailed one. Click anywhere on the main image
to point it somewhere else — it stays there as you page through the
session, which is what makes two frames comparable at all.

| | |
| --- | --- |
| ← → | previous / next frame |
| Home / End | first, last |
| PgUp / PgDn | ten at a time |
| drag the bar | scrub through the list with the mouse |
| Ctrl+F | find a frame by its time |
| 1-6 | tick a category — several per frame, `1` says the frame is fine |
| 0 | untick everything on this frame |
| A | tick exactly what the detector said, ready to validate |
| Enter | record it and move on |
| wheel, drag | zoom under the cursor, pan |
| F | fit the frame to the window |
| click | point the close-up at that spot |
| `[` `]` | zoom the close-up out / in (wheel over it does the same) |
| C | put the close-up back in the middle of the frame |
| D | next image source (or pick one from the menu at the top) |
| L | keep the zoom and pan across frames, to compare one corner |

The category keys **tick**, they do not toggle: going through one
category means pressing the same key frame after frame, and a toggle
turns the second press on an already-ticked frame into a silent
unticking. `0`, or the mouse, undoes one.

One step past the last frame is an **end card** across the image area,
saying which list it was and how much of it is labelled. Paging is done
with the eyes on the picture, and a list that quietly starts again looks
exactly like carrying on — so the card is a seam rather than a wall: →
once more does go round to the first frame, it just costs a deliberate
press.

The **bar along the bottom** is the map of the session and the way
around it. Each frame you have labelled is a green tick on it, so what is
done and what is left read at a glance; the handle sits at the frame on
screen; and clicking or dragging anywhere on it goes there, which is how
you reach the far end without paging through the middle. The pointer
changes over it to say so.

### Choosing what to walk through

The drop-down at the top holds two halves, and they answer different
questions:

- **what the detector said** — everything, only what it flagged, one of
  its categories, or only what it thought was clean;
- **what you said** — labelled or not yet, then one entry per category
  you have actually ticked, so `Mine: Satellites` collects your own picks
  rather than the tool's.

Below both, *Detector and I disagree*. Each entry carries how many frames
it holds, worked out when the menu opens — which is also why a category
you have never ticked is not listed at all, and appears by itself the
moment it holds something.

**Ctrl+F** finds a frame by its time — type `0040` for
`20260830-004055`. It matches the timestamp, not the whole name, because
every frame in a session shares its first thirty characters. Pick one
from the list and it goes there, widening the filter to *All frames* if
the current view does not hold it.

Labels go to `<session>/labels.csv`, one line per frame, written the
moment you validate — so a session can be gone through over several
evenings, and coming back to a frame shows what you said last time. It
sits beside the frames rather than under `REVIEW/`, which every run of
the detector empties.

### Which image it shows

Three per frame, any of which may be missing. The menu at the top of the
window offers the ones the frame on screen actually has, and `D` walks
through the same list from the keyboard:

- the untouched **JPG preview** in `JPG/` — the default;
- the copy `REVIEW/` filed, which carries the `--debug` drawings;
- the **FITS itself**, stretched for the screen.

A session with no `JPG/` folder therefore works exactly the same: the
frames are read from the FITS instead, and nothing else changes. Reading
one costs about what decoding a JPEG costs — 57 ms against 77 ms on a
2160×3840 frame — because the statistics come from a subsample and the
stretch goes through a lookup table rather than over eight million
pixels in floating point.

Two things to know about the FITS view. It is greyscale, and on a
Seestar it is the **raw Bayer mosaic**, so the sky carries a fine
checkerboard the JPG does not have; star shapes and trails read the same
on both, which is what the view is for. And it is stretched here for the
eye, whereas the detector measures the linear data — so a faint thing
may show up in one and not the other.

Needs tkinter and Pillow. numpy and astropy only for the FITS source,
which is simply not offered if they are missing.

## Tuning

Every threshold lives in the `Settings` dataclass at the top of the file,
documented in place. Override without editing it:

```python
from dataclasses import replace
from subframe_inspector import DEFAULT_SETTINGS, classify_session

settings = replace(DEFAULT_SETTINGS, trailing_elongation_ratio=1.45)
classify_session("LBN 550_sub", mode="copy", settings=settings)
```

The ones you are most likely to move:

- **`trailing_elongation_ratio`** (1.30) — how oval the stars must be,
  against the sharpest quarter of the session, to count as trailed. 1.20
  is strict, 1.45 catches only obvious ovals. Also on the command line,
  as `--trailing`.
- **`trailing_radius_ratio`** (1.35) — how much the stars may swell,
  against the same reference, before the frame counts as soft. This is
  the focus-and-seeing half of the trailing verdict; raise it if you only
  care about the mount.
- **`track_min_span`** (150 arcsec) — the shortest trail worth
  reporting. Raise it if short corner streaks are not worth your time.
- **`wide_track_width`** (18 arcsec) — where a track stops being an
  ordinary satellite and becomes a meteor or an aircraft. No confirmed
  satellite has measured wider than 11.
- **`doubling_min_fraction`** (0.20) — sensitivity to star doubling. On a
  dense field the noise floor sits around 6%, so this leaves 3× margin.

Everything that measures the sky is in arcseconds, converted to pixels
once per session. No threshold is quoted in pixels of one image and
applied to another.

`calibrate_doubling.py` checks the doubling detector against real frames:
false alarms, sensitivity to a synthetic doubling of known offset,
behaviour when the second image is fainter, and the noise floor your
threshold has to clear. Worth running after touching any doubling
threshold, or on a field with a very different star density.

```
python calibrate_doubling.py "LBN 550_sub" [frame_count]
```

## Performance

Frames are read once, decimated to a constant angular scale, and measured
with a single vectorised pass over all connected components; one extra
pass at full resolution covers the centre of the frame.

The two morphological openings are about 45% of the work, and they are
already optimal — scipy runs the van Herk algorithm, so the kernel size
costs nothing. What limits the rest is memory bandwidth rather than
cores, so the worker pool stops scaling some way before it runs out of
them.

Expect the wall-clock figure to move a lot between machines, and even on
one: on a laptop the same session can take twice as long on battery, or
after an hour of continuous work, as it does on mains from cold. Compare
runs taken under the same conditions.
