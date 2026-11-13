# -*- coding: utf-8 -*-
"""Where a session's frames and previews are, whatever the folder looks like.

Every tool here used to assume one layout: the light frames at the root of
a '<Target>_sub' folder sitting beside the script, their previews in a
'JPG/' folder inside it. That held for one owner's disk and nobody else's.
The Seestar itself writes '<Target>_sub/Lights/' and '<Target>_sub/JPG/'
on some firmware and puts everything side by side on others, and people
reorganise their nights in their own way after that.

So a layout is described by **relative patterns**, tried from the session
folder:

- ``frames``: tried in order, and the first that matches anything wins.
  That is what lets ``Lights/*.fit*`` take precedence over ``*.fit*``, so
  a stacked result left at the root of a session is not read as one more
  sub when the subs live in ``Lights/``.
- ``previews``: all of them are searched, and a frame takes the first JPG
  carrying its name. Previews are optional.

Whatever is written by the tools themselves -- ``REVIEW/``, ``labels.csv``
-- lives at the session folder, the one the patterns are relative to.

Standard library only: the labelling window runs without numpy, and so
must this.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass, field

# Tried in order for the frames, all of them for the previews. Matching is
# case-insensitive on every platform, so 'Lights/' finds 'lights/' too.
DEFAULT_FRAME_PATTERNS = ("Lights/*.fit*", "*.fit*")
DEFAULT_PREVIEW_PATTERNS = ("JPG/*.jpg", "JPG/*.jpeg", "Lights_JPG/*.jpg",
                            "Lights/*.jpg", "*.jpg", "*.jpeg")

# Never a sub: thumbnails, and stacks recognisable by their name -- the
# Seestar's 'Stacked_...' and 'Video_Stacked_...', anything else calling
# itself a stack, and the '<N>x<exposure>sec' a stacked result is usually
# named after ('SH2-132_301x60sec_T21degC_...').
EXCLUDED_NAMES = ("*_thn.*", "*stack*")
STACK_NAME = re.compile(r"\d+\s*x\s*\d+(?:\.\d+)?\s*s(?:ec)?(?![a-z])", re.IGNORECASE)
TOOL_FOLDERS = ("review",)

# Folders a session search never descends into: they are part of a
# session, or nobody's photographs.
SKIPPED_FOLDERS = {"review", "jpg", "lights", "lights_jpg", "__pycache__",
                   ".venv", "venv", ".git", "stack"}

FRAME_EXTENSIONS = (".fit", ".fits", ".jpg", ".jpeg")

CONFIG_FILENAME = "subframe_inspector_settings.json"

# The two programs, as their windows and shortcuts name them.
APP_NAME = "Subframe Inspector"
LABELLER_NAME = "Subframe Labeller"


def frame_stem(filename: str) -> str:
    """A file name without the extensions in play here.

    Not splitext: a Seestar name carries its exposure as '60.0s', so
    splitext takes '.0s_IRCUT_20260830-000305' for an extension.
    """
    stem = os.path.basename(filename)
    lowered = stem.lower()
    for extension in FRAME_EXTENSIONS:
        if lowered.endswith(extension):
            return stem[:-len(extension)]
    return stem


@dataclass(frozen=True)
class Layout:
    """How a session folder is organised, as patterns relative to it."""

    frames: tuple[str, ...] = DEFAULT_FRAME_PATTERNS
    previews: tuple[str, ...] = DEFAULT_PREVIEW_PATTERNS

    @staticmethod
    def parse(text: str) -> tuple[str, ...]:
        """Patterns typed by a person: separated by ';' or new lines."""
        parts = text.replace("\n", ";").split(";")
        return tuple(p.strip().replace("\\", "/") for p in parts if p.strip())

    def describe(self) -> str:
        return (f"frames: {'; '.join(self.frames)} | "
                f"previews: {'; '.join(self.previews)}")


DEFAULT_LAYOUT = Layout()


@dataclass
class Session:
    """One night: the folder it lives in, its frames, their previews."""

    root: str
    frames: list[str] = field(default_factory=list)
    previews: dict[str, str] = field(default_factory=dict)
    frame_pattern: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.root))

    def preview_for(self, frame_path: str) -> str | None:
        return self.previews.get(frame_stem(frame_path))


# ----------------------------------------------------------------------
#  Matching
# ----------------------------------------------------------------------

def _listdir(folder: str) -> list[str]:
    try:
        return sorted(os.listdir(folder))
    except OSError:
        return []


def _excluded(name: str) -> bool:
    lowered = name.lower()
    return (any(fnmatch.fnmatchcase(lowered, p) for p in EXCLUDED_NAMES)
            or bool(STACK_NAME.search(frame_stem(name))))


def read_fits_header(path: str, limit: int = 36) -> dict[str, str]:
    """The primary header's cards, without astropy.

    A header is 2880-byte blocks of 80-character cards ending at 'END';
    reading it by hand keeps this module free of numpy, and costs one
    small read per file.
    """
    cards: dict[str, str] = {}
    try:
        with open(path, "rb") as handle:
            for _ in range(limit):
                block = handle.read(2880)
                if len(block) < 2880:
                    break
                for start in range(0, 2880, 80):
                    card = block[start:start + 80].decode("ascii", "replace")
                    key = card[:8].strip()
                    if key == "END":
                        return cards
                    if card[8:10] == "= ":
                        value = card[10:].split("/", 1)[0].strip().strip("'").strip()
                        cards.setdefault(key, value)
    except OSError:
        pass
    return cards


def is_stacked(path: str) -> bool:
    """Whether a FITS is a stacked result rather than one sub.

    For a stack whose name gives nothing away -- a Seestar stack renamed
    to its timestamp looks exactly like a sub. The header does not lie:
    a stack counts what went into it (STACKCNT, NCOMBINE), and it is a
    debayered colour image, where a sub is a raw two-dimensional mosaic.
    """
    header = read_fits_header(path)

    def number(key: str) -> float:
        try:
            return float(header.get(key, "0") or 0)
        except ValueError:
            return 0.0

    return (number("STACKCNT") > 1 or number("NCOMBINE") > 1
            or number("NAXIS") >= 3)


def drop_stacks(frames: list[str]) -> list[str]:
    """Remove the stacks a name did not give away.

    Reading every header of a thousand-frame night would slow the search
    over a whole drive for nothing: a folder of subs does not hide a
    stack among them. So the first and last frames are checked, and only
    when one of them turns out to be a stack is the whole list read.
    """
    if not frames:
        return frames
    if not (is_stacked(frames[0]) or is_stacked(frames[-1])):
        return frames
    return [path for path in frames if not is_stacked(path)]


def match_pattern(root: str, pattern: str) -> list[str]:
    """Files under ``root`` matching one relative pattern, sorted by name.

    Each path component may hold wildcards, and every comparison ignores
    case, so a layout written on Windows means the same on a Mac. '**'
    is not supported: a session is a couple of levels deep at most, and
    a recursive pattern would walk into REVIEW/ and pick up its copies.
    """
    parts = [p for p in pattern.replace("\\", "/").split("/") if p and p != "."]
    if not parts:
        return []
    folders = [root]
    for part in parts[:-1]:
        wanted = part.lower()
        nxt = []
        for folder in folders:
            for name in _listdir(folder):
                path = os.path.join(folder, name)
                if (name.lower() not in TOOL_FOLDERS
                        and fnmatch.fnmatchcase(name.lower(), wanted)
                        and os.path.isdir(path)):
                    nxt.append(path)
        folders = nxt
    wanted = parts[-1].lower()
    found = []
    for folder in folders:
        for name in _listdir(folder):
            path = os.path.join(folder, name)
            if (fnmatch.fnmatchcase(name.lower(), wanted)
                    and not _excluded(name) and os.path.isfile(path)):
                found.append(path)
    return sorted(found, key=lambda p: (os.path.basename(p), p))


def resolve(root: str, layout: Layout = DEFAULT_LAYOUT) -> Session:
    """Find a session's frames and previews. Empty frames if it is not one."""
    root = os.path.abspath(root)
    session = Session(root=root)
    for pattern in layout.frames:
        frames = drop_stacks([p for p in match_pattern(root, pattern)
                              if p.lower().endswith((".fit", ".fits"))])
        if frames:
            session.frames = frames
            session.frame_pattern = pattern
            break
    for pattern in layout.previews:
        for path in match_pattern(root, pattern):
            if path.lower().endswith((".jpg", ".jpeg")):
                session.previews.setdefault(frame_stem(path), path)
    return session


def session_root(folder: str, layout: Layout = DEFAULT_LAYOUT) -> str:
    """The session a folder belongs to, for someone who picked 'Lights/'.

    If the parent resolves with its frames inside this very folder, the
    parent is the session -- that is where the previews and REVIEW/ are.
    """
    folder = os.path.abspath(folder)
    parent = os.path.dirname(folder)
    if parent and parent != folder:
        candidate = resolve(parent, layout)
        if candidate.frames and os.path.dirname(candidate.frames[0]) == folder:
            return parent
    return folder


def find_sessions(parent: str, layout: Layout = DEFAULT_LAYOUT,
                  max_depth: int = 3) -> list[Session]:
    """Every session under a folder -- the folder itself included.

    Does not descend into a session once found, so a 'Lights/' folder is
    never reported as a session of its own.
    """
    found: list[Session] = []

    def walk(folder: str, depth: int) -> None:
        session = resolve(folder, layout)
        if session.frames:
            found.append(session)
            return
        if depth >= max_depth:
            return
        for name in _listdir(folder):
            if name.startswith(".") or name.lower() in SKIPPED_FOLDERS:
                continue
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                walk(path, depth + 1)

    walk(os.path.abspath(parent), 0)
    return sorted(found, key=lambda s: s.root.lower())


# ----------------------------------------------------------------------
#  Saved preferences
# ----------------------------------------------------------------------

def config_path() -> str:
    """Beside the tools, so a copied folder carries its owner's layout."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        CONFIG_FILENAME)


def load_config() -> dict:
    try:
        with open(config_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(config: dict) -> None:
    try:
        with open(config_path(), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
    except OSError:
        pass  # a preference that cannot be saved is not worth a crash


def configured_layout(config: dict | None = None) -> Layout:
    """The layout the user saved, or the default for what they left out."""
    data = (config if config is not None else load_config()).get("layout") or {}
    frames = tuple(data.get("frames") or ()) or DEFAULT_FRAME_PATTERNS
    previews = tuple(data.get("previews") or ()) or DEFAULT_PREVIEW_PATTERNS
    return Layout(frames=frames, previews=previews)


def remember_folder(config: dict, folder: str, limit: int = 8) -> None:
    recent = [f for f in config.get("recent", []) if f != folder]
    config["recent"] = [folder] + recent[:limit - 1]
