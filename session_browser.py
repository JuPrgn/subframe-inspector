# -*- coding: utf-8 -*-
"""The session list both windows open on, and what they share besides.

Pick a folder -- one night, or a whole drive of them -- and every session
under it is listed with what is known about it: how many frames, whether
it has previews, whether the detector has been run on it and what it
flagged, how much of it has been labelled. The folder layout those
sessions are read with can be changed from here, and is saved beside the
tools.

tkinter and the standard library only, like the labelling window.
"""

from __future__ import annotations

import csv
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from session_layout import (DEFAULT_LAYOUT, Layout, Session,
                            configured_layout, find_sessions,
                            remember_folder, resolve, save_config,
                            session_root)

APP_NAME = "Subframe Inspector"
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
#  Running without a console
# ----------------------------------------------------------------------

def log_path() -> str:
    return os.path.join(HERE, "subframe_inspector_errors.log")


def install_error_hooks(root: tk.Tk) -> None:
    """Make an exception visible when there is no console to print it.

    Launched through pythonw, a traceback goes nowhere and the window
    simply stops responding. So it is written to a log beside the tools
    and shown in a dialog that says where the log is.
    """
    def report(kind, value, trace) -> None:
        text = "".join(traceback.format_exception(kind, value, trace))
        try:
            with open(log_path(), "a", encoding="utf-8") as handle:
                handle.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}")
        except OSError:
            pass
        try:
            messagebox.showerror(APP_NAME, f"{value}\n\nDetails written to:\n"
                                           f"{log_path()}", parent=root)
        except tk.TclError:
            pass

    root.report_callback_exception = report
    sys.excepthook = report


def open_in_explorer(path: str) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 -- a folder the user asked to see
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def launch_tool(script: str, *arguments: str) -> None:
    """Start one of the other tools in its own process, with no console."""
    executable = sys.executable
    if sys.platform.startswith("win"):
        windowed = os.path.join(os.path.dirname(executable), "pythonw.exe")
        if os.path.isfile(windowed):
            executable = windowed
    subprocess.Popen([executable, os.path.join(HERE, script), *arguments],
                     cwd=HERE)


# ----------------------------------------------------------------------
#  What is known about a session
# ----------------------------------------------------------------------

def session_status(session: Session) -> dict[str, str]:
    """The columns of the list: cheap facts read off the session folder."""
    report = os.path.join(session.root, "REVIEW", "report.csv")
    analysed, flagged = "", ""
    if os.path.isfile(report):
        analysed = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(os.path.getmtime(report)))
        try:
            with open(report, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            flagged = str(sum(1 for row in rows if (row.get("categories") or "").strip()))
        except (OSError, csv.Error):
            flagged = "?"
    labelled = ""
    labels = os.path.join(session.root, "labels.csv")
    if os.path.isfile(labels):
        try:
            with open(labels, newline="", encoding="utf-8") as handle:
                labelled = str(sum(1 for _ in csv.DictReader(handle)))
        except (OSError, csv.Error):
            labelled = "?"
    return {"frames": str(len(session.frames)),
            "previews": str(len(session.previews)) if session.previews else "—",
            "analysed": analysed or "—",
            "flagged": flagged or "—",
            "labelled": labelled or "—"}


# ----------------------------------------------------------------------
#  Folder layout dialog
# ----------------------------------------------------------------------

class LayoutDialog:
    """Edit the relative patterns sessions are read with.

    Shows, live, what the patterns find in a folder of the user's
    choosing -- a layout that cannot be checked before it is saved is a
    layout that will be wrong.
    """

    def __init__(self, parent: tk.Misc, config: dict, sample: str | None,
                 on_save: Callable[[], None]) -> None:
        self.config = config
        self.on_save = on_save
        self.sample = sample
        layout = configured_layout(config)

        self.window = tk.Toplevel(parent)
        self.window.title("Folder layout")
        self.window.transient(parent)
        self.window.grab_set()
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, wraplength=520, justify="left", text=(
            "Where a session keeps its files, as patterns relative to the "
            "session folder, one per line. '*' matches anything; case does "
            "not matter. REVIEW/ and labels.csv are written in the session "
            "folder itself.")).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="Light frames (first pattern that finds "
                              "anything wins):").grid(row=1, column=0, sticky="w",
                                                      pady=(10, 0))
        self.frames_text = tk.Text(frame, height=4, width=40)
        self.frames_text.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        self.frames_text.insert("1.0", "\n".join(layout.frames))

        ttk.Label(frame, text="JPG previews (all searched, optional):").grid(
            row=1, column=1, sticky="w", pady=(10, 0))
        self.previews_text = tk.Text(frame, height=4, width=40)
        self.previews_text.grid(row=2, column=1, sticky="nsew")
        self.previews_text.insert("1.0", "\n".join(layout.previews))

        test_row = ttk.Frame(frame)
        test_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(test_row, text="Test on a session…",
                   command=self.choose_sample).pack(side="left")
        self.result = ttk.Label(test_row, text="", wraplength=420, justify="left")
        self.result.pack(side="left", padx=8)

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Defaults", command=self.defaults).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self.window.destroy).pack(
            side="left", padx=6)
        ttk.Button(buttons, text="Save", command=self.save).pack(side="left")

        for text in (self.frames_text, self.previews_text):
            text.bind("<KeyRelease>", lambda _e: self.test())
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self.test()

    def layout(self) -> Layout:
        frames = Layout.parse(self.frames_text.get("1.0", "end"))
        previews = Layout.parse(self.previews_text.get("1.0", "end"))
        return Layout(frames=frames or DEFAULT_LAYOUT.frames,
                      previews=previews or DEFAULT_LAYOUT.previews)

    def choose_sample(self) -> None:
        folder = filedialog.askdirectory(parent=self.window,
                                         title="A session folder to test on")
        if folder:
            self.sample = folder
            self.test()

    def test(self) -> None:
        if not self.sample:
            self.result.config(text="Pick a session to see what these find.")
            return
        session = resolve(self.sample, self.layout())
        if not session.frames:
            self.result.config(text=f"{os.path.basename(self.sample)}: "
                                    f"no light frames found.")
            return
        matched = sum(1 for f in session.frames if session.preview_for(f))
        self.result.config(text=(
            f"{os.path.basename(self.sample)}: {len(session.frames)} frames "
            f"via '{session.frame_pattern}', {matched} with a preview."))

    def defaults(self) -> None:
        for text, values in ((self.frames_text, DEFAULT_LAYOUT.frames),
                             (self.previews_text, DEFAULT_LAYOUT.previews)):
            text.delete("1.0", "end")
            text.insert("1.0", "\n".join(values))
        self.test()

    def save(self) -> None:
        layout = self.layout()
        if layout == DEFAULT_LAYOUT:
            self.config.pop("layout", None)
        else:
            self.config["layout"] = {"frames": list(layout.frames),
                                     "previews": list(layout.previews)}
        save_config(self.config)
        self.window.destroy()
        self.on_save()


# ----------------------------------------------------------------------
#  The session list
# ----------------------------------------------------------------------

COLUMNS = (("frames", "Frames", 70), ("previews", "JPG", 60),
           ("analysed", "Analysed", 130), ("flagged", "Flagged", 70),
           ("labelled", "Labelled", 70))


class SessionBrowser(ttk.Frame):
    """A folder bar and the sessions found under it.

    ``on_open`` is called with a session on double-click or Enter;
    ``on_select`` whenever the selection changes.
    """

    def __init__(self, parent: tk.Misc, config: dict,
                 on_open: Callable[[Session], None] | None = None,
                 on_select: Callable[[list[Session]], None] | None = None,
                 multiple: bool = False) -> None:
        super().__init__(parent)
        self.config_data = config
        self.on_open = on_open
        self.on_select = on_select
        self.sessions: dict[str, Session] = {}
        self.results: queue.Queue = queue.Queue()
        self.search_id = 0

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Label(bar, text="Folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        self.folder_box = ttk.Combobox(bar, textvariable=self.folder_var,
                                       values=config.get("recent", []))
        self.folder_box.pack(side="left", fill="x", expand=True, padx=4)
        self.folder_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        self.folder_box.bind("<Return>", lambda _e: self.refresh())
        ttk.Button(bar, text="Browse…", command=self.browse).pack(side="left")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(bar, text="Folder layout…", command=self.edit_layout).pack(side="left")

        table = ttk.Frame(self)
        table.pack(fill="both", expand=True, pady=(6, 0))
        self.tree = ttk.Treeview(table, columns=[c[0] for c in COLUMNS],
                                 selectmode="extended" if multiple else "browse")
        self.tree.heading("#0", text="Session")
        self.tree.column("#0", width=320, stretch=True)
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, stretch=False, anchor="center")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._opened)
        self.tree.bind("<Return>", self._opened)
        self.tree.bind("<<TreeviewSelect>>", self._selected)

        self.status = ttk.Label(self, text="")
        self.status.pack(fill="x", pady=(4, 0))

        recent = config.get("recent") or []
        if recent and os.path.isdir(recent[0]):
            self.folder_var.set(recent[0])
            self.after(50, self.refresh)
        else:
            self.status.config(text="Choose a folder: one night, or a folder "
                                    "holding many of them.")

    # -- folder ----------------------------------------------------------

    def browse(self) -> None:
        start = self.folder_var.get() or os.path.expanduser("~")
        folder = filedialog.askdirectory(parent=self, initialdir=start,
                                         title="A session, or a folder of sessions")
        if folder:
            self.folder_var.set(os.path.normpath(folder))
            self.refresh()

    def edit_layout(self) -> None:
        sample = self.selected()[0].root if self.selected() else None
        LayoutDialog(self, self.config_data, sample, self.refresh)

    def refresh(self) -> None:
        """Search the folder in the background; a drive can take seconds."""
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            self.status.config(text=f"Not a folder: {folder}" if folder else "")
            return
        remember_folder(self.config_data, os.path.normpath(folder))
        save_config(self.config_data)
        self.folder_box.config(values=self.config_data.get("recent", []))

        self.search_id += 1
        search = self.search_id
        layout = configured_layout(self.config_data)
        self.status.config(text=f"Searching {folder} …")

        def work() -> None:
            start = session_root(folder, layout)
            sessions = find_sessions(start, layout)
            rows = [(s, session_status(s)) for s in sessions]
            self.results.put((search, folder, rows))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self) -> None:
        try:
            search, folder, rows = self.results.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return
        if search != self.search_id:
            return
        self._fill(folder, rows)

    def _fill(self, folder: str, rows) -> None:
        keep = {s.root for s in self.selected()}
        self.tree.delete(*self.tree.get_children())
        self.sessions = {}
        base = os.path.abspath(folder)
        for session, status in rows:
            name = os.path.relpath(session.root, base)
            name = session.name if name == "." else name
            self.sessions[session.root] = session
            self.tree.insert("", "end", iid=session.root, text=name,
                             values=[status[c[0]] for c in COLUMNS])
        if not rows:
            self.status.config(text=f"No light frames found under {folder}. "
                                    f"Check 'Folder layout…'.")
        else:
            total = sum(len(s.frames) for s, _ in rows)
            self.status.config(text=f"{len(rows)} session(s), {total} frames.")
        reselect = [r for r in keep if r in self.sessions]
        if reselect:
            self.tree.selection_set(reselect)
        elif len(rows) == 1:
            self.tree.selection_set(rows[0][0].root)
        self._selected()

    def update_session(self, root: str) -> None:
        """Re-read one row, after a run or a labelling pass changed it."""
        if root not in self.sessions:
            return
        session = resolve(root, configured_layout(self.config_data))
        self.sessions[root] = session
        status = session_status(session)
        self.tree.item(root, values=[status[c[0]] for c in COLUMNS])

    # -- selection -------------------------------------------------------

    def selected(self) -> list[Session]:
        return [self.sessions[i] for i in self.tree.selection() if i in self.sessions]

    def _selected(self, _event=None) -> None:
        if self.on_select:
            self.on_select(self.selected())

    def _opened(self, _event=None) -> None:
        chosen = self.selected()
        if chosen and self.on_open:
            self.on_open(chosen[0])
