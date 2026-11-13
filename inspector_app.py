# -*- coding: utf-8 -*-
"""The analysis window: pick sessions, run the detector, review the result.

Everything the command line of subframe_inspector.py does, without a
console: choose a folder, see the sessions under it, set the few options
worth setting, and run. The report the detector prints goes into the log
pane, the progress into the bar, and the buttons underneath lead to what
comes next -- the REVIEW folder, the labelling window, and the one
irreversible action, deleting flagged FITS.

The detector itself is not touched: this calls classify_session exactly
as main() does, and the deletion goes through the same two functions,
behind the same four guarantees (off unless asked, after the previews
are filed, every frame listed, the word 'delete' typed).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from dataclasses import replace

import tkinter as tk
from tkinter import messagebox, ttk

from session_browser import (APP_NAME, SessionBrowser, install_error_hooks,
                             launch_tool, open_in_explorer)
from session_layout import Session, load_config, save_config


class QueueWriter:
    """A stdout that hands every line to the window, from any thread."""

    def __init__(self, sink: queue.Queue) -> None:
        self.sink = sink

    def write(self, text: str) -> int:
        if text:
            self.sink.put(("log", text))
        return len(text)

    def flush(self) -> None:
        pass


class DeleteDialog:
    """The deletion of flagged FITS, with every guarantee the CLI gives.

    It lists every frame it covers with the categories that put it there,
    says so loudly when that is half the session or more, and stays
    disabled until the word 'delete' is typed -- not a click, which is too
    easy to give by reflex.
    """

    def __init__(self, parent: tk.Misc, session: Session,
                 categories_by_frame: dict[str, list[str]], inspector) -> None:
        self.inspector = inspector
        self.session = session
        self.categories_by_frame = categories_by_frame
        self.window = tk.Toplevel(parent)
        self.window.title(f"Delete flagged FITS — {session.name}")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.geometry("640x560")

        body = ttk.Frame(self.window, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, foreground="#b00020", wraplength=600, justify="left",
                  text="This permanently deletes the .fit files themselves, "
                       "not their previews. It cannot be undone.").pack(anchor="w")

        used = sorted({c for cs in categories_by_frame.values() for c in cs})
        picks = ttk.Frame(body)
        picks.pack(fill="x", pady=(10, 4))
        ttk.Label(picks, text="Categories:").pack(side="left")
        self.wanted = {}
        for category in used:
            var = tk.BooleanVar(value=True)
            self.wanted[category] = var
            ttk.Checkbutton(picks, text=category, variable=var,
                            command=self.refresh).pack(side="left", padx=4)

        self.listing = tk.Text(body, height=16, wrap="none")
        self.listing.pack(fill="both", expand=True)
        self.summary = ttk.Label(body, wraplength=600, justify="left")
        self.summary.pack(anchor="w", pady=(6, 0))

        confirm = ttk.Frame(body)
        confirm.pack(fill="x", pady=(10, 0))
        ttk.Label(confirm, text="Type 'delete' to confirm:").pack(side="left")
        self.typed = tk.StringVar()
        self.typed.trace_add("write", lambda *_: self.refresh())
        ttk.Entry(confirm, textvariable=self.typed, width=12).pack(side="left", padx=6)
        self.go = ttk.Button(confirm, text="Delete", command=self.delete)
        self.go.pack(side="right")
        ttk.Button(confirm, text="Cancel", command=self.window.destroy).pack(
            side="right", padx=6)
        self.refresh()

    def selection(self) -> dict[str, list[str]]:
        from subframe_inspector import select_for_deletion
        wanted = {c for c, var in self.wanted.items() if var.get()}
        return select_for_deletion(self.categories_by_frame, wanted)

    def refresh(self) -> None:
        from subframe_inspector import frame_label
        chosen = self.selection()
        self.listing.config(state="normal")
        self.listing.delete("1.0", "end")
        for path, categories in chosen.items():
            self.listing.insert("end", f"{frame_label(path):20s} {' '.join(categories)}\n")
        self.listing.config(state="disabled")
        total = len(self.session.frames)
        share = len(chosen) / total if total else 0.0
        text = f"{len(chosen)} of {total} frames would be deleted."
        if share >= 0.5:
            text += (f"  ❗ That is {share:.0%} of the session. If that is not "
                     f"what you meant, the threshold is the thing to change, "
                     f"not the answer to this question.")
        self.summary.config(text=text)
        ready = chosen and self.typed.get().strip().lower() == "delete"
        self.go.state(["!disabled"] if ready else ["disabled"])

    def delete(self) -> None:
        from subframe_inspector import delete_flagged_frames
        chosen = self.selection()
        if not chosen or self.typed.get().strip().lower() != "delete":
            return
        self.window.destroy()
        self.inspector.log(f"\n🗑️ Deleting {len(chosen)} .fit file(s)…\n")
        old = sys.stdout
        sys.stdout = QueueWriter(self.inspector.events)
        try:
            deleted, failed = delete_flagged_frames(chosen)
        finally:
            sys.stdout = old
        self.inspector.log(f"🗑️  {deleted} .fit file(s) deleted"
                           + (f", {failed} failed" if failed else "") + ".\n")
        self.inspector.forget_run(self.session.root)
        self.inspector.browser.refresh()


class Inspector:
    """The main window."""

    def __init__(self, root: tk.Tk) -> None:
        from subframe_inspector import DEFAULT_SETTINGS
        self.root = root
        self.defaults = DEFAULT_SETTINGS
        self.config = load_config()
        self.events: queue.Queue = queue.Queue()
        self.running = False
        # The verdicts of the last run, per session, for the deletion.
        # Only a run made in this window counts: deleting on the strength
        # of a report.csv written by some other run, under other settings,
        # is exactly the kind of thing the guarantees are there to stop.
        self.last_run: dict[str, dict[str, list[str]]] = {}

        root.title(APP_NAME)
        root.geometry("1100x780")
        root.minsize(820, 560)
        install_error_hooks(root)

        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)
        panes = ttk.PanedWindow(main, orient="vertical")
        panes.pack(fill="both", expand=True)

        top = ttk.Frame(panes)
        self.browser = SessionBrowser(top, self.config, on_select=self.selected,
                                      on_open=lambda s: self.run(), multiple=True)
        self.browser.pack(fill="both", expand=True)
        panes.add(top, weight=3)

        bottom = ttk.Frame(panes)
        panes.add(bottom, weight=2)
        self._build_options(bottom)
        self._build_actions(bottom)
        self._build_log(bottom)

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.selected([])
        self.root.after(100, self._drain)

    # -- widgets ---------------------------------------------------------

    def _build_options(self, parent: ttk.Frame) -> None:
        saved = self.config.get("options", {})
        box = ttk.LabelFrame(parent, text="Options", padding=8)
        box.pack(fill="x", pady=(8, 0))

        ttk.Label(box, text="Previews of flagged frames:").grid(row=0, column=0, sticky="w")
        self.mode = tk.StringVar(value=saved.get("mode", "copy"))
        modes = (("Copy into REVIEW/", "copy"), ("Move into REVIEW/", "move"),
                 ("Reports only", "report"))
        for column, (text, value) in enumerate(modes, start=1):
            ttk.Radiobutton(box, text=text, value=value,
                            variable=self.mode).grid(row=0, column=column, sticky="w", padx=6)

        ttk.Label(box, text="Trailing sensitivity:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.trailing = tk.DoubleVar(value=saved.get(
            "trailing", self.defaults.trailing_elongation_ratio))
        scale = ttk.Scale(box, from_=1.10, to=1.60, variable=self.trailing,
                          command=lambda _v: self._trailing_text())
        scale.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        self.trailing_label = ttk.Label(box, width=34)
        self.trailing_label.grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Button(box, text="Default", width=8, command=lambda: (
            self.trailing.set(self.defaults.trailing_elongation_ratio),
            self._trailing_text())).grid(row=1, column=4, pady=(6, 0))
        self._trailing_text()

        self.debug = tk.BooleanVar(value=saved.get("debug", False))
        ttk.Checkbutton(box, text="Draw what was detected on the copies "
                                  "(slower)", variable=self.debug).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(box, text="Worker processes:").grid(row=2, column=3, sticky="e", pady=(6, 0))
        self.workers = tk.StringVar(value=saved.get("workers", "Auto"))
        ttk.Combobox(box, textvariable=self.workers, width=6, state="readonly",
                     values=["Auto"] + [str(n) for n in range(1, (os.cpu_count() or 8) + 1)]
                     ).grid(row=2, column=4, sticky="w", pady=(6, 0))

        self.export = tk.BooleanVar(value=saved.get("export_cases", False))
        ttk.Checkbutton(box, text="Also export a case bundle (for tuning the "
                                  "thresholds later)", variable=self.export).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        box.columnconfigure(2, weight=1)

    def _trailing_text(self) -> None:
        value = round(self.trailing.get(), 2)
        word = ("strict" if value < 1.20 else "lenient" if value > 1.40
                else "balanced")
        self.trailing_label.config(text=f"{value:.2f}x — {word} "
                                        f"(lower catches milder trailing)")

    def _build_actions(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))
        self.run_button = ttk.Button(row, text="▶ Analyse", command=self.run)
        self.run_button.pack(side="left")
        self.review_button = ttk.Button(row, text="Open REVIEW folder",
                                        command=self.open_review)
        self.review_button.pack(side="left", padx=6)
        self.label_button = ttk.Button(row, text="Label / view frames…",
                                       command=self.label)
        self.label_button.pack(side="left")
        self.delete_button = ttk.Button(row, text="Delete flagged FITS…",
                                        command=self.delete)
        self.delete_button.pack(side="right")

        progress = ttk.Frame(parent)
        progress.pack(fill="x", pady=(8, 0))
        self.bar = ttk.Progressbar(progress, mode="determinate")
        self.bar.pack(side="left", fill="x", expand=True)
        self.status = ttk.Label(progress, width=40)
        self.status.pack(side="left", padx=(8, 0))

    def _build_log(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, pady=(8, 0))
        self.text = tk.Text(frame, height=10, wrap="word", state="disabled",
                            font=("Consolas", 9))
        scroll = ttk.Scrollbar(frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # -- state -----------------------------------------------------------

    def selected(self, sessions: list[Session]) -> None:
        idle = not self.running
        one = len(sessions) == 1
        self.run_button.state(["!disabled"] if idle and sessions else ["disabled"])
        self.label_button.state(["!disabled"] if one else ["disabled"])
        has_review = one and os.path.isdir(os.path.join(sessions[0].root, "REVIEW"))
        self.review_button.state(["!disabled"] if has_review else ["disabled"])
        can_delete = (idle and one and bool(self.last_run.get(sessions[0].root)))
        self.delete_button.state(["!disabled"] if can_delete else ["disabled"])
        self.run_button.config(text=f"▶ Analyse {len(sessions)} sessions"
                               if len(sessions) > 1 else "▶ Analyse")

    def forget_run(self, root: str) -> None:
        self.last_run.pop(root, None)

    def log(self, text: str) -> None:
        self.text.config(state="normal")
        self.text.insert("end", text)
        self.text.see("end")
        self.text.config(state="disabled")

    def save_options(self) -> None:
        self.config["options"] = {
            "mode": self.mode.get(), "trailing": round(self.trailing.get(), 2),
            "debug": self.debug.get(), "workers": self.workers.get(),
            "export_cases": self.export.get()}
        save_config(self.config)

    # -- actions ---------------------------------------------------------

    def run(self) -> None:
        sessions = self.browser.selected()
        if self.running or not sessions:
            return
        mode = self.mode.get()
        if mode == "move" and not messagebox.askyesno(
                APP_NAME, "Move mode takes the previews of flagged frames out "
                          "of their folder and into REVIEW/. Continue?",
                parent=self.root):
            return
        self.save_options()
        workers = self.workers.get()
        settings = replace(self.defaults,
                           trailing_elongation_ratio=round(self.trailing.get(), 2),
                           debug=self.debug.get(),
                           export_cases=self.export.get(),
                           max_workers=None if workers == "Auto" else int(workers))
        self.running = True
        self.selected(sessions)
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")
        threading.Thread(target=self._work, args=(sessions, mode, settings),
                         daemon=True).start()

    def _work(self, sessions: list[Session], mode: str, settings) -> None:
        """Runs on a thread; talks to the window only through the queue."""
        from subframe_inspector import classify_session
        old = sys.stdout
        sys.stdout = QueueWriter(self.events)
        try:
            for number, session in enumerate(sessions, start=1):
                self.events.put(("session", (number, len(sessions), session.name)))
                print(f"📂 {session.root}\n   {len(session.frames)} frames, "
                      f"{len(session.previews)} JPG previews\n")
                try:
                    result = classify_session(
                        session, mode=None if mode == "report" else mode,
                        settings=settings,
                        progress=lambda done, total: self.events.put(
                            ("progress", (done, total))))
                except Exception as error:  # one bad night must not stop the rest
                    print(f"\n⚠️ {session.name} failed: {error}\n")
                    result = None
                self.events.put(("done-session", (session.root, result, mode)))
                print("\n")
        finally:
            sys.stdout = old
            self.events.put(("finished", None))

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "progress":
                    done, total = payload
                    self.bar.config(maximum=total, value=done)
                    self.status.config(text=f"{self._current} — {done}/{total} frames")
                elif kind == "session":
                    number, count, name = payload
                    self._current = (f"{name} ({number}/{count})" if count > 1 else name)
                    self.bar.config(value=0)
                    self.status.config(text=f"{self._current} — reading…")
                elif kind == "done-session":
                    root, result, mode = payload
                    # Deletion is offered only after a run that filed the
                    # previews, as on the command line.
                    if result is not None and mode != "report":
                        self.last_run[root] = result
                    else:
                        self.last_run.pop(root, None)
                    self.browser.update_session(root)
                elif kind == "finished":
                    self.running = False
                    self.status.config(text="Done.")
                    self.selected(self.browser.selected())
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    _current = ""

    def open_review(self) -> None:
        chosen = self.browser.selected()
        if chosen:
            path = os.path.join(chosen[0].root, "REVIEW")
            if os.path.isdir(path):
                open_in_explorer(path)

    def label(self) -> None:
        chosen = self.browser.selected()
        if chosen:
            launch_tool("label_frames.py", chosen[0].root)

    def delete(self) -> None:
        chosen = self.browser.selected()
        if not chosen or self.running:
            return
        result = self.last_run.get(chosen[0].root)
        if not result:
            messagebox.showinfo(APP_NAME, "Nothing flagged in the last run.",
                                parent=self.root)
            return
        DeleteDialog(self.root, chosen[0], result, self)

    def close(self) -> None:
        if self.running and not messagebox.askyesno(
                APP_NAME, "An analysis is still running. Quit anyway?",
                parent=self.root):
            return
        self.save_options()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        from astropy.io import fits  # noqa: F401
    except ImportError as error:
        root.withdraw()
        messagebox.showerror(APP_NAME, f"A required package is missing: {error}\n\n"
                                       f"Run setup.bat again to install it.")
        return 1
    Inspector(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    # Windows spawns the worker processes rather than forking them, and
    # each one re-imports this file: without the guard, every worker
    # would open a window of its own.
    raise SystemExit(main())
