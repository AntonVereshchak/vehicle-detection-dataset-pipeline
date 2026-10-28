"""Frame Selector — interactive manual curation for Frame Extraction v2.

Install: python -m pip install pillow
Run:     python frame_selector.py

Selections are saved to dataset/candidates/frame_selections.json. Originals are never edited.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk, ImageOps

ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "dataset" / "candidates"
SELECTIONS = CANDIDATES / "frame_selections.json"
EXPORT = ROOT / "dataset" / "selected"
TARGETS = {"train_A": 90, "train_B": 20, "train_C": 50, "train_D": 50}
PAGE_SIZE = 20
THUMB_SIZE = (208, 125)
SUPPORTED = {".jpg", ".jpeg", ".png"}


def load_saved():
    if not SELECTIONS.exists():
        return set()
    with SELECTIONS.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("selected", []), list):
        raise ValueError("Invalid selections file format")
    return set(str(x) for x in payload["selected"])


def save_atomic(selected):
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".frame_selections_", suffix=".tmp", dir=CANDIDATES)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "selected": sorted(selected)}, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, SELECTIONS)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class Selector:
    def __init__(self, window):
        self.window = window
        window.title("Frame Selector | Test fourth law")
        window.geometry("1190x810")
        window.minsize(950, 650)
        self.groups = {g: sorted(p for p in (CANDIDATES / g).iterdir()
                                 if p.is_file() and p.suffix.lower() in SUPPORTED)
                       if (CANDIDATES / g).is_dir() else [] for g in TARGETS}
        if not any(self.groups.values()):
            messagebox.showerror("No frames", f"No JPG/PNG images found in:\n{CANDIDATES}")
            window.destroy()
            return
        try:
            self.selected = load_saved()
        except (OSError, ValueError, json.JSONDecodeError) as e:
            messagebox.showerror("Selection file error", f"Cannot read saved selections:\n{e}\n\nFile was NOT overwritten.")
            window.destroy()
            return
        self.group = tk.StringVar(value=next(g for g, files in self.groups.items() if files))
        self.page = 0
        self.photo_refs = []
        self.pending_click = None
        self.build_ui()
        self.render()
        window.bind("<Left>", lambda _: self.move_page(-1))
        window.bind("<Right>", lambda _: self.move_page(1))

    def build_ui(self):
        header = ttk.Frame(self.window, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="Video:", font=("Segoe UI", 11)).pack(side="left")
        for group in TARGETS:
            ttk.Radiobutton(header, text=group.replace("train_", "Video "), variable=self.group,
                            value=group, command=self.change_group).pack(side="left", padx=8)
        self.status = ttk.Label(self.window, padding=(14, 0), font=("Segoe UI", 11))
        self.status.pack(anchor="w")
        self.help = ttk.Label(self.window, text="Click a thumbnail to select/unselect · Double-click to open zoom viewer · Left/Right arrows to change pages",
                              padding=(14, 6))
        self.help.pack(anchor="w")
        container = ttk.Frame(self.window)
        container.pack(fill="both", expand=True, padx=12, pady=5)
        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.canvas_window, width=e.width))
        footer = ttk.Frame(self.window, padding=12)
        footer.pack(fill="x")
        ttk.Button(footer, text="◀ Previous", command=lambda: self.move_page(-1)).pack(side="left")
        self.page_label = ttk.Label(footer, width=16, anchor="center")
        self.page_label.pack(side="left", padx=8)
        ttk.Button(footer, text="Next ▶", command=lambda: self.move_page(1)).pack(side="left")
        ttk.Button(footer, text="Export selected frames", command=self.export).pack(side="right", padx=5)
        ttk.Button(footer, text="Save selections", command=self.save).pack(side="right", padx=5)

    def key(self, path):
        return path.relative_to(CANDIDATES).as_posix()

    def change_group(self):
        self.page = 0
        self.render()

    def move_page(self, delta):
        max_page = max(0, (len(self.groups[self.group.get()]) - 1) // PAGE_SIZE)
        new_page = min(max(0, self.page + delta), max_page)
        if new_page != self.page:
            self.page = new_page
            self.render()

    def single_click(self, path):
        # Delay the selection toggle so a double-click cannot toggle twice.
        self.cancel_pending_click()
        self.pending_click = self.window.after(280, lambda p=path: self._finish_click(p))

    def _finish_click(self, path):
        self.pending_click = None
        self.toggle(path)

    def cancel_pending_click(self):
        if self.pending_click is not None:
            self.window.after_cancel(self.pending_click)
            self.pending_click = None

    def double_click(self, path):
        self.cancel_pending_click()
        self.zoom(path)

    def toggle(self, path):
        key = self.key(path)
        if key in self.selected:
            self.selected.remove(key)
        else:
            self.selected.add(key)
        self.save(silent=True)
        self.render()

    def save(self, silent=False):
        try:
            save_atomic(self.selected)
            if not silent:
                messagebox.showinfo("Saved", f"Selections saved to:\n{SELECTIONS}")
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    def render(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.photo_refs.clear()
        group = self.group.get()
        files = self.groups[group]
        start = self.page * PAGE_SIZE
        page_files = files[start:start + PAGE_SIZE]
        count = sum(self.key(p) in self.selected for p in files)
        total = sum(sum(self.key(p) in self.selected for p in group_files) for group_files in self.groups.values())
        self.status.config(text=f"{group}: selected {count} / target {TARGETS[group]}  |  candidates {len(files)}  |  total selected {total} / 210")
        self.page_label.config(text=f"Page {self.page + 1} / {max(1, (len(files) + PAGE_SIZE - 1) // PAGE_SIZE)}")
        for i, path in enumerate(page_files):
            chosen = self.key(path) in self.selected
            tile = tk.Frame(self.grid_frame, bg="#b7e8cb" if chosen else "#e5e7eb", bd=3, relief="flat")
            tile.grid(row=i // 4, column=i % 4, padx=7, pady=7, sticky="n")
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
                    display = Image.new("RGB", THUMB_SIZE, "#20262d")
                    display.paste(im.convert("RGB"), ((THUMB_SIZE[0] - im.width) // 2,
                                                       (THUMB_SIZE[1] - im.height) // 2))
                    photo = ImageTk.PhotoImage(display)
                self.photo_refs.append(photo)
                label = tk.Label(tile, image=photo, bg=tile["bg"], cursor="hand2")
            except (OSError, ValueError):
                label = tk.Label(tile, text="Cannot open image", width=28, height=7)
            label.pack(padx=4, pady=4)
            caption = tk.Label(tile, text=("✓ " if chosen else "□ ") + path.stem,
                               bg=tile["bg"], font=("Segoe UI", 9), cursor="hand2")
            caption.pack(pady=(0, 5))
            for widget in (tile, label, caption):
                widget.bind("<Button-1>", lambda e, p=path: self.single_click(p))
                widget.bind("<Double-Button-1>", lambda e, p=path: self.double_click(p))
        self.canvas.yview_moveto(0)

    def zoom(self, path):
        popup = tk.Toplevel(self.window)
        popup.title(path.name + " — mouse wheel: zoom | drag: pan | Space: select | Esc: close")
        popup.geometry("1150x800")
        with Image.open(path) as original:
            img = ImageOps.exif_transpose(original).convert("RGB")

        toolbar = ttk.Frame(popup, padding=6)
        toolbar.pack(fill="x")
        canvas = tk.Canvas(popup, bg="#151515", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        state = {"scale": None, "x": 0.0, "y": 0.0, "photo": None, "drag": None}

        def fit():
            w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
            state["scale"] = min(w / img.width, h / img.height)
            state["x"], state["y"] = w / 2, h / 2
            redraw()

        def redraw():
            if state["scale"] is None:
                return
            w = max(1, round(img.width * state["scale"]))
            h = max(1, round(img.height * state["scale"]))
            # Render only the visible region to avoid enormous temporary images at high zoom.
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            left = state["x"] - w / 2
            top = state["y"] - h / 2
            x0, y0 = max(0, int(-left)), max(0, int(-top))
            x1, y1 = min(w, int(cw - left) + 1), min(h, int(ch - top) + 1)
            canvas.delete("all")
            if x1 <= x0 or y1 <= y0:
                return
            box = (x0 / state["scale"], y0 / state["scale"],
                   x1 / state["scale"], y1 / state["scale"])
            cropped = img.crop(box).resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
            state["photo"] = ImageTk.PhotoImage(cropped)
            canvas.create_image(left + x0, top + y0, anchor="nw", image=state["photo"])
            zoom_label.configure(text=f"{state['scale'] * 100:.0f}%")

        def wheel(event):
            factor = 1.2 if event.delta > 0 else 1 / 1.2
            old = state["scale"]
            new = min(8.0, max(0.03, old * factor))
            if new == old:
                return
            # Keep the pixel under the mouse cursor fixed when zooming.
            state["x"] = event.x - (event.x - state["x"]) * new / old
            state["y"] = event.y - (event.y - state["y"]) * new / old
            state["scale"] = new
            redraw()

        def drag_start(event):
            state["drag"] = (event.x, event.y)

        def drag_move(event):
            if state["drag"] is None:
                return
            px, py = state["drag"]
            state["x"] += event.x - px
            state["y"] += event.y - py
            state["drag"] = (event.x, event.y)
            redraw()

        def toggle_in_popup(event=None):
            self.toggle(path)
            chosen = self.key(path) in self.selected
            select_label.configure(text="Selected ✓" if chosen else "Not selected")

        ttk.Button(toolbar, text="Fit to window", command=fit).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Select / Unselect", command=toggle_in_popup).pack(side="left", padx=4)
        select_label = ttk.Label(toolbar, text="Selected ✓" if self.key(path) in self.selected else "Not selected")
        select_label.pack(side="left", padx=10)
        zoom_label = ttk.Label(toolbar, text="")
        zoom_label.pack(side="right", padx=10)
        canvas.bind("<MouseWheel>", wheel)
        canvas.bind("<ButtonPress-1>", drag_start)
        canvas.bind("<B1-Motion>", drag_move)
        canvas.bind("<ButtonRelease-1>", lambda _: state.update(drag=None))
        canvas.bind("<Configure>", lambda _: fit() if state["scale"] is None else redraw())
        popup.bind("<space>", toggle_in_popup)
        popup.bind("<Escape>", lambda _: popup.destroy())
        popup.focus_set()

    def export(self):
        valid = []
        for group, files in self.groups.items():
            valid.extend((group, p) for p in files if self.key(p) in self.selected)
        if not valid:
            messagebox.showwarning("Nothing selected", "Select some frames first.")
            return
        if not messagebox.askyesno("Export", f"Copy {len(valid)} selected frames into:\n{EXPORT}\n\nExisting files with the same names will NOT be overwritten.\nPreviously exported files will NOT be deleted."):
            return
        copied = skipped = 0
        try:
            for group, path in valid:
                target_dir = EXPORT / group
                target_dir.mkdir(parents=True, exist_ok=True)
                dest = target_dir / path.name
                if dest.exists():
                    skipped += 1
                else:
                    shutil.copy2(path, dest)
                    copied += 1
            self.save(silent=True)
            messagebox.showinfo("Export complete", f"Copied: {copied}\nAlready existed: {skipped}\n\nLocation: {EXPORT}\n\nNote: export does not remove old files. For a clean final export, use an empty export folder.")
        except OSError as e:
            messagebox.showerror("Export failed", str(e))


def main():
    app = tk.Tk()
    Selector(app)
    app.mainloop()


if __name__ == "__main__":
    main()
