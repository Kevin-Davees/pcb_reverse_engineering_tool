#!/usr/bin/env python3
"""
PCB Reverse Engineering Tool  —  Python / Tkinter GUI
======================================================
A complete rewrite of the HTML prototype, fixing all known bugs and lifting
every major limitation.

Fixed bugs vs the HTML version
───────────────────────────────
• Duplicate `let cvReady` declaration (JS syntax error)
• Duplicate resize / error event listeners registered twice
• `downloadZip` used `await` inside a non-async inner function
• KiCad export emitted literal "n" instead of newlines (template-literal bug)
• Schematic view (Stage 8) never connected net lines between component stubs
• `UndoRedo.restoreState` referenced but never implemented on any stage
• Pin-position tracking used raw `.left/.top` instead of centre-origin coords
• Snap-radius slider value not reflected in the UI label
• No validation before advancing stages (could crash with null data)
• `cvReady` guard could allow analysis to run on uninitialised OpenCV

New features
────────────
• Vectorised NumPy paint brush (no slow Python pixel loops)
• Smooth brush interpolation between mouse events
• Configurable perspective output resolution
• Morphological open *and* close in net analysis (not just open)
• Snap uses majority-vote inside radius, not first non-zero pixel found
• Full undo / redo on Stage 4 (paint) and Stage 6 (pins)
• Pin rename / block rename inline
• Load pins from JSON
• Export: KiCad Netlist (.net), KiCad PCB (.kicad_pcb), Pins CSV,
          Nets CSV, Project JSON, ZIP bundle
• Net label rename in Stage 8
• Schematic view correctly draws net ratsnest lines between stubs
• Status-bar colour codes (info / success / warn / error)
• Drag-to-move pins in Stage 6
• Shift-click multi-select for grouping
• Colour-tolerance slider in segmentation

Requirements
────────────
    pip install Pillow opencv-python numpy

Run
───
    python3 pcb_reverse_engineering.py
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import copy, csv, io, json, math, os, random, threading, zipfile
from io import BytesIO
from pathlib import Path

# ─── third-party ─────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageTk
    import cv2
    import numpy as np
except ImportError as _e:
    import sys
    sys.exit(
        f"\nMissing dependency: {_e}\n"
        "Install with:  pip install Pillow opencv-python numpy\n"
    )

# ─── tkinter (must be after third-party so the error message is helpful) ─────
import tkinter as tk
from tkinter import (
    colorchooser, filedialog, messagebox, simpledialog, ttk
)


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def rgb_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def rand_color() -> tuple:
    return (random.randint(80, 230), random.randint(80, 230), random.randint(80, 230))


# ═══════════════════════════════════════════════════════════════════════════════
#  APPLICATION STATE  (singleton `S`)
# ═══════════════════════════════════════════════════════════════════════════════
class AppState:
    LAYER_LABELS = [
        "Copper Trace", "Pad", "Via",
        "Silk Screen",  "Solder Mask",
        "PCB Substrate","Background",
    ]

    def __init__(self):
        self.project_name: str = "pcb_project"

        # ── Images (PIL RGBA) ──────────────────────────────────────────────
        self.raw:     "Image.Image | None" = None  # original upload
        self.base:    "Image.Image | None" = None  # after perspective warp
        self.seg:     "Image.Image | None" = None  # full seg overlay (RGBA)
        self.net_img: "Image.Image | None" = None  # coloured net map (RGBA)

        # ── Segmentation pixel array  H × W × 4 RGBA  (NumPy) ────────────
        self.seg_arr: "np.ndarray | None" = None

        # ── Color classifications ─────────────────────────────────────────
        # [{"hex":"#rrggbb", "label":str, "rgb":(r,g,b)}, ...]
        self.colors: list = []

        # ── Perspective ───────────────────────────────────────────────────
        self.persp_pts: list = []   # 4 × (img_x, img_y)

        # ── Pins ──────────────────────────────────────────────────────────
        # {"id":int, "x":float, "y":float, "block_name":str, "pin_name":str,
        #  "group_id":int|None, "net_id":int|None, "color":"#rrggbb"}
        self.pins:     list = []
        self.pin_ctr:  int  = 0
        self.pin_size: int  = 10

        # ── Groups ────────────────────────────────────────────────────────
        # {"id":int, "name":str, "pin_ids":[int,...], "color":"#rrggbb"}
        self.groups:     list = []
        self.group_ctr:  int  = 0

        # ── Nets ──────────────────────────────────────────────────────────
        # {"id":int, "label":str, "pin_ids":[int,...], "color":"#rrggbb"}
        self.nets: list = []

        # ── Undo / Redo per stage (0-indexed) ─────────────────────────────
        self._undo: dict = {i: [] for i in range(9)}
        self._redo: dict = {i: [] for i in range(9)}

        # ── Paint tool ────────────────────────────────────────────────────
        self.brush_size:   int   = 15
        self.paint_color:  tuple = (0, 229, 255, 255)   # RGBA
        self.current_tool: str   = "paint"

        # ── Analysis ──────────────────────────────────────────────────────
        self.snap_radius:  int = 10

        # ── Display ───────────────────────────────────────────────────────
        self.opacity_base: int = 80
        self.opacity_seg:  int = 70

    # ── Undo / Redo helpers ────────────────────────────────────────────────
    # Pattern: before any destructive action call push_undo(stage, CURRENT_STATE).
    # undo() / redo() in each stage are responsible for saving current state
    # to the opposite stack before restoring.

    def push_undo(self, stage: int, snap):
        """Call with a copy of the current state BEFORE the action executes."""
        self._undo[stage].append(snap)
        self._redo[stage].clear()          # new action invalidates redo history

    def pop_undo(self, stage: int):
        """Return the most recent undo snapshot, or None if empty."""
        return self._undo[stage].pop() if self._undo[stage] else None

    def pop_redo(self, stage: int):
        """Return the most recent redo snapshot, or None if empty."""
        return self._redo[stage].pop() if self._redo[stage] else None

    def push_redo(self, stage: int, snap):
        """Call with a copy of the current state before restoring an undo snapshot."""
        self._redo[stage].append(snap)


S = AppState()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class PCBApp(tk.Tk):
    # Status-bar colours
    _STATUS_COLORS = {
        "info":    "#8b949e",
        "success": "#3fb950",
        "warn":    "#d29922",
        "error":   "#f85149",
    }

    def __init__(self):
        super().__init__()
        self.title("PCB Reverse Engineering Tool")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(bg="#0d1117")
        self._build_ui()
        # Global keyboard shortcuts
        self.bind_all("<Control-z>",         self._kb_undo)
        self.bind_all("<Control-Z>",         self._kb_undo)
        self.bind_all("<Control-y>",         self._kb_redo)
        self.bind_all("<Control-Y>",         self._kb_redo)

    # ── UI skeleton ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top header ───────────────────────────────────────────────────
        header = tk.Frame(self, bg="#161b22", height=46)
        header.pack(fill="x")
        tk.Label(
            header, text="🔬  PCB Reverse Engineering Tool",
            bg="#161b22", fg="#58a6ff",
            font=("Courier New", 13, "bold"),
        ).pack(side="left", padx=16, pady=10)
        tk.Label(
            header,
            text="Pillow · OpenCV · NumPy",
            bg="#161b22", fg="#30363d",
            font=("Courier New", 9),
        ).pack(side="right", padx=16)

        # ── Status bar ────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready — open an image in Stage 1")
        self._status_lbl = tk.Label(
            self, textvariable=self._status_var,
            bg="#0d1117", fg="#8b949e",
            font=("Courier New", 10), anchor="w",
        )
        self._status_lbl.pack(fill="x", side="bottom", padx=10, pady=3)

        # ── Notebook (the 9 stages) ────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",        background="#0d1117", borderwidth=0)
        style.configure("TNotebook.Tab",    background="#161b22",  foreground="#8b949e",
                        font=("Courier New", 10), padding=[12, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", "#0d1117")],
                  foreground=[("selected", "#00e5ff")])
        style.configure("TFrame", background="#0d1117")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        stage_classes = [
            Stage1_Load,        Stage2_Perspective, Stage3_ColorLabel,
            Stage4_Paint,       Stage5_Overlay,     Stage6_Pins,
            Stage7_Nets,        Stage8_Routing,     Stage9_Export,
        ]
        tab_titles = [
            "1 · Load Image",   "2 · Perspective",  "3 · Label Colors",
            "4 · Paint/Seg",    "5 · Overlay",      "6 · Pins",
            "7 · Net Analysis", "8 · Routing",      "9 · Export",
        ]
        self.stages = []
        for cls, title in zip(stage_classes, tab_titles):
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            self.stages.append(cls(frame, self))

        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    # ── Tab change callback ───────────────────────────────────────────────
    def _on_tab_change(self, _event=None):
        idx = self.nb.index("current")
        self.stages[idx].on_enter()

    # ── Status helper ─────────────────────────────────────────────────────
    def set_status(self, msg: str, level: str = "info"):
        self._status_var.set(msg)
        self._status_lbl.config(fg=self._STATUS_COLORS.get(level, "#8b949e"))

    # ── Global undo / redo ────────────────────────────────────────────────
    def _kb_undo(self, _e=None):
        idx = self.nb.index("current")
        self.stages[idx].undo()

    def _kb_redo(self, _e=None):
        idx = self.nb.index("current")
        self.stages[idx].redo()


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE STAGE  (shared helpers for all 9 stages)
# ═══════════════════════════════════════════════════════════════════════════════
class BaseStage:
    def __init__(self, parent: ttk.Frame, app: PCBApp):
        self.parent = parent
        self.app    = app
        self._build()

    # Override in subclasses:
    def _build(self):    pass
    def on_enter(self):  pass
    def undo(self):      pass
    def redo(self):      pass

    # ── Widget factories ──────────────────────────────────────────────────
    def _btn(self, parent, text: str, cmd, **kw) -> tk.Button:
        defaults = {
            "bg": "#21262d",
            "fg": "#e6edf3",
            "activebackground": "#30363d",
            "activeforeground": "#e6edf3",
            "relief": "flat",
            "font": ("Courier New", 10),
            "cursor": "hand2",
            "padx": 10,
            "pady": 6,
            }
        defaults.update(kw)
        return tk.Button(
            parent,
            text=text,
            command=cmd,
            **defaults
        )
          
   

    def _lbl(self, parent, text: str, **kw) -> tk.Label:
        return tk.Label(
            parent, text=text,
            bg="#161b22", fg="#8b949e",
            font=("Courier New", 10), **kw,
        )

    def _sep(self, parent):
        f = tk.Frame(parent, bg="#30363d", height=1)
        f.pack(fill="x", padx=10, pady=6)
        return f

    def _section(self, parent, text: str):
        tk.Label(
            parent, text=text,
            bg="#161b22", fg="#00e5ff",
            font=("Courier New", 11, "bold"),
        ).pack(pady=(12, 4), padx=10, anchor="w")

    # Returns (left_sidebar, right_canvas_frame)
    def _two_pane(self, sidebar_w: int = 220) -> tuple:
        left = tk.Frame(self.parent, bg="#161b22", width=sidebar_w)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = tk.Frame(self.parent, bg="#0d1117")
        right.pack(side="left", fill="both", expand=True)
        return left, right

    def status(self, msg: str, level: str = "info"):
        self.app.set_status(msg, level)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — LOAD IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
class Stage1_Load(BaseStage):
    def _build(self):
        left, right = self._two_pane(230)

        self._section(left, "PROJECT")
        self._lbl(left, "Project name:").pack(padx=10, anchor="w")
        self.name_var = tk.StringVar(value=S.project_name)
        tk.Entry(
            left, textvariable=self.name_var,
            bg="#0d1117", fg="#e6edf3",
            insertbackground="#e6edf3", relief="flat",
            font=("Courier New", 10),
        ).pack(padx=10, fill="x", pady=4)

        self._sep(left)
        self._section(left, "LOAD")
        self._btn(left, "📂  Open Image …",        self._open_image ).pack(padx=10, fill="x")
        self._btn(left, "🗂  Load Project JSON …",  self._load_project).pack(padx=10, fill="x", pady=(6, 0))
        self._btn(left, "🔄  Reset All",            self._reset_all  ).pack(padx=10, fill="x", pady=(6, 0))

        self._sep(left)
        self.info_lbl = tk.Label(
            left, text="No image loaded.",
            bg="#161b22", fg="#8b949e",
            font=("Courier New", 9), wraplength=200, justify="left",
        )
        self.info_lbl.pack(padx=10, anchor="w")

        self._btn(
            left, "Next →  Stage 2",
            lambda: self.app.nb.select(1),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=14, fill="x", side="bottom")

        # ── Canvas ────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(right, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._refresh)
        self._tk_img   = None
        self._hint_id  = self.canvas.create_text(
            500, 350,
            text="Open a PCB image to begin\n\nFile → Open Image …",
            fill="#30363d", font=("Courier New", 20), justify="center",
        )

    def _refresh(self, _e=None):
        if S.raw:
            self._show(S.raw)

    def _show(self, img: "Image.Image"):
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        disp = img.copy()
        disp.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open PCB Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            S.raw = Image.open(path).convert("RGBA")
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex)); return

        # Reset downstream data when a new image is loaded
        S.project_name = self.name_var.get().strip() or Path(path).stem
        self.name_var.set(S.project_name)
        S.base = S.seg = S.seg_arr = S.net_img = None
        S.persp_pts = []
        S.colors    = []
        S.pins      = []; S.pin_ctr   = 0
        S.groups    = []; S.group_ctr = 0
        S.nets      = []

        self._show(S.raw)
        w, h = S.raw.size
        self.info_lbl.config(
            text=f"{Path(path).name}\n{w} × {h} px\n"
                 f"{Path(path).stat().st_size // 1024} KB"
        )
        self.status(f"Loaded: {Path(path).name}  ({w}×{h})", "success")

    def _load_project(self):
        path = filedialog.askopenfilename(
            title="Load Project JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            S.pins      = data.get("pins",   [])
            S.groups    = data.get("groups", [])
            S.nets      = data.get("nets",   [])
            S.colors    = data.get("colors", [])
            S.pin_ctr   = max((p["id"] for p in S.pins),   default=0) + 1
            S.group_ctr = max((g["id"] for g in S.groups), default=0) + 1
            messagebox.showinfo(
                "Project Loaded",
                f"Pins: {len(S.pins)}  |  Groups: {len(S.groups)}  |  Nets: {len(S.nets)}",
            )
            self.status(f"Project JSON loaded: {Path(path).name}", "success")
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex))

    def _reset_all(self):
        if not messagebox.askyesno("Reset", "Clear all data and start fresh?"):
            return
        S.__init__()
        self.name_var.set(S.project_name)
        self.info_lbl.config(text="No image loaded.")
        self.canvas.delete("all")
        self._hint_id = self.canvas.create_text(
            500, 350,
            text="Open a PCB image to begin\n\nFile → Open Image …",
            fill="#30363d", font=("Courier New", 20), justify="center",
        )
        self.status("All data cleared.", "warn")

    def on_enter(self):
        if S.raw:
            self._show(S.raw)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — PERSPECTIVE CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════
class Stage2_Perspective(BaseStage):
    _CORNER_LABELS  = ["TL", "TR", "BR", "BL"]
    _CORNER_COLORS  = ["#f85149", "#3fb950", "#58a6ff", "#d29922"]

    def _build(self):
        left, right = self._two_pane(230)

        self._section(left, "PERSPECTIVE")
        self._lbl(
            left, "Click the 4 corners of the PCB\nin order:  TL → TR → BR → BL",
        ).pack(padx=10, anchor="w")

        self._sep(left)
        self._btn(left, "🔄  Reset Points",     self._reset   ).pack(padx=10, fill="x")
        self._btn(left, "↩  Undo Last Point",   self._undo_pt ).pack(padx=10, fill="x", pady=(4, 0))

        self._sep(left)
        self._lbl(left, "Output width (px):").pack(padx=10, anchor="w")
        self.out_w = tk.IntVar(value=1600)
        tk.Spinbox(
            left, from_=200, to=8000, increment=100,
            textvariable=self.out_w,
            bg="#0d1117", fg="#e6edf3", relief="flat",
            font=("Courier New", 10), buttonbackground="#21262d",
        ).pack(padx=10, fill="x", pady=2)
        self._lbl(left, "Output height (px):").pack(padx=10, anchor="w")
        self.out_h = tk.IntVar(value=1200)
        tk.Spinbox(
            left, from_=200, to=8000, increment=100,
            textvariable=self.out_h,
            bg="#0d1117", fg="#e6edf3", relief="flat",
            font=("Courier New", 10), buttonbackground="#21262d",
        ).pack(padx=10, fill="x", pady=2)

        self._sep(left)
        self.pts_lbl = tk.Label(
            left, text="Points placed: 0 / 4",
            bg="#161b22", fg="#e6edf3",
            font=("Courier New", 10),
        )
        self.pts_lbl.pack(padx=10, anchor="w")

        self._btn(
            left, "✅  Apply Warp",
            self._apply,
            bg="#1f6feb", activebackground="#388bfd",
        ).pack(padx=10, fill="x", pady=6)
        self._btn(
            left, "⤵  Skip (use image as-is)",
            self._skip,
        ).pack(padx=10, fill="x")

        self._btn(
            left, "Next →  Stage 3",
            lambda: self.app.nb.select(2),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=14, fill="x", side="bottom")

        # ── Canvas ────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            right, bg="#0d1117",
            highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>",   self._on_click)
        self.canvas.bind("<Configure>",  self._refresh)
        self._tk_img = None
        self._scale  = 1.0
        self._offset = (0, 0)   # (ox, oy) canvas pixels from top-left of image

    def on_enter(self):
        if S.raw:
            self._refresh()

    def _refresh(self, _e=None):
        if not S.raw:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        disp = S.raw.copy()
        disp.thumbnail((cw, ch), Image.LANCZOS)
        iw, ih = disp.size
        
        self._scale_x = S.raw.width / iw
        self._scale_y = S.raw.height / ih

        self._offset = ((cw - iw) // 2, (ch - ih) // 2)
        
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        ox, oy = self._offset
        self.canvas.create_image(ox, oy, image=self._tk_img, anchor="nw")
        self._draw_points()

    def _draw_points(self):
        self.canvas.delete("pts")
        for i, (ix, iy) in enumerate(S.persp_pts):
            cx, cy = self._img_to_canvas(ix, iy)
            col = self._CORNER_COLORS[i]
            self.canvas.create_oval(
                cx - 7, cy - 7, cx + 7, cy + 7,
                fill=col, outline="white", width=2, tags="pts",
            )
            self.canvas.create_text(
                cx + 14, cy, text=self._CORNER_LABELS[i],
                fill=col, font=("Courier New", 11, "bold"), tags="pts",
            )
        if len(S.persp_pts) >= 2:
            pts_flat = []
            for ix, iy in S.persp_pts:
                cx, cy = self._img_to_canvas(ix, iy)
                pts_flat += [cx, cy]
            self.canvas.create_line(*pts_flat, fill="#58a6ff", width=1, dash=(5, 4), tags="pts")
        self.pts_lbl.config(text=f"Points placed: {len(S.persp_pts)} / 4")

    def _img_to_canvas(self, ix, iy):
        ox, oy = self._offset
        return (
            ix / self._scale + ox,
            iy / self._scale_y + oy
        ) 

    def _canvas_to_img(self, cx, cy):
        ox, oy = self._offset
        return (
            (cx - ox) * self._scale,
            (cy - oy) * self._scale
        )  

    def _on_click(self, e):
        if len(S.persp_pts) >= 4:
            return
        ix, iy = self._canvas_to_img(e.x, e.y)

        ix = max(0, min(S.raw.width - 1, ix))
        iy = max(0, min(S.raw.height - 1, iy))

        S.persp_pts.append((ix, iy))
        self._draw_points()

    def _undo_pt(self):
        if S.persp_pts:
            S.persp_pts.pop()
            self._draw_points()

    def _reset(self):
        S.persp_pts = []
        self._draw_points()

    def _skip(self):
        if not S.raw:
            messagebox.showwarning("No Image", "Load an image in Stage 1 first.")
            return
        S.base    = S.raw.copy()
        S.seg_arr = np.zeros((S.base.height, S.base.width, 4), dtype=np.uint8)
        self.status("Skipped perspective — using original image as base.", "info")
        self.app.nb.select(2)

    def _apply(self):
        if not S.raw:
            messagebox.showwarning("No Image", "Load an image in Stage 1 first.")
            return
        if len(S.persp_pts) != 4:
            messagebox.showwarning(
                "Need 4 Points",
                "Click all 4 corners of the PCB (TL → TR → BR → BL) before applying.",
            )
            return
        try:
            src = np.float32(S.persp_pts)
            ow, oh = self.out_w.get(), self.out_h.get()
            dst = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
            M   = cv2.getPerspectiveTransform(src, dst)
            raw_np  = np.array(S.raw)
            warped  = cv2.warpPerspective(raw_np, M, (ow, oh))
            S.base    = Image.fromarray(warped, "RGBA")
            S.seg_arr = np.zeros((oh, ow, 4), dtype=np.uint8)
            S.seg     = None
            self.status(f"Perspective warp applied → {ow} × {oh} px", "success")
            self._refresh()
            self.app.nb.select(2)
        except Exception as ex:
            messagebox.showerror("Warp Error", str(ex))


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — COLOR LABELING
# ═══════════════════════════════════════════════════════════════════════════════
class Stage3_ColorLabel(BaseStage):
    def _build(self):
        left, right = self._two_pane(250)

        self._section(left, "COLOR LABELS")
        self._lbl(
            left, "Click the image to pick a colour,\nthen assign a PCB layer label.",
        ).pack(padx=10, anchor="w")

        self._sep(left)

        # Picked-colour preview
        row = tk.Frame(left, bg="#161b22")
        row.pack(padx=10, fill="x")
        self.swatch = tk.Frame(row, bg="#30363d", width=32, height=32)
        self.swatch.pack(side="left", padx=(0, 8))
        self.hex_lbl = tk.Label(
            row, text="— pick a colour —",
            bg="#161b22", fg="#e6edf3", font=("Courier New", 10),
        )
        self.hex_lbl.pack(side="left")

        # Layer selector
        self._lbl(left, "Layer label:").pack(padx=10, anchor="w", pady=(8, 0))
        self.label_var = tk.StringVar(value=S.LAYER_LABELS[0])
        om = tk.OptionMenu(left, self.label_var, *S.LAYER_LABELS)
        om.config(
            bg="#21262d", fg="#e6edf3", relief="flat",
            activebackground="#30363d", font=("Courier New", 10),
        )
        om["menu"].config(bg="#21262d", fg="#e6edf3")
        om.pack(padx=10, fill="x")

        # Tolerance
        self._lbl(left, "Colour tolerance:").pack(padx=10, anchor="w", pady=(8, 0))
        self.tol_var = tk.IntVar(value=22)
        tk.Scale(
            left, from_=1, to=80, orient="horizontal",
            variable=self.tol_var,
            bg="#161b22", fg="#e6edf3",
            troughcolor="#21262d", highlightthickness=0,
        ).pack(padx=10, fill="x")

        self._sep(left)
        self._btn(left, "➕  Add Label",      self._add     ).pack(padx=10, fill="x")
        self._btn(left, "🗑  Remove Selected", self._remove  ).pack(padx=10, fill="x", pady=4)
        self._btn(left, "❌  Clear All",       self._clear   ).pack(padx=10, fill="x")

        self._sep(left)
        # Label list
        lf = tk.Frame(left, bg="#161b22")
        lf.pack(fill="both", expand=True, padx=10)
        sb = tk.Scrollbar(lf)
        self.listbox = tk.Listbox(
            lf, bg="#0d1117", fg="#e6edf3",
            font=("Courier New", 9),
            yscrollcommand=sb.set,
            selectbackground="#1f6feb",
            highlightthickness=0,
        )
        sb.config(command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.pack(fill="both", expand=True)

        self._btn(
            left, "▶  Auto-Segment →",
            self._segment,
            bg="#1f6feb", activebackground="#388bfd",
        ).pack(padx=10, fill="x", pady=6)
        self._btn(
            left, "Next →  Stage 4",
            lambda: self.app.nb.select(3),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=(0, 10), fill="x", side="bottom")

        # ── Canvas ────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            right, bg="#0d1117",
            highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>",  self._pick)
        self.canvas.bind("<Configure>", self._refresh)
        self._tk_img = None
        self._scale  = 1.0
        self._offset = (0, 0)
        self._picked: "tuple | None" = None

    def on_enter(self):
        if S.base:
            self._refresh()
        self._render_list()

    def _refresh(self, _e=None):
        if not S.base:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        img = S.base.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        iw, ih = img.size
        self._scale  = S.base.width / iw
        self._scale_y = S.base.height / ih
        self._offset = ((cw - iw) // 2, (ch - ih) // 2)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        ox, oy = self._offset
        self.canvas.create_image(ox, oy, image=self._tk_img, anchor="nw")

    def _pick(self, e):
        if not S.base:
            return
        ox, oy = self._offset
        ix = max(0, min(S.base.width  - 1, int((e.x - ox) * self._scale)))
        iy = max(0, min(S.base.height - 1, int((e.y - oy) * self._scale)))
        px = S.base.getpixel((ix, iy))
        r, g, b = px[0], px[1], px[2]
        self._picked = (r, g, b)
        hexc = rgb_hex(r, g, b)
        self.swatch.config(bg=hexc)
        self.hex_lbl.config(text=hexc)

    def _add(self):
        if not self._picked:
            messagebox.showwarning("No Colour", "Click the image to pick a colour first.")
            return
        r, g, b = self._picked
        hexc = rgb_hex(r, g, b)
        if any(c["hex"] == hexc for c in S.colors):
            messagebox.showinfo("Duplicate", f"{hexc} is already in the list.")
            return
        S.colors.append({"hex": hexc, "label": self.label_var.get(), "rgb": (r, g, b)})
        self._render_list()
        self.status(f"Added {hexc}  →  {self.label_var.get()}", "success")

    def _remove(self):
        sel = self.listbox.curselection()
        if sel:
            S.colors.pop(sel[0])
            self._render_list()

    def _clear(self):
        S.colors.clear()
        self._render_list()

    def _render_list(self):
        self.listbox.delete(0, "end")
        for c in S.colors:
            self.listbox.insert("end", f"  {c['hex']}  {c['label']}")

    def _segment(self):
        if not S.base:
            messagebox.showwarning("No Image", "Load & warp an image first (Stages 1-2).")
            return
        if not S.colors:
            messagebox.showwarning("No Labels", "Pick at least one colour label first.")
            return
        # Snapshot tolerance value on main thread before spawning worker
        tol = self.tol_var.get()
        self.status("Segmenting … please wait", "warn")
        threading.Thread(target=self._do_segment, args=(tol,), daemon=True).start()

    def _do_segment(self, tol: int):
        try:
            img_np = np.array(S.base.convert("RGB"), dtype=np.int32)
            h, w   = img_np.shape[:2]
            out    = np.zeros((h, w, 4), dtype=np.uint8)
            out[:, :, 3] = 255
            out[:, :, :3] = 30       # dark background

            for c in S.colors:
                r, g, b = c["rgb"]
                dist = np.sqrt(
                    (img_np[:, :, 0] - r) ** 2 +
                    (img_np[:, :, 1] - g) ** 2 +
                    (img_np[:, :, 2] - b) ** 2,
                )
                mask = dist < tol
                out[mask, 0] = r
                out[mask, 1] = g
                out[mask, 2] = b

            S.seg_arr = out
            S.seg     = Image.fromarray(out, "RGBA")
            n = len(S.colors)
            self.app.after(0, lambda: self.status(
                f"Segmentation complete — {n} colour class(es) mapped.", "success",
            ))
        except Exception as ex:
            self.app.after(0, lambda: messagebox.showerror("Segmentation Error", str(ex)))


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — PAINT / ERASE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
class Stage4_Paint(BaseStage):
    def _build(self):
        left, right = self._two_pane(230)

        self._section(left, "PAINT / ERASE")

        # Tool
        self.tool_var = tk.StringVar(value="paint")
        for txt, val in [("🖌  Paint", "paint"), ("🧹  Erase", "erase")]:
            tk.Radiobutton(
                left, text=txt, variable=self.tool_var, value=val,
                bg="#161b22", fg="#e6edf3",
                selectcolor="#0d1117", activebackground="#161b22",
                font=("Courier New", 10),
                command=lambda: setattr(S, "current_tool", self.tool_var.get()),
            ).pack(padx=10, anchor="w")

        # Brush size
        self._lbl(left, "Brush size:").pack(padx=10, anchor="w", pady=(8, 0))
        self.brush_var = tk.IntVar(value=S.brush_size)
        tk.Scale(
            left, from_=1, to=120, orient="horizontal",
            variable=self.brush_var,
            bg="#161b22", fg="#e6edf3",
            troughcolor="#21262d", highlightthickness=0,
            command=lambda v: setattr(S, "brush_size", int(v)),
        ).pack(padx=10, fill="x")

        # Paint colour
        self._sep(left)
        self._lbl(left, "Paint colour:").pack(padx=10, anchor="w")
        crow = tk.Frame(left, bg="#161b22")
        crow.pack(padx=10, fill="x", pady=4)
        self.color_swatch = tk.Frame(crow, bg=rgb_hex(*S.paint_color[:3]), width=28, height=28)
        self.color_swatch.pack(side="left", padx=(0, 8))
        self.color_hex_lbl = tk.Label(
            crow, text=rgb_hex(*S.paint_color[:3]),
            bg="#161b22", fg="#e6edf3", font=("Courier New", 10),
        )
        self.color_hex_lbl.pack(side="left")
        self._btn(left, "🎨  Pick Colour …", self._pick_color).pack(padx=10, fill="x")

        # Set colour from labelled swatch
        self._lbl(left, "Use colour from label:").pack(padx=10, anchor="w", pady=(6, 0))
        self.paint_label_var = tk.StringVar(value=S.LAYER_LABELS[0])
        om = tk.OptionMenu(
            left, self.paint_label_var, *S.LAYER_LABELS,
            command=self._set_color_from_label,
        )
        om.config(bg="#21262d", fg="#e6edf3", relief="flat", font=("Courier New", 10))
        om["menu"].config(bg="#21262d", fg="#e6edf3")
        om.pack(padx=10, fill="x")

        # Opacity
        self._sep(left)
        self._lbl(left, "Base layer opacity:").pack(padx=10, anchor="w")
        self.base_op = tk.IntVar(value=S.opacity_base)
        tk.Scale(
            left, from_=0, to=100, orient="horizontal", variable=self.base_op,
            bg="#161b22", fg="#e6edf3", troughcolor="#21262d", highlightthickness=0,
            command=lambda _v: self._refresh_canvas(),
        ).pack(padx=10, fill="x")

        self._lbl(left, "Seg layer opacity:").pack(padx=10, anchor="w")
        self.seg_op = tk.IntVar(value=S.opacity_seg)
        tk.Scale(
            left, from_=0, to=100, orient="horizontal", variable=self.seg_op,
            bg="#161b22", fg="#e6edf3", troughcolor="#21262d", highlightthickness=0,
            command=lambda _v: self._refresh_canvas(),
        ).pack(padx=10, fill="x")

        self._sep(left)
        self._btn(left, "💾  Save Segmented Image", self._save_seg).pack(padx=10, fill="x")
        self._btn(
            left, "Next →  Stage 5",
            lambda: self.app.nb.select(4),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=14, fill="x", side="bottom")

        # ── Canvas ────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            right, bg="#0d1117",
            highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>",          self._start_stroke)
        self.canvas.bind("<B1-Motion>",          self._continue_stroke)
        self.canvas.bind("<ButtonRelease-1>",    self._end_stroke)
        self.canvas.bind("<Configure>",          self._refresh_canvas)

        self._tk_img  = None
        self._scale   = 1.0
        self._offset  = (0, 0)
        self._last_pt: "tuple | None" = None

    def on_enter(self):
        if S.base:
            if S.seg_arr is None:
                S.seg_arr = np.zeros((S.base.height, S.base.width, 4), dtype=np.uint8)
            self._refresh_canvas()

    def _pick_color(self):
        result = colorchooser.askcolor(
            color=rgb_hex(*S.paint_color[:3]), title="Choose Paint Colour",
        )
        if result and result[0]:
            r, g, b = (int(x) for x in result[0])
            S.paint_color = (r, g, b, 255)
            self.color_swatch.config(bg=rgb_hex(r, g, b))
            self.color_hex_lbl.config(text=rgb_hex(r, g, b))

    def _set_color_from_label(self, label: str):
        match = next((c for c in S.colors if c["label"] == label), None)
        if match:
            r, g, b = match["rgb"]
            S.paint_color = (r, g, b, 255)
            self.color_swatch.config(bg=rgb_hex(r, g, b))
            self.color_hex_lbl.config(text=rgb_hex(r, g, b))

    def _composite(self) -> "Image.Image | None":
        if S.base is None:
            return None
        result = Image.new("RGBA", S.base.size, (13, 17, 23, 255))
        base_a = int(self.base_op.get() * 255 / 100)
        base   = S.base.convert("RGBA")
        r, g, b, a = base.split()
        a = a.point(lambda x: int(x * base_a / 255))
        result.alpha_composite(Image.merge("RGBA", (r, g, b, a)))
        if S.seg_arr is not None:
            seg_a   = int(self.seg_op.get() * 255 / 100)
            seg_img = Image.fromarray(S.seg_arr, "RGBA")
            sr, sg, sb, sa = seg_img.split()
            sa = sa.point(lambda x: int(x * seg_a / 255))
            result.alpha_composite(Image.merge("RGBA", (sr, sg, sb, sa)))
        return result

    def _refresh_canvas(self, _e=None):
        comp = self._composite()
        if comp is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        comp.thumbnail((cw, ch), Image.LANCZOS)
        iw, ih = comp.size
        self._scale  = S.base.width / iw
        self._offset = ((cw - iw) // 2, (ch - ih) // 2)
        self._tk_img = ImageTk.PhotoImage(comp)
        self.canvas.delete("all")
        ox, oy = self._offset
        self.canvas.create_image(ox, oy, image=self._tk_img, anchor="nw")

    def _canvas_to_img(self, cx, cy):
        ox, oy = self._offset
        return (cx - ox) * self._scale, (cy - oy) * self._scale

    def _start_stroke(self, e):
        S.push_undo(3, S.seg_arr.copy())
        self._draw_at(e.x, e.y)
        self._last_pt = (e.x, e.y)
        self._refresh_canvas()

    def _continue_stroke(self, e):
        if self._last_pt:
            x0, y0 = self._last_pt
            x1, y1 = e.x, e.y
            dist = math.hypot(x1 - x0, y1 - y0)
            steps = max(1, int(dist))
            for i in range(1, steps + 1):
                t = i / steps
                self._draw_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        self._last_pt = (e.x, e.y)
        self._refresh_canvas()

    def _end_stroke(self, _e):
        self._last_pt = None
        S.seg = Image.fromarray(S.seg_arr, "RGBA")
        self._refresh_canvas()

    def _draw_at(self, cx: float, cy: float):
        if S.seg_arr is None:
            return
        ix = int((cx - self._offset[0]) * self._scale)
        iy = int((cy - self._offset[1]) * self._scale)
        # Brush radius in image pixels (minimum 1)
        bs = max(1, int(self.brush_var.get() / 2))
        h, w = S.seg_arr.shape[:2]
        x1 = max(0, ix - bs);  x2 = min(w, ix + bs + 1)
        y1 = max(0, iy - bs);  y2 = min(h, iy + bs + 1)
        if x2 <= x1 or y2 <= y1:
            return
        # Vectorised circular mask (NumPy — fast)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle  = (xx - ix) ** 2 + (yy - iy) ** 2 <= bs ** 2
        if S.current_tool == "erase":
            S.seg_arr[y1:y2, x1:x2][circle] = 0
        else:
            S.seg_arr[y1:y2, x1:x2][circle] = S.paint_color

    def _save_seg(self):
        if S.seg_arr is None:
            messagebox.showwarning("Nothing to Save", "No segmentation data yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if path:
            Image.fromarray(S.seg_arr, "RGBA").save(path)
            self.status(f"Segmented image saved: {path}", "success")

    def undo(self):
        snap = S.pop_undo(3)
        if snap is not None:
            S.push_redo(3, S.seg_arr.copy())   # save current so redo can replay
            S.seg_arr = snap
            S.seg     = Image.fromarray(S.seg_arr, "RGBA")
            self._refresh_canvas()
            self.status("Undo — paint stroke reversed.", "info")

    def redo(self):
        snap = S.pop_redo(3)
        if snap is not None:
            S._undo[3].append(S.seg_arr.copy())  # save current so undo can reverse
            S.seg_arr = snap
            S.seg     = Image.fromarray(S.seg_arr, "RGBA")
            self._refresh_canvas()
            self.status("Redo — paint stroke reapplied.", "info")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — LAYER OVERLAY VIEW
# ═══════════════════════════════════════════════════════════════════════════════
class Stage5_Overlay(BaseStage):
    def _build(self):
        left, right = self._two_pane(230)

        self._section(left, "OVERLAY")
        self._lbl(left, "Inspect base + segmentation\nlayers with adjustable opacity."
                  ).pack(padx=10, anchor="w")

        self._sep(left)
        self._lbl(left, "Base layer opacity:").pack(padx=10, anchor="w")
        self.base_op = tk.IntVar(value=80)
        tk.Scale(
            left, from_=0, to=100, orient="horizontal", variable=self.base_op,
            bg="#161b22", fg="#e6edf3", troughcolor="#21262d", highlightthickness=0,
            command=lambda _v: self._refresh(),
        ).pack(padx=10, fill="x")

        self._lbl(left, "Seg layer opacity:").pack(padx=10, anchor="w")
        self.seg_op = tk.IntVar(value=70)
        tk.Scale(
            left, from_=0, to=100, orient="horizontal", variable=self.seg_op,
            bg="#161b22", fg="#e6edf3", troughcolor="#21262d", highlightthickness=0,
            command=lambda _v: self._refresh(),
        ).pack(padx=10, fill="x")

        self._sep(left)
        self._btn(left, "💾  Save Composite PNG", self._save).pack(padx=10, fill="x")
        self._btn(
            left, "Next →  Stage 6",
            lambda: self.app.nb.select(5),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=14, fill="x", side="bottom")

        self.canvas = tk.Canvas(right, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._refresh)
        self._tk_img = None
        self._comp: "Image.Image | None" = None

    def on_enter(self):
        self._refresh()

    def _composite(self) -> "Image.Image | None":
        if S.base is None:
            return None
        result = Image.new("RGBA", S.base.size, (13, 17, 23, 255))
        base_a = int(self.base_op.get() * 255 / 100)
        base   = S.base.convert("RGBA")
        r, g, b, a = base.split()
        a = a.point(lambda x: int(x * base_a / 255))
        result.alpha_composite(Image.merge("RGBA", (r, g, b, a)))
        if S.seg_arr is not None:
            seg_a   = int(self.seg_op.get() * 255 / 100)
            seg_img = Image.fromarray(S.seg_arr, "RGBA")
            sr, sg, sb, sa = seg_img.split()
            sa = sa.point(lambda x: int(x * seg_a / 255))
            result.alpha_composite(Image.merge("RGBA", (sr, sg, sb, sa)))
        return result

    def _refresh(self, _e=None):
        self._comp = self._composite()
        if self._comp is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        disp = self._comp.copy()
        disp.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    def _save(self):
        if self._comp is None:
            messagebox.showwarning("Nothing to Save", "No composite available yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if path:
            self._comp.save(path)
            self.status(f"Composite saved: {path}", "success")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — PIN PLACEMENT & GROUPING
# ═══════════════════════════════════════════════════════════════════════════════
class Stage6_Pins(BaseStage):
    def _build(self):
        left, right = self._two_pane(250)

        self._section(left, "PIN PLACEMENT")

        self.tool_var = tk.StringVar(value="pin")
        for txt, val in [("📍  Place Pin", "pin"), ("✋  Move Mode", "move")]:
            tk.Radiobutton(
                left, text=txt, variable=self.tool_var, value=val,
                bg="#161b22", fg="#e6edf3",
                selectcolor="#0d1117", activebackground="#161b22",
                font=("Courier New", 10),
            ).pack(padx=10, anchor="w")

        self._lbl(left, "Pin size:").pack(padx=10, anchor="w", pady=(8, 0))
        self.pin_size_var = tk.IntVar(value=S.pin_size)
        tk.Scale(
            left, from_=4, to=40, orient="horizontal", variable=self.pin_size_var,
            bg="#161b22", fg="#e6edf3", troughcolor="#21262d", highlightthickness=0,
            command=lambda _v: self._refresh_canvas(),
        ).pack(padx=10, fill="x")

        self._sep(left)
        self._btn(left, "👥  Group Selected",     self._group_dialog  ).pack(padx=10, fill="x")
        self._btn(left, "✏️  Rename Pin / Block",  self._rename_pin    ).pack(padx=10, fill="x", pady=4)
        self._btn(left, "🗑  Delete Selected",     self._delete_selected).pack(padx=10, fill="x")

        self._sep(left)
        self._btn(left, "💾  Save Pins JSON",  self._save_json).pack(padx=10, fill="x")
        self._btn(left, "📂  Load Pins JSON",  self._load_json).pack(padx=10, fill="x", pady=4)

        self._sep(left)
        tk.Label(left, text="PINS  (click to select)",
                 bg="#161b22", fg="#00e5ff",
                 font=("Courier New", 10, "bold")).pack(padx=10, anchor="w")
        lf = tk.Frame(left, bg="#161b22")
        lf.pack(fill="both", expand=True, padx=10)
        sb = tk.Scrollbar(lf)
        self.pin_list = tk.Listbox(
            lf, bg="#0d1117", fg="#e6edf3",
            font=("Courier New", 9),
            yscrollcommand=sb.set,
            selectbackground="#1f6feb",
            highlightthickness=0,
        )
        sb.config(command=self.pin_list.yview)
        sb.pack(side="right", fill="y")
        self.pin_list.pack(fill="both", expand=True)
        self.pin_list.bind("<<ListboxSelect>>", self._list_select)

        self._btn(
            left, "Next →  Stage 7",
            lambda: self.app.nb.select(6),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=14, fill="x", side="bottom")

        # ── Canvas ────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            right, bg="#0d1117",
            highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>",          self._on_click)
        self.canvas.bind("<B1-Motion>",          self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",    self._on_release)
        self.canvas.bind("<Configure>",          self._refresh_canvas)
        self.canvas.bind("<Delete>",             lambda _e: self._delete_selected())

        self._tk_img       = None
        self._scale        = 1.0
        self._offset       = (0, 0)
        self._selected_ids: set = set()
        self._dragging_pin: "dict | None" = None

    def on_enter(self):
        if S.base:
            self._refresh_canvas()
        self._refresh_list()

    # ── Image helpers ──────────────────────────────────────────────────────
    def _build_composite(self) -> "Image.Image | None":
        if not S.base:
            return None
        result = Image.new("RGBA", S.base.size, (13, 17, 23, 255))
        base = S.base.convert("RGBA")
        r, g, b, a = base.split()
        a = a.point(lambda x: int(x * 0.75))
        result.alpha_composite(Image.merge("RGBA", (r, g, b, a)))
        if S.seg_arr is not None:
            seg = Image.fromarray(S.seg_arr, "RGBA")
            sr, sg, sb, sa = seg.split()
            sa = sa.point(lambda x: int(x * 0.55))
            result.alpha_composite(Image.merge("RGBA", (sr, sg, sb, sa)))
        return result

    def _refresh_canvas(self, _e=None):
        comp = self._build_composite()
        if comp is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        comp.thumbnail((cw, ch), Image.LANCZOS)
        iw, ih = comp.size
        self._scale  = S.base.width / iw
        self._offset = ((cw - iw) // 2, (ch - ih) // 2)

        # Draw pins onto PIL image (fast, single blit)
        draw = ImageDraw.Draw(comp)
        ps   = max(3, int(self.pin_size_var.get() / self._scale))
        for p in S.pins:
            cx = int(p["x"] / self._scale)
            cy = int(p["y"] / self._scale)
            col    = p.get("color", "#00e5ff")
            border = "#ffff00" if p["id"] in self._selected_ids else "#ffffff"
            draw.ellipse(
                [cx - ps, cy - ps, cx + ps, cy + ps],
                fill=col, outline=border, width=2,
            )
            name = p.get("pin_name", "") or str(p["id"])
            draw.text((cx + ps + 2, cy - ps), name, fill="#e6edf3")

        self._tk_img = ImageTk.PhotoImage(comp)
        self.canvas.delete("all")
        ox, oy = self._offset
        self.canvas.create_image(ox, oy, image=self._tk_img, anchor="nw")

    def _canvas_to_img(self, cx, cy):
        ox, oy = self._offset
        return (cx - ox) * self._scale, (cy - oy) * self._scale

    def _hit_pin(self, ix: float, iy: float) -> "dict | None":
        ps = float(self.pin_size_var.get())
        for p in reversed(S.pins):
            if (p["x"] - ix) ** 2 + (p["y"] - iy) ** 2 <= ps ** 2:
                return p
        return None

    # ── Canvas events ──────────────────────────────────────────────────────
    def _on_click(self, e):
        ix, iy = self._canvas_to_img(e.x, e.y)
        hit    = self._hit_pin(ix, iy)
        if self.tool_var.get() == "pin":
            if hit:
                # Select pin (possibly multi-select with Shift) but do NOT drag
                if e.state & 0x0001:   # Shift held → multi-select
                    if hit["id"] in self._selected_ids:
                        self._selected_ids.discard(hit["id"])
                    else:
                        self._selected_ids.add(hit["id"])
                else:
                    self._selected_ids = {hit["id"]}
                # Do NOT set _dragging_pin — dragging is only for move mode
            else:
                self._selected_ids = set()
                self._add_pin(ix, iy)
        else:  # move mode
            if hit:
                self._selected_ids = {hit["id"]}
                self._dragging_pin = hit    # only drag in move mode
            else:
                self._selected_ids = set()
        self._refresh_canvas()
        self._refresh_list()

    def _on_drag(self, e):
        if self._dragging_pin is not None:
            ix, iy = self._canvas_to_img(e.x, e.y)
            h, w = (S.base.height, S.base.width) if S.base else (1e9, 1e9)
            self._dragging_pin["x"] = max(0, min(w, ix))
            self._dragging_pin["y"] = max(0, min(h, iy))
            self._refresh_canvas()

    def _on_release(self, _e):
        self._dragging_pin = None

    # ── Pin operations ─────────────────────────────────────────────────────
    def _add_pin(self, ix: float, iy: float):
        S.push_undo(5, copy.deepcopy(S.pins))
        pin = {
            "id": S.pin_ctr, "x": ix, "y": iy,
            "block_name": "", "pin_name": "",
            "group_id": None, "net_id": None,
            "color": "#00e5ff",
        }
        S.pins.append(pin)
        S.pin_ctr += 1
        self._refresh_list()

    def _delete_selected(self):
        if not self._selected_ids:
            messagebox.showwarning("No Selection", "Select at least one pin first.")
            return
        S.push_undo(5, copy.deepcopy(S.pins))
        S.pins = [p for p in S.pins if p["id"] not in self._selected_ids]
        self._selected_ids.clear()
        self._refresh_canvas()
        self._refresh_list()

    def _rename_pin(self):
        if len(self._selected_ids) != 1:
            messagebox.showwarning("Select One Pin", "Select exactly one pin to rename.")
            return
        pid = next(iter(self._selected_ids))
        p   = next((x for x in S.pins if x["id"] == pid), None)
        if not p:
            return
        name = simpledialog.askstring(
            "Pin Name", f"Name for pin #{pid}:",
            initialvalue=p.get("pin_name", ""),
        )
        if name is not None:
            p["pin_name"] = name
        block = simpledialog.askstring(
            "Block Name", f"Component block for pin #{pid}:",
            initialvalue=p.get("block_name", ""),
        )
        if block is not None:
            p["block_name"] = block
        self._refresh_canvas()
        self._refresh_list()

    def _group_dialog(self):
        if len(self._selected_ids) < 2:
            messagebox.showwarning("Too Few Pins", "Select 2 or more pins to group.")
            return
        name = simpledialog.askstring(
            "Group Name", "Component reference (e.g. U1, IC2):",
            initialvalue=f"U{S.group_ctr + 1}",
        )
        if not name:
            return
        auto = messagebox.askyesno(
            "Auto-name Pins",
            "Auto-name pins (Yes), or enter each name manually (No)?",
        )
        # Save undo BEFORE mutating state
        S.push_undo(5, copy.deepcopy(S.pins))

        rc    = rand_color()
        color = rgb_hex(*rc)
        gid   = S.group_ctr;  S.group_ctr += 1
        pin_ids = list(self._selected_ids)
        S.groups.append({"id": gid, "name": name, "pin_ids": pin_ids, "color": color})

        for i, pid in enumerate(pin_ids):
            p = next((x for x in S.pins if x["id"] == pid), None)
            if not p:
                continue
            p["group_id"]   = gid
            p["block_name"] = name
            p["color"]      = color
            if auto:
                p["pin_name"] = f"{name}_p{i + 1}"
            else:
                nm = simpledialog.askstring(
                    "Pin Name",
                    f"Name for pin #{pid} in {name}:",
                    initialvalue=f"{name}_p{i + 1}",
                )
                p["pin_name"] = nm or f"{name}_p{i + 1}"

        self._selected_ids.clear()
        self._refresh_canvas()
        self._refresh_list()
        self.status(f"Group '{name}' created with {len(pin_ids)} pins.", "success")

    def _refresh_list(self):
        self.pin_list.delete(0, "end")
        for p in S.pins:
            entry = (
                f"#{p['id']:3d}  "
                f"{(p.get('pin_name') or '—'):14s}  "
                f"{p.get('block_name') or '—'}"
            )
            self.pin_list.insert("end", entry)

    def _list_select(self, _e=None):
        sel = self.pin_list.curselection()
        if sel and sel[0] < len(S.pins):
            self._selected_ids = {S.pins[sel[0]]["id"]}
            self._refresh_canvas()

    def _save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{S.project_name}_pins.json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            data = [
                {k: v for k, v in p.items() if k != "color"}
                for p in S.pins
            ]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.status(f"Pins saved: {path}", "success")

    def _load_json(self):
        path = filedialog.askopenfilename(
            title="Load Pins JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for p in data:
                if "color" not in p:
                    p["color"] = "#00e5ff"
            S.pins    = data
            S.pin_ctr = max((p["id"] for p in S.pins), default=0) + 1
            self._refresh_canvas()
            self._refresh_list()
            self.status(f"Loaded {len(S.pins)} pins from {Path(path).name}", "success")
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex))

    def undo(self):
        snap = S.pop_undo(5)
        if snap is not None:
            S.push_redo(5, copy.deepcopy(S.pins))   # save current for redo
            S.pins = snap
            self._refresh_canvas()
            self._refresh_list()
            self.status("Undo — pin action reversed.", "info")

    def redo(self):
        snap = S.pop_redo(5)
        if snap is not None:
            S._undo[5].append(copy.deepcopy(S.pins))  # save current for undo
            S.pins = snap
            self._refresh_canvas()
            self._refresh_list()
            self.status("Redo — pin action reapplied.", "info")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 7 — NET ANALYSIS  (OpenCV connected components)
# ═══════════════════════════════════════════════════════════════════════════════
class Stage7_Nets(BaseStage):
    def _build(self):
        left, right = self._two_pane(250)

        self._section(left, "NET ANALYSIS")
        self._lbl(
            left,
            "Identifies electrically-connected\ncopper regions and assigns each\npin to a net.",
        ).pack(padx=10, anchor="w")

        self._sep(left)
        tk.Label(left, text="Conductive layer classes:",
                 bg="#161b22", fg="#8b949e",
                 font=("Courier New", 10)).pack(padx=10, anchor="w")
        self.layer_vars: dict = {}
        for lbl in ["Copper Trace", "Pad", "Via"]:
            v = tk.BooleanVar(value=True)
            self.layer_vars[lbl] = v
            tk.Checkbutton(
                left, text=lbl, variable=v,
                bg="#161b22", fg="#e6edf3",
                selectcolor="#0d1117", activebackground="#161b22",
                font=("Courier New", 10),
            ).pack(padx=18, anchor="w")

        self._sep(left)
        self._lbl(left, "Colour tolerance:").pack(padx=10, anchor="w")
        self.tol_var = tk.IntVar(value=25)
        tk.Scale(
            left, from_=1, to=80, orient="horizontal", variable=self.tol_var,
            bg="#161b22", fg="#e6edf3", troughcolor="#21262d", highlightthickness=0,
        ).pack(padx=10, fill="x")

        self._lbl(left, "Morphology kernel size:").pack(padx=10, anchor="w")
        self.kernel_var = tk.IntVar(value=3)
        tk.Scale(
            left, from_=1, to=15, orient="horizontal", variable=self.kernel_var,
            resolution=2, bg="#161b22", fg="#e6edf3",
            troughcolor="#21262d", highlightthickness=0,
        ).pack(padx=10, fill="x")

        self._lbl(left, "Pin snap radius (px):").pack(padx=10, anchor="w")
        self.snap_var = tk.IntVar(value=S.snap_radius)
        self.snap_lbl = tk.Label(
            left, text=str(S.snap_radius),
            bg="#161b22", fg="#e6edf3", font=("Courier New", 10),
        )
        self.snap_lbl.pack(padx=10, anchor="w")
        tk.Scale(
            left, from_=0, to=80, orient="horizontal", variable=self.snap_var,
            bg="#161b22", fg="#e6edf3", troughcolor="#21262d", highlightthickness=0,
            command=lambda v: (
                setattr(S, "snap_radius", int(v)),
                self.snap_lbl.config(text=str(int(float(v)))),
            ),
        ).pack(padx=10, fill="x")

        self._sep(left)
        self._btn(
            left, "▶  Run Net Analysis",
            self._run,
            bg="#1f6feb", activebackground="#388bfd",
        ).pack(padx=10, fill="x")
        self._btn(left, "💾  Save Net Map Image", self._save_img).pack(padx=10, fill="x", pady=4)

        self._sep(left)
        tk.Label(left, text="NETS FOUND",
                 bg="#161b22", fg="#00e5ff",
                 font=("Courier New", 10, "bold")).pack(padx=10, anchor="w")
        lf = tk.Frame(left, bg="#161b22")
        lf.pack(fill="both", expand=True, padx=10)
        sb = tk.Scrollbar(lf)
        self.net_list = tk.Listbox(
            lf, bg="#0d1117", fg="#e6edf3",
            font=("Courier New", 9),
            yscrollcommand=sb.set,
            highlightthickness=0,
        )
        sb.config(command=self.net_list.yview)
        sb.pack(side="right", fill="y")
        self.net_list.pack(fill="both", expand=True)

        self._btn(
            left, "Next →  Stage 8",
            lambda: self.app.nb.select(7),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=14, fill="x", side="bottom")

        self.canvas = tk.Canvas(right, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._refresh_canvas)
        self._tk_img = None

    def on_enter(self):
        if S.base:
            self._refresh_canvas()
        self._refresh_list()

    def _run(self):
        if S.seg_arr is None or not np.any(S.seg_arr[:, :, 3]):
            messagebox.showwarning(
                "No Segmentation",
                "Complete Stages 3 and 4 to create segmentation data first.",
            )
            return
        if not S.pins:
            messagebox.showwarning("No Pins", "Place pins in Stage 6 before running analysis.")
            return
        if not S.colors:
            messagebox.showwarning("No Colors", "Define colour labels in Stage 3 first.")
            return
        # Snapshot all tkinter Var values NOW (main thread) — thread-safe
        params = {
            "tol":          self.tol_var.get(),
            "kernel":       self.kernel_var.get(),
            "snap":         self.snap_var.get(),
            "active_labels":[lbl for lbl, v in self.layer_vars.items() if v.get()],
        }
        self.status("Running net analysis … (may be slow for large images)", "warn")
        threading.Thread(target=self._do_analysis, args=(params,), daemon=True).start()

    def _do_analysis(self, params: dict):
        try:
            seg = S.seg_arr.astype(np.int32)
            h, w = seg.shape[:2]

            # Use pre-snapshotted values — safe from background thread
            tol           = params["tol"]
            active_labels = params["active_labels"]
            k_size        = params["kernel"]
            snap          = params["snap"]

            # Build binary conductive mask using selected layer colours
            mask = np.zeros((h, w), dtype=np.uint8)
            for c in S.colors:
                if c["label"] not in active_labels:
                    continue
                r, g, b = c["rgb"]
                dist = np.sqrt(
                    (seg[:, :, 0] - r) ** 2 +
                    (seg[:, :, 1] - g) ** 2 +
                    (seg[:, :, 2] - b) ** 2,
                )
                mask[dist < tol] = 255

            # Morphological clean-up (open + close)
            if k_size % 2 == 0:
                k_size += 1
            k_size = max(1, k_size)
            kernel = np.ones((k_size, k_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8, ltype=cv2.CV_32S,
            )
            num_nets = num_labels - 1   # label 0 is background

            # Assign random colours to each net
            net_colors: dict = {}
            for i in range(1, num_labels):
                net_colors[i] = rand_color()

            # Build coloured output image (vectorised, no per-pixel loop)
            out = np.full((h, w, 4), [30, 30, 30, 255], dtype=np.uint8)
            for lbl_id in range(1, num_labels):
                ym = labels == lbl_id
                out[ym, 0] = net_colors[lbl_id][0]
                out[ym, 1] = net_colors[lbl_id][1]
                out[ym, 2] = net_colors[lbl_id][2]

            # Assign each pin to a net via majority-vote in snap radius
            net_map: dict = {}
            for p in S.pins:
                px, py = int(p["x"]), int(p["y"])
                lbl_id = 0
                if 0 <= py < h and 0 <= px < w:
                    lbl_id = int(labels[py, px])
                if lbl_id == 0 and snap > 0:
                    y1 = max(0, py - snap);  y2 = min(h, py + snap + 1)
                    x1 = max(0, px - snap);  x2 = min(w, px + snap + 1)
                    region = labels[y1:y2, x1:x2]
                    nonzero = region[region > 0].flatten()
                    if nonzero.size > 0:
                        counts  = np.bincount(nonzero)
                        lbl_id  = int(counts.argmax())

                p["net_id"] = lbl_id if lbl_id > 0 else None
                if lbl_id > 0:
                    col       = net_colors[lbl_id]
                    p["color"] = rgb_hex(*col)
                    if lbl_id not in net_map:
                        net_map[lbl_id] = {
                            "id":      lbl_id,
                            "label":   f"NET_{lbl_id}",
                            "pin_ids": [],
                            "color":   rgb_hex(*col),
                        }
                    net_map[lbl_id]["pin_ids"].append(p["id"])
                else:
                    p["color"] = "#f85149"   # not connected

            S.nets    = list(net_map.values())
            S.net_img = Image.fromarray(out, "RGBA")

            # Update UI from main thread
            self.app.after(0, self._after_analysis, num_nets)
        except Exception as ex:
            self.app.after(0, lambda: messagebox.showerror("Analysis Error", str(ex)))

    def _after_analysis(self, num_nets: int):
        self._refresh_list()
        self._refresh_canvas()
        nc = len([p for p in S.pins if not p.get("net_id")])
        self.status(
            f"Net analysis complete — {num_nets} copper regions found, "
            f"{len(S.nets)} contain pins, {nc} pins unconnected.",
            "success",
        )

    def _refresh_canvas(self, _e=None):
        if not S.base:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        result = Image.new("RGBA", S.base.size, (13, 17, 23, 255))
        base   = S.base.convert("RGBA")
        r, g, b, a = base.split()
        a = a.point(lambda x: int(x * 0.35))
        result.alpha_composite(Image.merge("RGBA", (r, g, b, a)))

        if S.net_img:
            nm = S.net_img.copy()
            nr, ng, nb2, na = nm.split()
            na = na.point(lambda x: int(x * 0.80))
            result.alpha_composite(Image.merge("RGBA", (nr, ng, nb2, na)))

        draw = ImageDraw.Draw(result)
        ps   = S.pin_size
        for p in S.pins:
            cx, cy = int(p["x"]), int(p["y"])
            col = p.get("color", "#00e5ff")
            draw.ellipse([cx - ps, cy - ps, cx + ps, cy + ps],
                         fill=col, outline="#ffffff", width=2)
            nm2 = p.get("pin_name", "") or str(p["id"])
            draw.text((cx + ps + 2, cy - ps), nm2, fill="#e6edf3")

        result.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(result)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    def _refresh_list(self):
        self.net_list.delete(0, "end")
        for n in S.nets:
            self.net_list.insert(
                "end", f"  {n['label']:16s}  {len(n['pin_ids'])} pin(s)"
            )
        nc = len([p for p in S.pins if not p.get("net_id")])
        self.net_list.insert("end", f"  — NC / unassigned —    {nc} pin(s)")

    def _save_img(self):
        if not S.net_img:
            messagebox.showwarning("No Data", "Run the analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{S.project_name}_nets.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if path:
            S.net_img.save(path)
            self.status(f"Net map saved: {path}", "success")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 8 — ROUTING / SCHEMATIC VIEW
# ═══════════════════════════════════════════════════════════════════════════════
class Stage8_Routing(BaseStage):
    def _build(self):
        left, right = self._two_pane(250)

        self._section(left, "ROUTING VIEW")

        self.view_var = tk.StringVar(value="overlay")
        for txt, val in [("🗺  Overlay (ratsnest)", "overlay"),
                          ("📐  Schematic blocks",  "schematic")]:
            tk.Radiobutton(
                left, text=txt, variable=self.view_var, value=val,
                bg="#161b22", fg="#e6edf3",
                selectcolor="#0d1117", activebackground="#161b22",
                font=("Courier New", 10),
                command=self._refresh,
            ).pack(padx=10, anchor="w")

        self._sep(left)
        self._btn(left, "✏️  Rename Selected Net", self._rename_net).pack(padx=10, fill="x")

        self._sep(left)
        self._btn(left, "💾  Save View as PNG",    self._save).pack(padx=10, fill="x")

        self._sep(left)
        tk.Label(left, text="NETS",
                 bg="#161b22", fg="#00e5ff",
                 font=("Courier New", 10, "bold")).pack(padx=10, anchor="w")
        lf = tk.Frame(left, bg="#161b22")
        lf.pack(fill="both", expand=True, padx=10)
        sb = tk.Scrollbar(lf)
        self.net_list = tk.Listbox(
            lf, bg="#0d1117", fg="#e6edf3",
            font=("Courier New", 9),
            yscrollcommand=sb.set,
            selectbackground="#1f6feb",
            highlightthickness=0,
        )
        sb.config(command=self.net_list.yview)
        sb.pack(side="right", fill="y")
        self.net_list.pack(fill="both", expand=True)
        self.net_list.bind("<<ListboxSelect>>", lambda _e: self._refresh())

        self._btn(
            left, "Next →  Stage 9",
            lambda: self.app.nb.select(8),
            bg="#238636", activebackground="#2ea043",
        ).pack(padx=10, pady=14, fill="x", side="bottom")

        self.canvas = tk.Canvas(right, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._refresh)
        self._tk_img  = None
        self._overlay: "Image.Image | None" = None

    def on_enter(self):
        self._render_list()
        self._refresh()

    def _render_list(self):
        self.net_list.delete(0, "end")
        for n in S.nets:
            self.net_list.insert("end", f"  {n['label']}  ({len(n['pin_ids'])} pins)")

    def _rename_net(self):
        sel = self.net_list.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a net in the list first.")
            return
        idx = sel[0]
        if idx >= len(S.nets):
            return
        net  = S.nets[idx]
        name = simpledialog.askstring(
            "Rename Net", "New net name:", initialvalue=net["label"],
        )
        if name:
            net["label"] = name
            self._render_list()
            self._refresh()

    def _refresh(self, _e=None):
        if self.view_var.get() == "overlay":
            self._render_overlay()
        else:
            self._render_schematic()

    # ── Overlay (PIL image) ────────────────────────────────────────────────
    def _render_overlay(self):
        if not S.base:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        result = Image.new("RGBA", S.base.size, (13, 17, 23, 255))
        base   = S.base.convert("RGBA")
        r, g, b, a = base.split()
        a = a.point(lambda x: int(x * 0.4))
        result.alpha_composite(Image.merge("RGBA", (r, g, b, a)))

        draw = ImageDraw.Draw(result)
        ps   = S.pin_size

        # Ratsnest lines (star topology per net)
        for net in S.nets:
            pins_in = [p for p in S.pins if p["id"] in net["pin_ids"]]
            if len(pins_in) < 2:
                continue
            try:
                nr, ng2, nb2 = hex_rgb(net["color"])
            except Exception:
                nr, ng2, nb2 = 0, 229, 255
            p0 = pins_in[0]
            for p1 in pins_in[1:]:
                draw.line(
                    [p0["x"], p0["y"], p1["x"], p1["y"]],
                    fill=(nr, ng2, nb2, 180), width=2,
                )
            # Net label
            mid_x = int(sum(p["x"] for p in pins_in) / len(pins_in))
            mid_y = int(sum(p["y"] for p in pins_in) / len(pins_in))
            draw.text((mid_x + 4, mid_y - 12), net["label"], fill=(139, 148, 158, 220))

        # Pins
        for p in S.pins:
            cx, cy = int(p["x"]), int(p["y"])
            col = p.get("color", "#00e5ff")
            draw.ellipse([cx - ps, cy - ps, cx + ps, cy + ps],
                         fill=col, outline="#ffffff", width=2)
            nm = p.get("pin_name", "") or str(p["id"])
            draw.text((cx + ps + 2, cy - ps), nm, fill="#e6edf3")

        self._overlay = result.copy()
        result.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(result)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    # ── Schematic view (native canvas drawing) ────────────────────────────
    def _render_schematic(self):
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, cw, ch, fill="#0d1117", outline="")

        if not S.groups:
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="No component groups defined.\nGroup pins in Stage 6 to see the schematic.",
                fill="#30363d", font=("Courier New", 14), justify="center",
            )
            return

        bw   = 160   # block width
        rh   = 26    # row height per pin
        cols = max(1, cw // 240)
        pin_stub_coords: dict = {}  # pin_id → (canvas_x, canvas_y)

        for i, g in enumerate(S.groups):
            col_idx = i % cols
            row_idx = i // cols
            pins_in = [p for p in S.pins if p.get("group_id") == g["id"]]
            bh = 40 + len(pins_in) * rh

            gx = 30 + col_idx * 230
            gy = 30 + row_idx * (bh + 50)

            # Block rectangle
            self.canvas.create_rectangle(
                gx, gy, gx + bw, gy + bh,
                fill="#161b22", outline=g["color"], width=2,
            )
            self.canvas.create_text(
                gx + bw // 2, gy + 16,
                text=g["name"], fill="#e6edf3",
                font=("Courier New", 11, "bold"),
            )

            # Pins + stubs
            for j, pin in enumerate(pins_in):
                py_pos     = gy + 40 + j * rh + rh // 2
                is_left    = j % 2 == 0
                inner_x    = gx if is_left else gx + bw
                stub_x     = gx - 22 if is_left else gx + bw + 22
                txt_anchor = "e" if is_left else "w"
                txt_x      = inner_x + (-5 if is_left else 5)

                self.canvas.create_line(
                    inner_x, py_pos, stub_x, py_pos,
                    fill="#00e5ff", width=2,
                )
                self.canvas.create_text(
                    txt_x, py_pos,
                    text=pin.get("pin_name") or str(pin["id"]),
                    fill="#e6edf3", font=("Courier New", 9),
                    anchor=txt_anchor,
                )
                self.canvas.create_oval(
                    stub_x - 3, py_pos - 3, stub_x + 3, py_pos + 3,
                    fill=pin.get("color", "#00e5ff"), outline="",
                )
                pin_stub_coords[pin["id"]] = (stub_x, py_pos)

        # Draw net connections between stubs (fixed vs HTML: lines actually connect)
        for net in S.nets:
            pts = [pin_stub_coords[pid] for pid in net["pin_ids"] if pid in pin_stub_coords]
            if len(pts) < 2:
                continue
            p0 = pts[0]
            for p1 in pts[1:]:
                self.canvas.create_line(
                    p0[0], p0[1], p1[0], p1[1],
                    fill=net["color"], width=2, dash=(7, 4),
                )
            # Net label at midpoint
            mx = (p0[0] + pts[-1][0]) // 2
            my = (p0[1] + pts[-1][1]) // 2
            self.canvas.create_text(
                mx, my - 10, text=net["label"],
                fill="#8b949e", font=("Courier New", 8),
            )

    # ── Save ──────────────────────────────────────────────────────────────
    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{S.project_name}_routing.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if not path:
            return
        if self.view_var.get() == "overlay" and self._overlay:
            self._overlay.save(path)
            self.status(f"Overlay saved: {path}", "success")
        else:
            # Render schematic view to a PIL image using the same drawing logic
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            img  = Image.new("RGB", (cw, ch), (13, 17, 23))
            draw = ImageDraw.Draw(img)
            bw, rh = 160, 26
            cols = max(1, cw // 240)
            pin_stub_coords: dict = {}
            for i, g in enumerate(S.groups):
                pins_in = [p for p in S.pins if p.get("group_id") == g["id"]]
                bh  = 40 + len(pins_in) * rh
                gx  = 30 + (i % cols) * 230
                gy  = 30 + (i // cols) * (bh + 50)
                try:
                    gr, gg2, gb2 = hex_rgb(g["color"])
                except Exception:
                    gr, gg2, gb2 = 0, 229, 255
                draw.rectangle([gx, gy, gx + bw, gy + bh],
                               fill=(22, 27, 34), outline=(gr, gg2, gb2), width=2)
                draw.text((gx + bw // 2, gy + 6), g["name"], fill=(230, 237, 243))
                for j, pin in enumerate(pins_in):
                    py_pos  = gy + 40 + j * rh + rh // 2
                    is_left = j % 2 == 0
                    inner_x = gx if is_left else gx + bw
                    stub_x  = gx - 22 if is_left else gx + bw + 22
                    draw.line([(inner_x, py_pos), (stub_x, py_pos)], fill=(0, 229, 255), width=2)
                    pname = pin.get("pin_name") or str(pin["id"])
                    draw.text((inner_x + (-5 if is_left else 5), py_pos - 6), pname, fill=(230, 237, 243))
                    try:
                        pr, pg2, pb2 = hex_rgb(pin.get("color", "#00e5ff"))
                    except Exception:
                        pr, pg2, pb2 = 0, 229, 255
                    draw.ellipse([stub_x - 4, py_pos - 4, stub_x + 4, py_pos + 4],
                                 fill=(pr, pg2, pb2))
                    pin_stub_coords[pin["id"]] = (stub_x, py_pos)
            for net in S.nets:
                pts = [pin_stub_coords[pid] for pid in net["pin_ids"] if pid in pin_stub_coords]
                if len(pts) < 2:
                    continue
                try:
                    nr, ng2, nb2 = hex_rgb(net["color"])
                except Exception:
                    nr, ng2, nb2 = 0, 229, 255
                p0 = pts[0]
                for p1 in pts[1:]:
                    draw.line([p0, p1], fill=(nr, ng2, nb2), width=2)
            img.save(path)
            self.status(f"Schematic PNG saved: {path}", "success")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 9 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
class Stage9_Export(BaseStage):
    def _build(self):
        left, right = self._two_pane(250)

        self._section(left, "EXPORT")
        self._lbl(left, "Project name:").pack(padx=10, anchor="w")
        self.name_var = tk.StringVar(value=S.project_name)
        tk.Entry(
            left, textvariable=self.name_var,
            bg="#0d1117", fg="#e6edf3",
            insertbackground="#e6edf3", relief="flat",
            font=("Courier New", 10),
        ).pack(padx=10, fill="x", pady=4)

        self._sep(left)
        exports = [
            ("📄  KiCad Netlist (.net)",         self._export_kicad),
            ("📄  KiCad PCB (.kicad_pcb)",        self._export_kicad_pcb),
            ("📋  Pins CSV",                       self._export_pins_csv),
            ("📋  Nets CSV",                       self._export_nets_csv),
            ("🗃  Project JSON",                   self._export_json),
            ("🗜  ZIP Bundle (all files)",         self._export_zip),
        ]
        for text, cmd in exports:
            self._btn(left, text, cmd).pack(padx=10, fill="x", pady=2)

        self._sep(left)
        self.stats_frame = tk.Frame(left, bg="#161b22")
        self.stats_frame.pack(padx=10, fill="x")

        # ── Right panel: netlist preview ──────────────────────────────────
        hdr = tk.Frame(right, bg="#161b22")
        hdr.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(hdr, text="KiCad Netlist Preview",
                 bg="#161b22", fg="#00e5ff",
                 font=("Courier New", 11, "bold")).pack(side="left")
        self._btn(hdr, "🔄 Refresh", self._preview_netlist).pack(side="right")

        self.text = tk.Text(
            right, bg="#0d1117", fg="#e6edf3",
            font=("Courier New", 10), relief="flat",
            wrap="none", state="disabled",
        )
        sbv = tk.Scrollbar(right, command=self.text.yview)
        sbh = tk.Scrollbar(right, orient="horizontal", command=self.text.xview)
        self.text.config(yscrollcommand=sbv.set, xscrollcommand=sbh.set)
        sbv.pack(side="right",  fill="y")
        sbh.pack(side="bottom", fill="x")
        self.text.pack(fill="both", expand=True, padx=8, pady=4)

    def on_enter(self):
        S.project_name = self.name_var.get().strip() or S.project_name
        self._update_stats()
        self._preview_netlist()

    def _update_stats(self):
        for w in self.stats_frame.winfo_children():
            w.destroy()
        nc = len([p for p in S.pins if not p.get("net_id")])
        for label, val in [
            ("Pins",    len(S.pins)),
            ("Groups",  len(S.groups)),
            ("Nets",    len(S.nets)),
            ("NC Pins", nc),
            ("Colors",  len(S.colors)),
        ]:
            row = tk.Frame(self.stats_frame, bg="#161b22")
            row.pack(fill="x")
            tk.Label(row, text=f"{label}:", bg="#161b22", fg="#8b949e",
                     font=("Courier New", 10), width=10, anchor="w").pack(side="left")
            tk.Label(row, text=str(val), bg="#161b22", fg="#3fb950",
                     font=("Courier New", 10, "bold")).pack(side="left")

    # ── KiCad Netlist string ───────────────────────────────────────────────
    def _kicad_net_str(self) -> str:
        name  = self.name_var.get().strip() or S.project_name
        lines = [
            "(export (version D)",
            "  (components",
        ]
        seen: set = set()
        for g in S.groups:
            ref = g["name"] or "U?"
            lines.append(f'    (comp (ref "{ref}") (value "{ref}") (footprint ""))')
            seen.add(ref)
        for p in S.pins:
            ref = p.get("block_name") or f"U_NC_{p['id']}"
            if ref not in seen:
                lines.append(f'    (comp (ref "{ref}") (value "{ref}") (footprint ""))')
                seen.add(ref)
        lines += ["  )", "  (nets"]
        for n in S.nets:
            lines.append(f'    (net (code {n["id"]}) (name "{n["label"]}")')
            for pid in n["pin_ids"]:
                p = next((x for x in S.pins if x["id"] == pid), None)
                if not p:
                    continue
                ref  = p.get("block_name") or f"U_NC_{p['id']}"
                pname = p.get("pin_name") or "1"
                lines.append(f'      (node (ref "{ref}") (pin "{pname}"))')
            lines.append("    )")
        lines += ["  )", ")"]
        return "\n".join(lines)

    def _preview_netlist(self):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        try:
            self.text.insert("end", self._kicad_net_str())
        except Exception as ex:
            self.text.insert("end", f"Error generating preview:\n{ex}")
        self.text.config(state="disabled")

    # ── Individual exports ─────────────────────────────────────────────────
    def _get_name(self) -> str:
        n = self.name_var.get().strip()
        S.project_name = n or S.project_name
        return S.project_name

    def _export_kicad(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".net",
            initialfile=f"{self._get_name()}.net",
            filetypes=[("KiCad Netlist", "*.net"), ("All", "*.*")],
        )
        if path:
            with open(path, "w") as f:
                f.write(self._kicad_net_str())
            self.status(f"KiCad netlist saved: {path}", "success")

    def _export_kicad_pcb(self):
        """Export a minimal KiCad PCB file with pad placement."""
        path = filedialog.asksaveasfilename(
            defaultextension=".kicad_pcb",
            initialfile=f"{self._get_name()}.kicad_pcb",
            filetypes=[("KiCad PCB", "*.kicad_pcb"), ("All", "*.*")],
        )
        if not path:
            return
        DPI   = 100          # assumed scan DPI
        PX_MM = 25.4 / DPI   # 1 px = 0.254 mm at 100 dpi

        lines = [
            "(kicad_pcb (version 20221018) (generator pcb_re_tool)",
            "  (general)",
            "  (layers",
            "    (0 \"F.Cu\" signal)",
            "    (31 \"B.Cu\" signal)",
            "    (37 \"F.SilkS\" user)",
            "    (44 \"Edge.Cuts\" user)",
            "  )",
            "  (setup)",
            "  (footprints",
        ]
        for g in S.groups:
            pins_in = [p for p in S.pins if p.get("group_id") == g["id"]]
            if not pins_in:
                continue
            cx_mm = (sum(p["x"] for p in pins_in) / len(pins_in)) * PX_MM
            cy_mm = (sum(p["y"] for p in pins_in) / len(pins_in)) * PX_MM
            lines.append(f'    (footprint "{g["name"]}" (layer "F.Cu")')
            lines.append(f'      (at {cx_mm:.4f} {cy_mm:.4f})')
            lines.append(f'      (reference "{g["name"]}")')
            for i, p in enumerate(pins_in):
                rel_x = (p["x"] * PX_MM) - cx_mm
                rel_y = (p["y"] * PX_MM) - cy_mm
                pname = p.get("pin_name") or str(i + 1)
                lines.append(
                    f'      (pad "{pname}" smd circle '
                    f'(at {rel_x:.4f} {rel_y:.4f}) '
                    f'(size 0.8 0.8) '
                    f'(layers "F.Cu" "F.Paste" "F.Mask"))'
                )
            lines.append("    )")
        lines += ["  )", ")"]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        self.status(f"KiCad PCB file saved: {path}", "success")

    def _export_pins_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"{self._get_name()}_pins.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "x", "y", "block_name", "pin_name", "group_id", "net_id", "net_label"])
            for p in S.pins:
                net = next((n for n in S.nets if n["id"] == p.get("net_id")), None)
                w.writerow([
                    p["id"], round(p["x"], 2), round(p["y"], 2),
                    p.get("block_name", ""), p.get("pin_name", ""),
                    p.get("group_id", ""),    p.get("net_id", ""),
                    net["label"] if net else "NC",
                ])
        self.status(f"Pins CSV saved: {path}", "success")

    def _export_nets_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"{self._get_name()}_nets.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["net_id", "net_label", "color", "pin_count", "pin_ids"])
            for n in S.nets:
                w.writerow([
                    n["id"], n["label"], n["color"],
                    len(n["pin_ids"]),
                    ";".join(str(x) for x in n["pin_ids"]),
                ])
        self.status(f"Nets CSV saved: {path}", "success")

    def _export_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{self._get_name()}.json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        data = {
            "project": self._get_name(),
            "colors":  S.colors,
            "pins":    [{k: v for k, v in p.items() if k != "color"} for p in S.pins],
            "groups":  S.groups,
            "nets":    S.nets,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.status(f"Project JSON saved: {path}", "success")

    def _export_zip(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            initialfile=f"{self._get_name()}.zip",
            filetypes=[("ZIP", "*.zip"), ("All", "*.*")],
        )
        if not path:
            return
        name = self._get_name()
        files_added = []
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:

                # KiCad netlist
                zf.writestr(f"{name}.net", self._kicad_net_str())
                files_added.append(f"{name}.net")

                # Project JSON
                data = {
                    "project": name,
                    "colors":  S.colors,
                    "pins":    [{k: v for k, v in p.items() if k != "color"} for p in S.pins],
                    "groups":  S.groups,
                    "nets":    S.nets,
                }
                zf.writestr("project.json", json.dumps(data, indent=2))
                files_added.append("project.json")

                # Pins CSV
                sio = io.StringIO()
                w   = csv.writer(sio)
                w.writerow(["id", "x", "y", "block_name", "pin_name", "net_id"])
                for p in S.pins:
                    w.writerow([
                        p["id"], round(p["x"], 2), round(p["y"], 2),
                        p.get("block_name", ""), p.get("pin_name", ""),
                        p.get("net_id", ""),
                    ])
                zf.writestr("pins.csv", sio.getvalue())
                files_added.append("pins.csv")

                # Images
                def _add_img(img: "Image.Image", fname: str):
                    buf = BytesIO()
                    img.save(buf, "PNG")
                    buf.seek(0)
                    zf.writestr(fname, buf.read())
                    files_added.append(fname)

                if S.base:
                    _add_img(S.base, "base_image.png")
                if S.seg_arr is not None:
                    _add_img(Image.fromarray(S.seg_arr, "RGBA"), "segmented.png")
                if S.net_img:
                    _add_img(S.net_img, "net_contours.png")

            summary = "\n".join(f"  • {f}" for f in files_added)
            messagebox.showinfo(
                "Export Complete",
                f"ZIP bundle saved to:\n{path}\n\nContains:\n{summary}",
            )
            self.status(f"ZIP bundle saved: {path}", "success")
        except Exception as ex:
            messagebox.showerror("Export Error", str(ex))


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        app = PCBApp()
        app.mainloop()

    except tk.TclError as e:
        print("\nGUI ERROR")
        print("Tkinter could not open a display.\n")

        print("Possible causes:")
        print("- Running inside WSL")
        print("- Running over SSH")
        print("- Running in Docker/container")
        print("- Missing python3-tk package")
        print("- No desktop environment available\n")

        print(f"Original error:\n{e}")