"""
Loader for the STAREVIZ user configuration file (stareviz.yml).

Usage
-----
from stareviz.viz_config import cfg

grid_opacity        = cfg.grid_opacity          # float
grid_color          = cfg.grid_rgba_str         # "rgba(0, 0, 0, 0.05)"
phase_colors        = cfg.phase_colors          # {1: "#0bdbdb", ...}
kipp                = cfg.kipp                  # KippColors object
default_roots       = cfg.default_roots         # ["/path/to/models"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Defaults  (mirror the shipped stareviz.yml)
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "solar_mixture": "AAD",   # GN93 | AGS05 | AGSS09 | AY18 | AAD
    "default_roots": [],       # default model directories (used when stareviz is called without arguments)
    "grid": {
        "opacity": 0.05,
        "color_r": 0,
        "color_g": 0,
        "color_b": 0,
        "width": 1,
    },
    "model_sort_order": "name",   # "name" | "created" | "last_opened"
    "model_sort_ascending": True,  # True = A->Z / oldest first; False = Z->A / newest first
    "isochrone_sort_by":        "logg",  # any parameter from AXIS_LABELS_PLOT; default: logg
    "isochrone_sort_ascending": False,   # False = descending (physically correct for logg)
    "legend": {
        "font_size":   18,   # font size in the legend
        "max_chars":   None, # max characters of model name to show (None = full name)
    },
    "freeze_opacity": 1.0,        # opacity of the frozen background layer (0.0–1.0)
    "track_line_width":      2.0,  # base line width for track plots (pixels)
    "track_line_width_step": 1.5,  # line width increment every 10 models
    "track_multicolor":      True,      # True = Plotly color cycle; False = single color
    "track_single_color":    "#636EFA", # color when track_multicolor is False
    "log_tick_multipliers": [1],  # multipliers N for log-scale ticks at N×10^k
                                  # e.g. [1] -> only 10^k; [1,2,5] -> 1×, 2×, 5×10^k
    "axis_title_font_size": 24,  # font size of axis titles
    "axis_tick_font_size":  20,  # font size of axis tick labels

    "isochrone": {
        "marker_color":  "red",    # colour of age marker dot
        "marker_size":   10,       # size of age marker dot in pixels
        "marker_border_width": 2,  # border thickness of age marker dot
        "line_color":    "black",  # colour of isochrone line
        "line_width":    2,        # thickness of isochrone line
    },
    "phase_colors": {
        1: "#0bdbdb",
        2: "black",
        3: "#07d63b",
        4: "blue",
        5: "red",
    },
    "kipp": {
        # Convective zones (envelope + additional zones conv1-conv5)
        "conv_fill":        "rgba(0,128,0,0.18)",    # green fill inside convective zone
        "conv_line":        "rgba(0,100,0,0.35)",    # green border of convective zone
        # Thermohaline mixing zone (conv6)
        "show_thermo":      True,
        "thermo_fill":      "rgba(250,0,100,0.4)",   # reddish fill
        "thermo_line":      "rgba(255,0,30,0.6)",    # reddish border
        # Stipple rings (hatching inside convective zones)
        "rings":            "rgba(0,90,0,0.40)",
        "rings_size":       2,
        "rings_line_width": 1,
        "rings_ny":         100,
        # Boundary lines (base of envelope, stellar surface)
        "boundary_line":    "rgba(0,90,0,0.70)",
        # Radiative-zone masks per evolutionary phase (opacity shared)
        "mask_opacity":     0.5,
        "mask_pms":         "173,216,230",   # light blue   (phase 1)
        "mask_ms":          "255,255,204",   # light yellow (phase 2)
        "mask_gb":         "255,200,120",   # orange       (phase 3)
        "mask_hb":         "255,140,100",   # salmon       (phase 4)
        "mask_agb":    "255,100,140",   # pink         (phase 5)
        # Smooth phase transitions
        "transition_steps": 0,     # 0 = sharp; 10 = smooth gradient
        "transition_frac":  0.01,  # default fraction of left-phase duration
        "transition_frac_pms":  0.05, # fraction for PMS->MS transition
    },
}

_SEARCH_PATHS: list[Path] = [
    Path(__file__).parent / "stareviz.yml",
    Path.cwd() / "stareviz.yml",
    Path.home() / ".config" / "stareviz.yml",
]


def _find_config_file() -> Optional[Path]:
    env = os.environ.get("STAREVIZ_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for p in _SEARCH_PATHS:
        if p.is_file():
            return p
    return None


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_raw() -> dict:
    if not _YAML_AVAILABLE:
        print("[stareviz] WARNING: PyYAML not installed -- using built-in defaults.")
        return _DEFAULTS

    path = _find_config_file()
    if path is None:
        print(
            "[stareviz] stareviz.yml not found. Searched:\n"
            + "\n".join(f"  {p}" for p in _SEARCH_PATHS)
            + "\nUsing built-in defaults."
        )
        return _DEFAULTS

    try:
        with open(path, "r", encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}
        merged = _deep_merge(_DEFAULTS, user_data)
        print(f"[stareviz] Loaded config from: {path}")
        return merged
    except Exception as exc:
        import warnings
        warnings.warn(f"[stareviz] Could not load config from {path}: {exc}. Using defaults.")
        return _DEFAULTS


# ---------------------------------------------------------------------------
# Typed config objects
# ---------------------------------------------------------------------------

class _KippColors:
    """Pre-parsed Kippenhahn colour values."""

    def __init__(self, raw: dict) -> None:
        k = raw.get("kipp", _DEFAULTS["kipp"])
        d = _DEFAULTS["kipp"]

        self.conv_fill:      str   = str(k.get("conv_fill",     d["conv_fill"]))
        self.conv_line:      str   = str(k.get("conv_line",     d["conv_line"]))
        self.show_thermo: bool = bool(k.get("show_thermo", d["show_thermo"]))
        self.thermo_fill:    str   = str(k.get("thermo_fill",   d["thermo_fill"]))
        self.thermo_line:    str   = str(k.get("thermo_line",   d["thermo_line"]))
        self.rings:          str   = str(k.get("rings",         d["rings"]))
        self.rings_size:     int   = int(k.get("rings_size",   d["rings_size"]))
        self.rings_line_width: float = float(k.get("rings_line_width", d["rings_line_width"]))
        self.rings_ny:       int   = int(k.get("rings_ny",     d["rings_ny"]))
        self.boundary_line:  str   = str(k.get("boundary_line", d["boundary_line"]))

        self.mask_opacity:   float = float(k.get("mask_opacity", d["mask_opacity"]))

        # Smooth transition parameters
        self.transition_steps:    int   = int(k.get("transition_steps",    d["transition_steps"]))
        self.transition_frac:     float = float(k.get("transition_frac",     d["transition_frac"]))
        self.transition_frac_pms: float = float(k.get("transition_frac_pms", d["transition_frac_pms"]))

        # Build per-phase mask rgba strings
        def _mask(key: str, default: str) -> str:
            rgb = str(k.get(key, default))
            return f"rgba({rgb},{self.mask_opacity})"

        self.mask_pms:      str = _mask("mask_pms",      d["mask_pms"])
        self.mask_ms:       str = _mask("mask_ms",       d["mask_ms"])
        self.mask_gb:      str = _mask("mask_gb",      d["mask_gb"])
        self.mask_hb:      str = _mask("mask_hb",      d["mask_hb"])
        self.mask_agb: str = _mask("mask_agb", d["mask_agb"])

    def stage_specs(self) -> list[tuple]:
        """
        Returns the stage_specs list used in app.py:
            [(phase_set, rgba_string), ...]
        """
        return [
            ({1}, self.mask_pms),
            ({2}, self.mask_ms),
            ({3}, self.mask_gb),
            ({4}, self.mask_hb),
            ({5}, self.mask_agb),
        ]


class _VizConfig:
    """Thin wrapper that exposes typed, pre-computed attributes."""

    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self._init_grid()
        self._init_phase_colors()
        self.solar_mixture: str = str(raw.get("solar_mixture", _DEFAULTS["solar_mixture"])).upper()
        self.model_sort_order:     str  = str(raw.get("model_sort_order",     _DEFAULTS["model_sort_order"]))
        self.model_sort_ascending:    bool = bool(raw.get("model_sort_ascending",    _DEFAULTS["model_sort_ascending"]))
        self.isochrone_sort_by:        str  = str(raw.get("isochrone_sort_by",        _DEFAULTS["isochrone_sort_by"]))
        self.isochrone_sort_ascending: bool = bool(raw.get("isochrone_sort_ascending", _DEFAULTS["isochrone_sort_ascending"]))
        self._init_isochrone()
        self._init_legend()
        self._init_default_roots()
        raw_ltm = self._raw.get("log_tick_multipliers", _DEFAULTS["log_tick_multipliers"])
        self.log_tick_multipliers: list = [float(x) for x in raw_ltm]
        self.freeze_opacity: float = float(self._raw.get("freeze_opacity", _DEFAULTS["freeze_opacity"]))
        self.track_line_width:      float = float(self._raw.get("track_line_width",      _DEFAULTS["track_line_width"]))
        self.track_line_width_step: float = float(self._raw.get("track_line_width_step", _DEFAULTS["track_line_width_step"]))
        self.track_multicolor:   bool = bool(self._raw.get("track_multicolor",   _DEFAULTS["track_multicolor"]))
        self.track_single_color: str  = str(self._raw.get("track_single_color",  _DEFAULTS["track_single_color"]))
        self.axis_title_font_size: int = int(self._raw.get("axis_title_font_size", _DEFAULTS["axis_title_font_size"]))
        self.axis_tick_font_size:  int = int(self._raw.get("axis_tick_font_size",  _DEFAULTS["axis_tick_font_size"]))
        self.kipp = _KippColors(raw)

    def _init_grid(self) -> None:
        g = self._raw.get("grid", _DEFAULTS["grid"])
        self.grid_opacity: float = float(g.get("opacity", 0.05))
        self.grid_color_r: int   = int(g.get("color_r", 0))
        self.grid_color_g: int   = int(g.get("color_g", 0))
        self.grid_color_b: int   = int(g.get("color_b", 0))
        self.grid_width:   int   = int(g.get("width",   1))

    @property
    def grid_rgba_str(self) -> str:
        return (
            f"rgba({self.grid_color_r}, {self.grid_color_g}, "
            f"{self.grid_color_b}, {self.grid_opacity})"
        )

    def _init_phase_colors(self) -> None:
        raw_pc = self._raw.get("phase_colors", _DEFAULTS["phase_colors"])
        self.phase_colors: Dict[int, str] = {
            int(k): str(v) for k, v in raw_pc.items()
        }

    def _init_legend(self) -> None:
        lg = self._raw.get("legend", _DEFAULTS["legend"])
        d  = _DEFAULTS["legend"]
        self.legend_font_size: int        = int(lg.get("font_size", d["font_size"]))
        raw_mc = lg.get("max_chars", d["max_chars"])
        self.legend_max_chars: Optional[int] = None if raw_mc is None else int(raw_mc)

    def _init_isochrone(self) -> None:
        iso = self._raw.get("isochrone", _DEFAULTS["isochrone"])
        d   = _DEFAULTS["isochrone"]
        self.isochrone_marker_color:        str   = str(iso.get("marker_color",        d["marker_color"]))
        self.isochrone_marker_size:         int   = int(iso.get("marker_size",         d["marker_size"]))
        self.isochrone_marker_border_width: int   = int(iso.get("marker_border_width", d["marker_border_width"]))
        self.isochrone_line_color:          str   = str(iso.get("line_color",          d["line_color"]))
        self.isochrone_line_width:          float = float(iso.get("line_width",        d["line_width"]))

    def _init_default_roots(self) -> None:
        raw_roots = self._raw.get("default_roots", _DEFAULTS["default_roots"])
        # Allow a single string as well as a list
        if isinstance(raw_roots, str):
            raw_roots = [raw_roots]
        self.default_roots: List[str] = [
            str(Path(r).expanduser()) for r in (raw_roots or [])
        ]

    def reload(self) -> None:
        """Re-read the config file from disk."""
        raw = _load_raw()
        self.__init__(raw)

    @property
    def chem_comp(self) -> dict:
        """Return the MODEL_CHEM_COMP dict for the configured solar_mixture."""
        from stareviz.constants import ALL_CHEM_COMPS
        _valid = tuple(ALL_CHEM_COMPS.keys())
        key = self.solar_mixture
        if key not in ALL_CHEM_COMPS:
            import warnings
            warnings.warn(
                f"[stareviz] Unknown solar_mixture '{key}'. "
                f"Valid options: {_valid}. Falling back to AY18."
            )
            key = "AY18"
        return ALL_CHEM_COMPS[key]

    def __repr__(self) -> str:
        return (
            f"<VizConfig grid_rgba='{self.grid_rgba_str}' "
            f"phase_colors={self.phase_colors}>"
        )


cfg = _VizConfig(_load_raw())