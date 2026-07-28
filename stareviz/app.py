import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context, Patch, no_update, MATCH, ALL
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .constants import (
    AXIS_LABELS_DROPDOWN,
    AXIS_LABELS_PLOT,
    PARAMETER_CATEGORIES,
    SURF_METAL_ISOTOPE_COLS,
    ATOMIC_MASSES,
    AXIS_SCALE_CHOICES,
    A_ELEM_SUN,
    ISOTOPE_TO_ELEMENT,
    HYDROGEN_ISOTOPES,
)
from .viz_config import cfg

# Solar composition selected via stareviz.yml → solar_mixture
MODEL_CHEM_COMP = cfg.chem_comp
from .utils import (
    get_axis_scale_options,
    get_default_axis_scale,
    get_effective_axis_scale,
)

from stareviz.registry import ModelRegistry, record_opened, parse_model_path
from stareviz.loader import load_hr_only, load_kipp_model, load_surf_model

import logging

logging.getLogger("werkzeug").setLevel(logging.ERROR)


def _solar_zx_from_model_comp() -> float:
    """(Z/X)_sun from MODEL_CHEM_COMP, consistent with SURF_METAL_ISOTOPE_COLS definition."""
    x_sun = float(MODEL_CHEM_COMP["H1"]) + float(MODEL_CHEM_COMP["H2"])
    z_sun = 0.0
    for col in SURF_METAL_ISOTOPE_COLS:
        key = col.replace("s_", "", 1)  # e.g. "O16", "Mg24", "heavy"
        z_sun += float(MODEL_CHEM_COMP.get(key, 0.0))
    return z_sun / x_sun


def _solar_alpha_over_h_from_model_comp(elements: tuple[str, ...]) -> float:
    """(N_alpha/N_H)_sun from MODEL_CHEM_COMP, using isotopic mass fractions and ATOMIC_MASSES."""
    n_h = float(MODEL_CHEM_COMP["H1"]) / float(ATOMIC_MASSES["H1"]) + float(MODEL_CHEM_COMP["H2"]) / float(
        ATOMIC_MASSES["H2"])

    n_alpha = 0.0
    for el in elements:
        for iso, x_iso in MODEL_CHEM_COMP.items():
            if not iso.startswith(el):
                continue
            at_mass = ATOMIC_MASSES.get(iso)
            if at_mass is None:
                # e.g. Ne23 may exist in MODEL_CHEM_COMP but not in ATOMIC_MASSES -> skip
                continue
            n_alpha += float(x_iso) / float(at_mass)

    return n_alpha / n_h


def compute_mh_surf(df: pd.DataFrame) -> pd.Series:
    """Compute [M/H] from surface mass fractions.

    [M/H] = log10( (Z/X) / (Z/X)_sun )

    Where:
      X = X(H1) + X(H2)
      Z = sum(metals from Li6..Cl37) + heavy

    If the required columns are not present (e.g. HR-only fast path),
    returns a NaN series.
    """
    required = ["s_H1", "s_H2"] + SURF_METAL_ISOTOPE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.Series(np.nan, index=df.index)

    x = pd.to_numeric(df["s_H1"], errors="coerce") + pd.to_numeric(df["s_H2"], errors="coerce")
    z = df[SURF_METAL_ISOTOPE_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    zx = z / x
    mh = np.log10(zx / _solar_zx_from_model_comp())

    mh = mh.replace([np.inf, -np.inf], np.nan)
    return mh


# ==========================================================
def compute_alpha_h_surf(df: pd.DataFrame, elements: tuple[str, ...]) -> pd.Series:
    """Compute [α/H] from surface mass fractions.

    [α/H] = log10( (N_alpha/N_H)_star / (N_alpha/N_H)_sun )

    N_H = X(H1)/A(H1) + X(H2)/A(H2)
    N_alpha = sum_{el in elements} sum_{isotopes of el} X(isotope)/A(isotope)

    Solar reference uses A_ELEM_SUN (log epsilon with A(H)=12):
      (N_el/N_H)_sun = 10^(A_el - 12)
    """
    required_h = ["s_H1", "s_H2"]
    if any(c not in df.columns for c in required_h):
        return pd.Series(np.nan, index=df.index)

    x_h1 = pd.to_numeric(df["s_H1"], errors="coerce")
    x_h2 = pd.to_numeric(df["s_H2"], errors="coerce")

    n_h = x_h1 / float(ATOMIC_MASSES["H1"]) + x_h2 / float(ATOMIC_MASSES["H2"])

    n_alpha = pd.Series(0.0, index=df.index, dtype="float64")

    for el in elements:
        cols = [c for c in df.columns if c.startswith(f"s_{el}")]
        if not cols:
            return pd.Series(np.nan, index=df.index)

        for c in cols:
            key = c.replace("s_", "", 1)  # e.g. "O16"
            at_mass = ATOMIC_MASSES.get(key)
            if at_mass is None:
                return pd.Series(np.nan, index=df.index)
            n_alpha = n_alpha + (pd.to_numeric(df[c], errors="coerce") / float(at_mass))

    # (N_alpha/N_H)_sun = sum 10^(A_el - 12)
    n_alpha_over_h_sun = _solar_alpha_over_h_from_model_comp(elements)

    ratio_star = (n_alpha / n_h).replace([np.inf, -np.inf], np.nan)
    alpha_h = np.log10(ratio_star / n_alpha_over_h_sun)
    alpha_h = alpha_h.replace([np.inf, -np.inf], np.nan)
    return alpha_h


def compute_number_ratio(df: pd.DataFrame, num: str, den: str) -> pd.Series:
    """Compute isotope number ratio (X_num/A_num) / (X_den/A_den)."""
    col_num = f"s_{num}"
    col_den = f"s_{den}"

    if col_num not in df.columns or col_den not in df.columns:
        return pd.Series(np.nan, index=df.index)

    at_mass_num = ATOMIC_MASSES.get(num)
    at_mass_den = ATOMIC_MASSES.get(den)
    if at_mass_num is None or at_mass_den is None:
        return pd.Series(np.nan, index=df.index)

    x_num = pd.to_numeric(df[col_num], errors="coerce")
    x_den = pd.to_numeric(df[col_den], errors="coerce")

    ratio = (x_num / at_mass_num) / (x_den / at_mass_den)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _parse_abundance_key(col: str):
    """Parse a virtual abundance key like 'surf:Li7' or 'cent:He4'.

    Returns (location, isotope) or (None, None) if not an abundance key.
    location is 'surf' or 'cent'.
    """
    if col and ":" in col:
        parts = col.split(":", 1)
        if parts[0] in ("surf", "cent"):
            return parts[0], parts[1]
    return None, None


def resolve_abundance_col(col: str, fmt: str, df: pd.DataFrame) -> pd.Series:
    """Resolve a virtual abundance key to a data series given format string.

    fmt is one of: 'XX' (X(X)), 'AX' (A(X)), 'XH' ([X/H]), 'XM' ([X/M]).
    Returns a pd.Series with the requested values.
    """
    location, isotope = _parse_abundance_key(col)
    if location is None:
        return pd.Series(np.nan, index=df.index)

    prefix_x = "s_" if location == "surf" else "c_"
    prefix_a = "A_" if location == "surf" else "Ac_"

    # Special case: Li7 surface A column is stored as A_Li (backward compat alias)
    if isotope == "Li7" and location == "surf":
        a_col = "A_Li" if "A_Li" in df.columns else "A_Li7"
    else:
        a_col = f"{prefix_a}{isotope}"

    x_col = f"{prefix_x}{isotope}"

    if fmt == "XX":
        if x_col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[x_col], errors="coerce")

    elif fmt == "AX":
        if a_col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[a_col], errors="coerce")

    elif fmt in ("XH", "XM"):
        # [X/H] = A(X)_star - A(X)_solar
        if a_col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        elem = ISOTOPE_TO_ELEMENT.get(isotope)
        if elem is None or elem not in A_ELEM_SUN:
            return pd.Series(np.nan, index=df.index)
        a_star = pd.to_numeric(df[a_col], errors="coerce")
        xh = a_star - A_ELEM_SUN[elem]
        if fmt == "XH":
            return xh
        else:  # [X/M] = [X/H] - [M/H]
            return xh - compute_mh_surf(df)

    return pd.Series(np.nan, index=df.index)


def _format_isotope_latex(isotope: str) -> str:
    """Convert isotope string like 'Li7', 'He4', 'C12' to LaTeX like '^{7}\\mathrm{Li}'."""
    import re as _re
    m = _re.match(r'^([A-Za-z]+)([0-9]+)$', isotope)
    if m:
        elem, mass = m.group(1), m.group(2)
        return rf"^{{{mass}}}\mathrm{{{elem}}}"
    return rf"\mathrm{{{isotope}}}"


def _abundance_axis_title(col: str, fmt: str) -> str:
    """Build axis title for a virtual abundance column given format (LaTeX)."""
    _, isotope = _parse_abundance_key(col)
    if isotope is None:
        return col
    iso_latex = _format_isotope_latex(isotope)
    suffix_key = "surf" if col.startswith("surf:") else "cent"
    suffix = rf"_{{\mathrm{{{suffix_key}}}}}"
    fmt_labels = {
        "XX": rf"$X({iso_latex}){suffix}$",
        "AX": rf"$A({iso_latex}){suffix}$",
        "XH": rf"$[{iso_latex}/\mathrm{{H}}]{suffix}$",
        "XM": rf"$[{iso_latex}/\mathrm{{M}}]{suffix}$",
    }
    return fmt_labels.get(fmt, col)


def _render_search_results(res: pd.DataFrame, currently_picked: list) -> list:
    """Build grouped html structure for the search dropdown."""
    picked_set = set(currently_picked or [])
    children = []

    # Maintain original sort order while grouping same-folder models together
    seen_folders: list[str] = []
    folder_rows: dict[str, list] = {}
    for _, row in res.iterrows():
        fp = row["folder_path"]
        if fp not in folder_rows:
            folder_rows[fp] = []
            seen_folders.append(fp)
        folder_rows[fp].append(row)

    for fp in seen_folders:
        rows = folder_rows[fp]
        folder_name = rows[0]["folder_name"]
        models = [(r["name"], r["path"]) for r in rows]

        if len(models) == 1:
            name, path = models[0]
            children.append(
                dcc.Checklist(
                    id={"type": "search-group-cb", "index": fp},
                    options=[{"label": name, "value": path}],
                    value=[path] if path in picked_set else [],
                    style={"fontSize": "16px"},
                    labelStyle={"display": "flex", "alignItems": "center",
                                "gap": "6px", "padding": "3px 0", "cursor": "pointer"},
                    inputStyle={"margin": 0},
                )
            )
        else:
            group_values = [path for _, path in models if path in picked_set]
            is_expanded = bool(group_values)
            children.append(html.Div([
                html.Div([
                    html.Button(
                        "▼" if is_expanded else "▶",
                        id={"type": "group-toggle-btn", "index": fp},
                        n_clicks=0,
                        style={"background": "none", "border": "none", "cursor": "pointer",
                               "padding": "0 3px", "fontSize": "11px", "color": "#555",
                               "lineHeight": "1", "flexShrink": 0},
                    ),
                    html.Span(folder_name, style={
                        "fontWeight": "600", "fontSize": "15px",
                        "overflow": "hidden", "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap", "minWidth": 0,
                    }),
                    html.Span(f" [{len(models)}]", style={
                        "color": "#999", "fontSize": "13px", "flexShrink": 0,
                    }),
                    html.Button(
                        "Load all",
                        id={"type": "load-all-btn", "index": fp},
                        n_clicks=0,
                        style={"marginLeft": "auto", "fontSize": "12px", "padding": "1px 8px",
                               "cursor": "pointer", "border": "1px solid #ccc",
                               "background": "#f5f5f5", "borderRadius": "4px",
                               "flexShrink": 0},
                    ),
                ], style={"display": "flex", "alignItems": "center",
                          "padding": "3px 0", "gap": "4px", "minWidth": 0}),
                html.Div(
                    id={"type": "group-body", "index": fp},
                    style={"display": "block" if is_expanded else "none",
                           "paddingLeft": "14px"},
                    children=[
                        dcc.Checklist(
                            id={"type": "search-group-cb", "index": fp},
                            options=[{"label": name, "value": path} for name, path in models],
                            value=group_values,
                            style={"fontSize": "15px"},
                            labelStyle={"display": "flex", "alignItems": "center",
                                        "gap": "6px", "padding": "2px 0", "cursor": "pointer"},
                            inputStyle={"margin": 0},
                        ),
                    ],
                ),
            ]))
    return children


def build_server(roots: list[str], port: int = 8050):
    registry = ModelRegistry(roots)
    index_df = registry.build_index()

    # --- progressive loading: HR first, full model in background ---
    executor = ThreadPoolExecutor(max_workers=4)
    HR_CACHE: dict[str, pd.DataFrame] = {}
    FULL_CACHE: dict[str, pd.DataFrame] = {}
    FULL_FUTURES = {}  # path -> Future
    KIPP_CACHE: dict[str, pd.DataFrame] = {}   # hr+v3+v4+v12 only; evicted when FULL_CACHE ready
    KIPP_FUTURES = {}  # path -> Future; launched only when kipp mode is active
    SURF_CACHE: dict[str, pd.DataFrame] = {}   # hr+s1-s4 only; evicted when FULL_CACHE ready
    SURF_FUTURES = {}  # path -> Future; launched proactively alongside FULL

    # Cache populated by every full render(): per-model cleaned/sorted interpolation
    # data + the trace indices of the age-marker/isochrone traces in the current figure.
    # Lets a slider-only move patch those traces in place via Patch() instead of
    # rebuilding (and re-shipping to the browser) the entire figure.
    AGE_MARKER_CACHE: dict = {}

    # --- helper: find category for parameter (for axis swap) ---
    param_to_category = {}
    for _cat, _params in PARAMETER_CATEGORIES.items():
        for _p in _params:
            param_to_category[_p] = _cat

    def display_label_dropdown(col: str) -> str:
        """Convert internal column names to display labels for dropdowns"""
        return AXIS_LABELS_DROPDOWN.get(col, col)

    def display_label_plot(col: str) -> str:
        """Convert internal column names to display labels for plot axes"""
        return AXIS_LABELS_PLOT.get(col, col)

    app = Dash(__name__, title="STAREVOL Visualizer", serve_locally=True, suppress_callback_exceptions=True)

    _RELOAD_SCRIPT = r"""
    <script>
    (function () {
        var _s = document.createElement('style');
        _s.textContent = [
            '.rl-btn { display: none; flex-shrink: 0;',
            '  margin-left: 3px; padding: 0; font-size: 15px; line-height: 1;',
            '  vertical-align: text-bottom; position: relative; top: 1px;',
            '  cursor: pointer; background: none; border: none;',
            '  white-space: nowrap; user-select: none; }',
            '#picked-models label:hover .rl-btn { display: block; }',
            '.grp-hdr { display: flex; align-items: center;',
            '  gap: 4px; padding: 2px 0; cursor: pointer; min-width: 0;',
            '  font-weight: 600; font-size: 18px; user-select: none; }',
            '.grp-hdr-toggle { font-size: 12px; flex-shrink: 0; color: #555; }',
            '.grp-hdr-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }',
            '.grp-hdr-count { color: #999; font-size: 14px; flex-shrink: 0; font-weight: normal; }',
            '.grp-spacer { visibility: hidden; pointer-events: none; }',
            '#picked-models > label { display: flex !important; flex-direction: row !important;',
            '  align-items: center !important; flex-wrap: nowrap !important; }',
            '#picked-models > label.grp-hidden { display: none !important; }',
            '#picked-models { grid-auto-flow: row dense; }'
        ].join('\n');
        (document.head || document.documentElement).appendChild(_s);

        var _ns = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;

        /* ── Inject 🔄 reload buttons ── */
        function _inject() {
            var c = document.getElementById('picked-models');
            if (!c) return;
            var cbs = c.querySelectorAll('input[type="checkbox"]');
            if (!cbs.length) return;
            cbs.forEach(function (cb) {
                var item = cb.parentElement;
                if (!item || item.dataset.rlInjected) return;
                var span = item.querySelector('[data-model-path]');
                if (!span) return;
                var modelPath = span.getAttribute('data-model-path');
                item.dataset.rlInjected = '1';
                var b = document.createElement('span');
                b.className = 'rl-btn';
                b.textContent = '\u{1F504}';
                b.title = 'Reload model';
                b.setAttribute('data-rl-path', modelPath);
                item.appendChild(b);
            });
        }

        /* ── Inject group headers for multi-model folders ── */
        var _grpCollapsed = {};   /* fp -> collapsed boolean, persists across rebuilds */
        var _lastLabelNodes = []; /* label elements processed on the previous run */

        function _injectGroups() {
            var c = document.getElementById('picked-models');
            if (!c) return;

            var labels = Array.from(c.querySelectorAll('label'));
            if (!labels.length) return;

            /* Skip only if both the data-fp sequence AND the actual label
               DOM nodes are unchanged. React sometimes replaces label nodes
               wholesale (same data-fp values, new elements) when re-rendering
               the checklist — in that case the new nodes won't carry our
               grid placement, so we must rebuild even though the signature
               string matches. */
            var sig = labels.map(function(l){
                var s = l.querySelector('[data-fp]');
                return s ? s.getAttribute('data-fp') : '';
            }).join('|');
            var sameNodes = labels.length === _lastLabelNodes.length &&
                labels.every(function(l, i){ return l === _lastLabelNodes[i]; });
            if (c.dataset.grpSig === sig && sameNodes) return;
            c.dataset.grpSig = sig;
            _lastLabelNodes = labels;

            /* Remove stale headers */
            c.querySelectorAll('.grp-hdr').forEach(function(h){ h.remove(); });

            /* Collect groups: fp → {folderName, labels, firstLabel} keeping order */
            var fpOrder = [], groups = {};
            labels.forEach(function(lbl) {
                var span = lbl.querySelector('[data-fp]');
                if (!span) return;
                var fp = span.getAttribute('data-fp');
                var fn = span.getAttribute('data-fn') || fp;
                if (!groups[fp]) { groups[fp] = {fn: fn, labels: [], first: lbl}; fpOrder.push(fp); }
                groups[fp].labels.push(lbl);
            });

            /* Simulate 2-column auto-flow to know which column each group
               header will land in, so its items can be placed in the same
               column. Every entry (single-model or group header) counts as
               1 cell; col toggles between 0 (→col1) and 1 (→col2). */
            var col = 0;
            fpOrder.forEach(function(fp) {
                groups[fp].headerCol = col + 1;   /* 1-based CSS column */
                col = 1 - col;
            });

            fpOrder.forEach(function(fp) {
                var g = groups[fp];
                if (g.labels.length <= 1) return;   /* single-model folder: no header */

                var hdr = document.createElement('div');
                hdr.className = 'grp-hdr';
                hdr.dataset.fp = fp;

                var tog = document.createElement('span');
                tog.className = 'grp-hdr-toggle';

                var nm = document.createElement('span');
                nm.className = 'grp-hdr-name';
                nm.textContent = g.fn;

                var cnt = document.createElement('span');
                cnt.className = 'grp-hdr-count';
                cnt.textContent = ' [' + g.labels.length + ']';

                hdr.appendChild(tog); hdr.appendChild(nm); hdr.appendChild(cnt);

                c.insertBefore(hdr, g.first);
                /* Place group items in the same column as their header. */
                g.labels.forEach(function(lbl) {
                    lbl.style.gridColumn = String(g.headerCol);
                    var sp = lbl.querySelector('[data-model-path]');
                    if (sp) sp.style.fontSize = '15px';
                });

                /* Restore previous collapse state, defaulting to collapsed */
                var collapsed = (fp in _grpCollapsed) ? _grpCollapsed[fp] : true;
                _grpCollapsed[fp] = collapsed;
                tog.textContent = collapsed ? '►' : '▼';
                g.labels.forEach(function(lbl) { lbl.classList.toggle('grp-hidden', collapsed); });

                hdr.addEventListener('click', function() {
                    collapsed = !collapsed;
                    _grpCollapsed[fp] = collapsed;
                    tog.textContent = collapsed ? '►' : '▼';
                    g.labels.forEach(function(lbl) {
                        lbl.classList.toggle('grp-hidden', collapsed);
                    });
                });
            });
        }

        document.addEventListener('click', function (e) {
            var b = e.target.closest && e.target.closest('.rl-btn');
            if (!b) return;
            var modelPath = b.getAttribute('data-rl-path');
            if (!modelPath) return;
            e.preventDefault();
            e.stopPropagation();
            setTimeout(function () {
                var inp = document.getElementById('reload-idx-input');
                if (!inp) return;
                _ns.call(inp, modelPath + ':' + Date.now());
                inp.dispatchEvent(new Event('input', {bubbles: true}));
            }, 0);
        }, true);

        function _run() { _inject(); _injectGroups(); }
        new MutationObserver(_run).observe(document.documentElement, {childList: true, subtree: true});
        _run();
    })();
    </script>
    """

    app.index_string = (
        '<!DOCTYPE html>\n'
        '<html>\n'
        '    <head>\n'
        '        {%metas%}\n'
        '        <title>{%title%}</title>\n'
        '        {%favicon%}\n'
        '        {%css%}\n'
        '    </head>\n'
        '    <body>\n'
        '        {%app_entry%}\n'
        '        <footer>\n'
        '            {%config%}\n'
        '            {%scripts%}\n'
        '            {%renderer%}\n'
        '        </footer>\n'
        + _RELOAD_SCRIPT +
        '    </body>\n'
        '</html>\n'
    )

    # Close search dropdown when clicking outside the search wrapper
    app.clientside_callback(
        """
        function(n) {
            if (!n) return window.dash_clientside.no_update;
            setTimeout(function() {
                document.addEventListener('click', function handler(e) {
                    var wrapper = document.getElementById('search-wrapper');
                    if (wrapper && !wrapper.contains(e.target)) {
                        var input = document.getElementById('search');
                        if (input && input.value !== '') {
                            var setter = Object.getOwnPropertyDescriptor(
                                window.HTMLInputElement.prototype, 'value').set;
                            setter.call(input, '');
                            input.dispatchEvent(new Event('input', {bubbles: true}));
                        }
                        document.removeEventListener('click', handler, true);
                    }
                }, true);
            }, 100);
            return window.dash_clientside.no_update;
        }
        """,
        Output("search-dropdown", "id"),
        Input("search-dropdown", "style"),
        prevent_initial_call=True,
    )

    app.layout = html.Div(
        style={"paddingTop": "2px", "paddingLeft": "16px", "paddingRight": "16px", "paddingBottom": "16px"}, children=[
            html.Img(src="/assets/logo.png", style={"height": "120px", "marginBottom": "4px", "display": "block"}),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "500px 1fr",
                    "columnGap": "50px",
                    "alignItems": "start",
                },
                children=[
                    html.Div([
                        html.Div(
                            html.Label("Search models (by folder name)", style={"fontSize": "18px"}),
                            style={"display": "flex", "alignItems": "center", "marginBottom": "2px"},
                        ),
                        html.Div(
                            id="search-wrapper",
                            style={"position": "relative"},
                            children=[
                                dcc.Input(
                                    id="search",
                                    placeholder="e.g. Z014227_MedRot",
                                    type="text",
                                    debounce=False,
                                    style={"width": "100%", "fontSize": "16px"},
                                ),
                                html.Div(
                                    id="search-dropdown",
                                    style={
                                        "display": "none",
                                        "position": "absolute",
                                        "top": "100%",
                                        "left": "0",
                                        "width": "100%",
                                        "zIndex": "1000",
                                        "background": "white",
                                        "border": "1px solid #ddd",
                                        "boxShadow": "0 4px 12px rgba(0,0,0,0.12)",
                                        "maxHeight": "240px",
                                        "overflowY": "auto",
                                        "padding": "6px",
                                    },
                                    children=[],
                                ),
                            ],
                        ),
                    ]),

                    html.Div([
                        # Header row: label + fast mode + clear button — mirrors the label row of left column
                        html.Div(
                            [
                                html.Span("Loaded models", style={"fontSize": "18px"}),
                                dcc.Checklist(
                                    id="fast-hr-mode",
                                    options=[{"label": " Fast mode (load only .hr files)", "value": "fast"}],
                                    value=[],
                                    style={"fontSize": "16px", "marginLeft": "20px"},
                                    labelStyle={"display": "flex", "alignItems": "center", "gap": "4px"},
                                ),
                                html.Div(
                                    id="load-status",
                                    style={
                                        "color": "#777",
                                        "fontSize": "13px",
                                        "lineHeight": "18px",
                                        "whiteSpace": "nowrap",
                                        "marginLeft": "20px",
                                    },
                                ),
                                html.Button(
                                    "Clear selection",
                                    id="btn-clear-models",
                                    n_clicks=0,
                                    title="Uncheck all selected models",
                                    style={
                                        "marginLeft": "auto",
                                        "padding": "2px 8px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #ddd",
                                        "cursor": "pointer",
                                        "fontSize": "16px",
                                        "background": "white",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                                "marginBottom": "2px",
                            },
                        ),

                        html.Div(
                            style={
                                "height": "95px",
                                "overflowY": "auto",
                                "border": "1px solid #ddd",
                                "padding": "6px",
                                "resize": "vertical",
                            },
                            children=[
                                dcc.Checklist(
                                    id="picked-models",
                                    value=[],
                                    options=[
                                        {
                                            "label": html.Span(
                                                n,
                                                style={
                                                    "display": "block",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                    "whiteSpace": "nowrap",
                                                    "minWidth": 0,
                                                },
                                                **{
                                                    "data-fp":   fp,
                                                    "data-fn":   fn,
                                                    "data-model-path": p,
                                                },
                                            ),
                                            "value": p,
                                        }
                                        for n, p, fp, fn in zip(
                                            index_df["name"],
                                            index_df["path"],
                                            index_df["folder_path"],
                                            index_df["folder_name"],
                                        )
                                    ],
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                        "gridAutoFlow": "row dense",
                                        "gap": "6px",
                                        "fontSize": "18px",
                                        "lineHeight": "1.1",
                                    },
                                    labelStyle={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "4px",
                                        "marginBottom": "0px",
                                        "minWidth": 0,
                                    },
                                    inputStyle={"margin": 0},
                                ),
                            ],
                        ),

                    ]),
                ],
            ),

            html.Hr(),

            # Main container with Age slider on the left and axis controls on the right
            html.Div(style={"display": "flex", "gap": "20px", "alignItems": "start", "marginLeft": "15px",
                            "position": "relative", "paddingTop": "0px", "paddingBottom": "0px"}, children=[
                # --- Age slider (LEFT SIDE) ---
                html.Div(style={"width": "550px", "paddingTop": "0px"}, children=[
                    html.Label("Age (Myr)", style={"fontSize": "18px", "marginBottom": "8px"}),

                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "10px"},
                        children=[
                            html.Div(style={"marginLeft": "-20px", "marginTop": "2px"}, children=[
                                dcc.Slider(
                                    id="age-slider",
                                    min=0,
                                    max=1e10,
                                    value=0,
                                    step=1e6,
                                    marks={},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ]),

                            html.Div(style={"display": "flex", "gap": "15px", "alignItems": "center"}, children=[
                                dcc.Input(
                                    id="age-input",
                                    type="number",
                                    value=0,
                                    min=0,
                                    debounce=True,
                                    style={"width": "120px", "fontSize": "16px", "padding": "4px 8px"},
                                ),

                                html.Div(id="isochrone-checkbox-container", style={"display": "none"}, children=[
                                    dcc.Checklist(
                                        id="show-isochrone",
                                        options=[{"label": "Show isochrone", "value": "on"}],
                                        value=[],
                                        style={"fontSize": "16px"},
                                    ),
                                ]),
                            ]),
                        ],
                    ),
                ]),

                # --- Axis controls (RIGHT SIDE) ---
                html.Div(style={"display": "flex", "gap": "12px", "alignItems": "start", "marginLeft": "2px"},
                         children=[

                             # X axis: buttons + dropdown wrapped together with tight gap
                             html.Div(style={"display": "flex", "gap": "2px", "alignItems": "start"}, children=[
                                 # X axis format selectors
                                 html.Div(style={"paddingTop": "22px", "width": "60px", "flexShrink": "0"}, children=[
                                     html.Div(id="x-age-units-container", style={"display": "none"}, children=[
                                         dcc.RadioItems(
                                             id="x-age-units",
                                             options=[
                                                 {"label": "Gyr", "value": "gyr"},
                                                 {"label": "Myr", "value": "myr"},
                                                 {"label": "yr", "value": "yr"},
                                             ],
                                             value="gyr",
                                             style={"fontSize": "14px"}
                                         )
                                     ]),
                                     html.Div(id="x-abund-format-container", style={"display": "none"}, children=[
                                         dcc.RadioItems(
                                             id="x-abund-format",
                                             options=[
                                                 {"label": "X(X)", "value": "XX"},
                                                 {"label": "A(X)", "value": "AX"},
                                                 {"label": "[X/H]", "value": "XH"},
                                                 {"label": "[X/M]", "value": "XM"},
                                             ],
                                             value="XX",
                                             style={"fontSize": "14px"},
                                         )
                                     ]),
                                 ]),
                                 # X axis
                                 html.Div(style={"width": "200px"}, children=[
                                     html.Label("X axis", style={"fontSize": "18px"}),
                                     dcc.Dropdown(
                                         id="xcategory",
                                         options=[{"label": cat, "value": cat} for cat in PARAMETER_CATEGORIES.keys()],
                                         value="Basic",
                                         clearable=False,
                                         placeholder="Select category...",
                                         style={"fontSize": "14px", "marginBottom": "4px"}
                                     ),
                                     dcc.Dropdown(
                                         id="xcol",
                                         value="Teff",
                                         clearable=False,
                                         style={"fontSize": "16px"}
                                     ),
                                     dcc.RadioItems(
                                         id="xscale",
                                         options=[
                                             {"label": "Lin", "value": "lin"},
                                             {"label": "Log", "value": "log"},
                                             {"label": "Log(X)", "value": "log(x)"}],
                                         value="lin",
                                         inline=True,
                                         style={"marginTop": "6px", "fontSize": "14px"}
                                     ),
                                 ]),
                             ]),

                             # Swap axes
                             html.Div(style={"paddingTop": "22px"}, children=[
                                 html.Button("⇄", id="swap-axes", style={
                                     "fontSize": "15px", "padding": "6px 10px", "cursor": "pointer",
                                     "border": "1px solid #ccc", "background": "#f5f5f5", "borderRadius": "4px"
                                 })
                             ]),

                             # Y axis: dropdown + buttons wrapped together with tight gap
                             html.Div(style={"display": "flex", "gap": "2px", "alignItems": "start"}, children=[
                                 # Y axis
                                 html.Div(style={"width": "200px"}, children=[
                                     html.Label("Y axis", style={"fontSize": "18px"}),
                                     dcc.Dropdown(
                                         id="ycategory",
                                         options=[{"label": cat, "value": cat} for cat in PARAMETER_CATEGORIES.keys()],
                                         value="Basic",
                                         clearable=False,
                                         placeholder="Select category...",
                                         style={"fontSize": "14px", "marginBottom": "4px"}
                                     ),
                                     dcc.Dropdown(
                                         id="ycol",
                                         value="L",
                                         clearable=False,
                                         style={"fontSize": "16px"}
                                     ),
                                     dcc.RadioItems(
                                         id="yscale",
                                         options=[
                                             {"label": "Lin", "value": "lin"},
                                             {"label": "Log", "value": "log"},
                                             {"label": "Log(X)", "value": "log(x)"}],
                                         value="lin",
                                         inline=True,
                                         style={"marginTop": "6px", "fontSize": "14px"}
                                     ),
                                 ]),
                                 # Y axis format selectors
                                 html.Div(style={"paddingTop": "22px", "width": "80px", "flexShrink": "0"}, children=[
                                     html.Div(id="y-kipp-coord-container", style={"display": "none"}, children=[
                                         dcc.RadioItems(
                                             id="y-kipp-coord",
                                             options=[
                                                 {"label": "M", "value": "M"},
                                                 {"label": "R", "value": "R"},
                                             ],
                                             value="M",
                                             style={"fontSize": "14px"}
                                         )
                                     ]),
                                     html.Div(id="y-age-units-container", style={"display": "none"}, children=[
                                         dcc.RadioItems(
                                             id="y-age-units",
                                             options=[
                                                 {"label": "Gyr", "value": "gyr"},
                                                 {"label": "Myr", "value": "myr"},
                                                 {"label": "yr", "value": "yr"},
                                             ],
                                             value="gyr",
                                             style={"fontSize": "14px"}
                                         )
                                     ]),
                                     html.Div(id="y-abund-format-container", style={"display": "none"}, children=[
                                         dcc.RadioItems(
                                             id="y-abund-format",
                                             options=[
                                                 {"label": "X(X)", "value": "XX"},
                                                 {"label": "A(X)", "value": "AX"},
                                                 {"label": "[X/H]", "value": "XH"},
                                                 {"label": "[X/M]", "value": "XM"},
                                             ],
                                             value="XX",
                                             style={"fontSize": "14px"},
                                         )
                                     ]),
                                 ]),
                             ]),

                             # Freeze controls (replaced subsample slider)
                             html.Div(style={"width": "280px", "marginLeft": "5px", "paddingTop": "22px"}, children=[
                                 html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
                                     html.Button(
                                         "📌 Freeze plot",
                                         id="btn-freeze",
                                         n_clicks=0,
                                         title="Pin the current plot as a static background layer",
                                         style={
                                             "height": "36px", "padding": "0 12px", "borderRadius": "4px",
                                             "border": "1px solid #aaa", "cursor": "pointer",
                                             "fontSize": "14px", "background": "#eef4ff",
                                         },
                                     ),
                                     html.Button(
                                         "🗑 Clear frozen",
                                         id="btn-clear-frozen",
                                         n_clicks=0,
                                         title="Remove all frozen background layers",
                                         style={
                                             "height": "36px", "padding": "0 12px", "borderRadius": "4px",
                                             "border": "1px solid #ddd", "cursor": "pointer",
                                             "fontSize": "14px", "background": "#fff0f0",
                                         },
                                     ),
                                 ]),
                                 html.Span(id="frozen-status",
                                           style={"fontSize": "13px", "color": "#666", "marginTop": "2px",
                                                  "display": "block", "minHeight": "18px"}),

                                 # Show phase + phase range
                                 html.Div(
                                     style={"marginTop": "12px", "display": "flex", "alignItems": "center",
                                            "gap": "5px"},
                                     children=[
                                         dcc.Checklist(
                                             id="show-phase",
                                             options=[{"label": "Show phase", "value": "on"}],
                                             value=[],
                                             style={
                                                 "fontSize": "16px",
                                                 "display": "flex",
                                                 "alignItems": "center",
                                                 "lineHeight": "14px",
                                                 "minWidth": "100px",
                                                 "margin": "0",
                                                 "padding": "0",
                                             },
                                             labelStyle={
                                                 "display": "flex",
                                                 "alignItems": "center",
                                                 "margin": "0",
                                                 "lineHeight": "14px",
                                             },
                                             inputStyle={
                                                 "marginRight": "2px",
                                                 "marginTop": "0px",
                                             },
                                         ),

                                         html.Div(
                                             style={"display": "flex", "alignItems": "center", "gap": "3px",
                                                    "marginLeft": "5px"},
                                             children=[
                                                 html.Span("min", style={"fontSize": "16px"}),
                                                 dcc.Input(
                                                     id="phase-min",
                                                     type="number",
                                                     min=1, max=5, step=1,
                                                     value=1,
                                                     style={"width": "24px", "fontSize": "14px", "padding": "3px 2px"},
                                                 ),
                                             ]),

                                         html.Div(style={"display": "flex", "alignItems": "center", "gap": "3px"},
                                                  children=[
                                                      html.Span("max", style={"fontSize": "16px"}),
                                                      dcc.Input(
                                                          id="phase-max",
                                                          type="number",
                                                          min=1, max=5, step=1,
                                                          value=5,
                                                          style={"width": "24px", "fontSize": "14px",
                                                                 "padding": "3px 2px"},
                                                      ),
                                                  ]),
                                     ],
                                 ),
                             ]),
                         ]),

            ]),

            html.Hr(),

            # Plot window
            dcc.Interval(id="poll-full-load", interval=1000, n_intervals=0, disabled=True),
            dcc.Store(id="frozen-images", data=[]),
            dcc.Store(id="frozen-composite", data=None),
            dcc.Store(id="reload-signal", data=None),
            dcc.Input(id="reload-idx-input", type="text", value="", debounce=False, style={"display": "none"}),
            dcc.Graph(
                id="plot",
                style={"height": "75vh", "paddingTop": "5px"},
                mathjax=True,
                config={"doubleClick": False},
            ),
            html.Div(id="export-filename-dummy", style={"display": "none"}),
        ])

    _DROPDOWN_HIDDEN = {
        "display": "none",
        "position": "absolute", "top": "100%", "left": "0", "width": "100%",
        "zIndex": "1000", "background": "white", "border": "1px solid #ddd",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.12)",
        "maxHeight": "240px", "overflowY": "auto", "padding": "6px",
    }
    _DROPDOWN_VISIBLE = {**_DROPDOWN_HIDDEN, "display": "block"}

    # --- Freeze: capture the current plot as PNG and store it; clear resets the store ---

    # Freeze button: clientside callback returns a Promise — Dash 2.x awaits it automatically
    app.clientside_callback(
        """
        function(n_clicks, ycol, xAgeUnits) {
            var NO_UPDATE = window.dash_clientside.no_update;
            if (!n_clicks) return [NO_UPDATE, NO_UPDATE];

            var graphDiv = document.getElementById('plot');
            if (!graphDiv) return [NO_UPDATE, NO_UPDATE];

            // Get the main SVG element that Plotly renders into
            var mainSvg = graphDiv.querySelector('.main-svg');
            if (!mainSvg) { console.log('[FREEZE] no .main-svg found'); return [NO_UPDATE, NO_UPDATE]; }

            var fl = graphDiv._fullLayout;
            var ml = (fl && fl._size) ? fl._size.l : 90;
            var mr = (fl && fl._size) ? fl._size.r : 620;
            var mt = (fl && fl._size) ? fl._size.t : 2;
            var mb = (fl && fl._size) ? fl._size.b : 100;
            var w  = fl ? fl.width  : mainSvg.clientWidth;
            var h  = fl ? fl.height : mainSvg.clientHeight;

            var cropW = w - ml - mr;
            var cropH = h - mt - mb;
            console.log('[FREEZE] size:', w, h, '| margins:', ml, mr, mt, mb, '| crop:', cropW, cropH);

            // Serialize the SVG to a string, inline all styles
            var svgClone = mainSvg.cloneNode(true);
            svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            // Remove previously frozen overlays (layout.images) so they don't get baked into this snapshot
            svgClone.querySelectorAll('image').forEach(function(el) { el.remove(); });
            // Add white background so SVG renders correctly
            var bgRect = document.createElementNS('http://www.w3.org/2000/svg','rect');
            bgRect.setAttribute('width', w); bgRect.setAttribute('height', h);
            bgRect.setAttribute('fill', 'white');
            svgClone.insertBefore(bgRect, svgClone.firstChild);

            var svgStr = new XMLSerializer().serializeToString(svgClone);
            var svgBlob = new Blob([svgStr], {type: 'image/svg+xml;charset=utf-8'});
            var url = URL.createObjectURL(svgBlob);

            return new Promise(function(resolve, reject) {
                var img = new Image();
                img.onload = function() {
                    console.log('[FREEZE] svg->img loaded:', img.naturalWidth, img.naturalHeight);
                    var canvas = document.createElement('canvas');
                    canvas.width  = cropW;
                    canvas.height = cropH;
                    var ctx = canvas.getContext('2d');
                    ctx.drawImage(img, ml, mt, cropW, cropH, 0, 0, cropW, cropH);
                    URL.revokeObjectURL(url);
                    var cropped = canvas.toDataURL('image/png');
                    console.log('[FREEZE] cropped length=', cropped.length);
                    // Use window-level accumulator to avoid Dash State staleness in async callbacks
                    if (!window._frozenImgs) window._frozenImgs = [];
                    var isKipp = !!(ycol && ycol.startsWith('__kipp'));
                    window._frozenImgs.push({src: cropped, w: cropW, h: cropH, is_kipp: isKipp, x_age_units: xAgeUnits || 'gyr'});
                    // If freezing kipp: force same age units on the overlay
                    resolve([window._frozenImgs.slice(), isKipp ? xAgeUnits : NO_UPDATE]);
                };
                img.onerror = function(e) {
                    console.log('[FREEZE] error loading svg:', e);
                    URL.revokeObjectURL(url);
                    reject(e);
                };
                img.src = url;
            });
        }
        """,
        Output("frozen-images", "data", allow_duplicate=True),
        Output("x-age-units", "value", allow_duplicate=True),
        Input("btn-freeze", "n_clicks"),
        State("ycol", "value"),
        State("x-age-units", "value"),
        prevent_initial_call=True,
    )

    # Clear button
    @app.callback(
        Output("frozen-images", "data"),
        Output("frozen-status", "children"),
        Input("btn-clear-frozen", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_frozen(n_clicks):
        return [], "Frozen layers cleared."

    # Composite all frozen PNGs into one image (JS-side, no Python image libs needed).
    # Each older frozen layer is dimmed by one extra power of freeze_opacity: the most
    # recently frozen layer gets freeze_opacity^1, the one before it freeze_opacity^2,
    # etc., so that re-freezing compounds the fade exactly as the user expects.
    _freeze_opacity_js = repr(max(0.0, min(1.0, float(cfg.freeze_opacity))))
    app.clientside_callback(
        """
        function(frozen_images) {
            if (!frozen_images || frozen_images.length === 0) return null;
            var n = frozen_images.length;
            var W = frozen_images[0].w, H = frozen_images[0].h;
            var canvas = document.createElement('canvas');
            canvas.width = W; canvas.height = H;
            var ctx = canvas.getContext('2d');
            var freezeOpacity = """ + _freeze_opacity_js + """;
            return new Promise(function(resolve) {
                var pending = n;
                var imgEls = new Array(n);
                frozen_images.forEach(function(entry, i) {
                    var img = new Image();
                    img.onload = function() {
                        imgEls[i] = img;
                        if (--pending === 0) {
                            imgEls.forEach(function(imgEl, idx) {
                                // Draw to temp canvas to strip white background
                                var tmp = document.createElement('canvas');
                                tmp.width = W; tmp.height = H;
                                var tc = tmp.getContext('2d');
                                tc.drawImage(imgEl, 0, 0, W, H);
                                var id = tc.getImageData(0, 0, W, H), d = id.data;
                                for (var j = 0; j < d.length; j += 4) {
                                    if (d[j] > 240 && d[j+1] > 240 && d[j+2] > 240) d[j+3] = 0;
                                }
                                tc.putImageData(id, 0, 0);
                                // idx counts from oldest (0) to newest (n-1) frozen layer;
                                // the newest frozen layer is one step behind the live plot,
                                // so it gets freezeOpacity^1, the previous one ^2, and so on.
                                ctx.globalAlpha = Math.pow(freezeOpacity, n - idx);
                                ctx.drawImage(tmp, 0, 0);
                            });
                            ctx.globalAlpha = 1.0;
                            resolve(canvas.toDataURL('image/png'));
                        }
                    };
                    img.onerror = function() { if (--pending === 0) resolve(canvas.toDataURL('image/png')); };
                    img.src = entry.src;
                });
            });
        }
        """,
        Output("frozen-composite", "data"),
        Input("frozen-images", "data"),
        prevent_initial_call=True,
    )

    # Status label update whenever store changes
    app.clientside_callback(
        """
        function(imgs) {
            if (!imgs || imgs.length === 0) {
                window._frozenImgs = [];
                return '';
            }
            return imgs.length === 1 ? '1 layer frozen.' : imgs.length + ' layers frozen.';
        }
        """,
        Output("frozen-status", "children", allow_duplicate=True),
        Input("frozen-images", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(value) {
            // Patch existing tooltip text immediately
            function patchTooltip() {
                var el = document.querySelector('#age-slider .rc-slider-tooltip-inner');
                if (el) {
                    var myr = Math.round(value / 1e6);
                    el.textContent = myr + ' Myr';
                }
            }
            patchTooltip();
            setTimeout(patchTooltip, 30);
            setTimeout(patchTooltip, 100);

            // Install MutationObserver once to patch tooltip on every show/change
            if (!window._ageSliderObserver) {
                var slider = document.getElementById('age-slider');
                if (slider) {
                    window._ageSliderObserver = new MutationObserver(function() {
                        var el = document.querySelector('#age-slider .rc-slider-tooltip-inner');
                        if (el && el.textContent && !el.textContent.includes('Myr')) {
                            var raw = parseFloat(el.textContent.replace(/[^0-9.]/g, ''));
                            if (!isNaN(raw)) {
                                el.textContent = Math.round(raw / 1e6) + ' Myr';
                            }
                        }
                    });
                    window._ageSliderObserver.observe(slider, {subtree: true, childList: true, characterData: true});
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("age-slider", "className"),
        Input("age-slider", "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("search-dropdown", "children"),
        Output("search-dropdown", "style"),
        Input("search", "value"),
        State("picked-models", "value"),
    )
    def do_search(q, currently_picked):
        if not q:
            return [], _DROPDOWN_HIDDEN
        res = registry.search(q)
        if res.empty:
            return (
                [html.Div("No matches", style={"color": "#b00", "fontSize": "16px", "padding": "4px"})],
                _DROPDOWN_VISIBLE,
            )
        return _render_search_results(res, currently_picked or []), _DROPDOWN_VISIBLE

    # Update X parameter options when category changes
    @app.callback(
        Output("xcol", "options"),
        Output("xcol", "value", allow_duplicate=True),
        Input("xcategory", "value"),
        State("xcol", "value"),
        prevent_initial_call='initial_duplicate'
    )
    def update_x_params(category, current_value):
        params = PARAMETER_CATEGORIES.get(category, [])
        # Exclude kipp diagram params from X axis
        params_x = [p for p in params if not str(p).startswith("__kipp")]
        options = [{"label": display_label_dropdown(p), "value": p} for p in params_x]
        # If current value exists in new category, keep it; otherwise use first param
        value = current_value if current_value in params_x else (params_x[0] if params_x else None)
        return options, value

    # Update Y parameter options when category changes
    @app.callback(
        Output("ycol", "options"),
        Output("ycol", "value", allow_duplicate=True),
        Input("ycategory", "value"),
        State("ycol", "value"),
        prevent_initial_call='initial_duplicate'
    )
    def update_y_params(category, current_value):
        params = PARAMETER_CATEGORIES.get(category, [])
        options = [{"label": display_label_dropdown(p), "value": p} for p in params]
        # If current value exists in new category, keep it; otherwise use first param
        value = current_value if current_value in params else (params[0] if params else None)
        return options, value

    def _force_log_x_for_abund_fmt(fmt: str):
        """Return (options, value) forcing log(x) when abundance fmt is AX/XH/XM."""
        opts = [{**o, "disabled": o["value"] != "log(x)"} for o in AXIS_SCALE_CHOICES]
        return opts, "log(x)"

    # Sync X scale radio based on selected X and Y parameters
    @app.callback(
        Output("xscale", "options"),
        Output("xscale", "value"),
        Input("xcol", "value"),
        Input("ycol", "value"),
        Input("x-abund-format", "value"),
        State("frozen-images", "data"),
    )
    def sync_xscale(xcol, ycol, x_abund_fmt, frozen_images):
        if str(ycol or "").startswith("__kipp"):
            opts = [{**o, "disabled": o["value"] != "lin"} for o in AXIS_SCALE_CHOICES]
            return opts, "lin"
        # Abundance formats A(X), [X/H], [X/M] are logarithmic by definition — force log(x)
        if _is_abund_col(str(xcol or "")) and x_abund_fmt in ("AX", "XH", "XM"):
            return _force_log_x_for_abund_fmt(x_abund_fmt)
        # If overlaying Age on X over a frozen Kippenhahn, force linear
        frozen_has_kipp = bool(
            frozen_images and any(isinstance(e, dict) and e.get("is_kipp") for e in frozen_images)
        )
        if frozen_has_kipp and str(xcol or "").lower() == "age":
            return get_axis_scale_options(xcol), "lin"
        return get_axis_scale_options(xcol), get_default_axis_scale(xcol)

    # Sync Y scale radio based on selected Y and X parameters
    @app.callback(
        Output("yscale", "options"),
        Output("yscale", "value"),
        Input("ycol", "value"),
        Input("xcol", "value"),
        Input("y-abund-format", "value"),
        Input("y-kipp-coord", "value"),
    )
    def sync_yscale(ycol, xcol, y_abund_fmt, kipp_coord):
        if str(ycol or "") == "__kipp_env":
            if str(kipp_coord or "M") == "R":
                # Radius-coordinate Kippenhahn: Lin/Log are both meaningful, Log(X) is not
                opts = [{**o, "disabled": o["value"] == "log(x)"} for o in AXIS_SCALE_CHOICES]
                return opts, "lin"
            # Mass-coordinate Kippenhahn — lin only
            opts = [{**o, "disabled": o["value"] != "lin"} for o in AXIS_SCALE_CHOICES]
            return opts, "lin"
        if str(xcol or "").startswith("__kipp"):
            opts = [{**o, "disabled": o["value"] != "lin"} for o in AXIS_SCALE_CHOICES]
            return opts, "lin"
        # Abundance formats A(X), [X/H], [X/M] are logarithmic by definition — force log(x)
        if _is_abund_col(str(ycol or "")) and y_abund_fmt in ("AX", "XH", "XM"):
            return _force_log_x_for_abund_fmt(y_abund_fmt)
        return get_axis_scale_options(ycol), get_default_axis_scale(ycol)

    @app.callback(
        Output("xcol", "value", allow_duplicate=True),
        Output("ycol", "value", allow_duplicate=True),
        Output("xcategory", "value", allow_duplicate=True),
        Output("ycategory", "value", allow_duplicate=True),
        Input("swap-axes", "n_clicks"),
        State("xcol", "value"),
        State("ycol", "value"),
        State("xcategory", "value"),
        State("ycategory", "value"),
        prevent_initial_call=True
    )
    def swap_axes(n_clicks, x, y, xcat, ycat):
        return y, x, ycat, xcat

    @app.callback(
        Output("swap-axes", "disabled"),
        Output("swap-axes", "style"),
        Input("ycol", "value"),
    )
    def toggle_swap_button(ycol):
        is_kipp = str(ycol or "").startswith("__kipp")
        if is_kipp:
            style = {
                "fontSize": "15px", "padding": "6px 10px", "cursor": "not-allowed",
                "border": "1px solid #ccc", "background": "#e0e0e0", "borderRadius": "4px",
                "opacity": "0.4"
            }
            return True, style
        else:
            style = {
                "fontSize": "15px", "padding": "6px 10px", "cursor": "pointer",
                "border": "1px solid #ccc", "background": "#f5f5f5", "borderRadius": "4px",
                "opacity": "1"
            }
            return False, style

    @app.callback(
        Output("xcol", "value", allow_duplicate=True),
        Output("xcategory", "value", allow_duplicate=True),
        Input("ycol", "value"),
        prevent_initial_call=True,
    )
    def auto_set_x_for_kipp(ycol):
        if str(ycol or "").startswith("__kipp"):
            return "Age", "Basic"
        return dash.no_update, dash.no_update

    @app.callback(
        Output("picked-models", "value"),
        Input("btn-clear-models", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_models(_):
        return []

    @app.callback(
        Output("picked-models", "value", allow_duplicate=True),
        Input({"type": "search-group-cb", "index": ALL}, "value"),
        State("picked-models", "value"),
        State({"type": "search-group-cb", "index": ALL}, "options"),
        prevent_initial_call=True,
    )
    def merge_picks(all_cb_values, current, all_cb_opts):
        if not callback_context.triggered:
            return current or []
        all_visible: set[str] = set()
        for opts in (all_cb_opts or []):
            for o in (opts or []):
                all_visible.add(o["value"])
        newly_selected: set[str] = set()
        for vals in (all_cb_values or []):
            newly_selected.update(vals or [])
        current_set = set(current or [])
        current_set -= (all_visible - newly_selected)
        current_set |= newly_selected
        return sorted(current_set)

    @app.callback(
        Output({"type": "group-body", "index": MATCH}, "style"),
        Output({"type": "group-toggle-btn", "index": MATCH}, "children"),
        Input({"type": "group-toggle-btn", "index": MATCH}, "n_clicks"),
        State({"type": "group-body", "index": MATCH}, "style"),
        prevent_initial_call=True,
    )
    def toggle_group(_, current_style):
        is_hidden = (current_style or {}).get("display") == "none"
        if is_hidden:
            return {"display": "block", "paddingLeft": "14px"}, "▼"
        return {"display": "none"}, "▶"

    @app.callback(
        Output("picked-models", "value", allow_duplicate=True),
        Input({"type": "load-all-btn", "index": ALL}, "n_clicks"),
        State("picked-models", "value"),
        prevent_initial_call=True,
    )
    def load_all_from_group(n_clicks_list, current):
        triggered = [t for t in callback_context.triggered
                     if t.get("value") and t["value"] > 0]
        if not triggered:
            return current or []
        import json as _json
        folder_path = _json.loads(triggered[0]["prop_id"].split(".")[0])["index"]
        folder_df = registry._index[registry._index["folder_path"] == folder_path]
        current_set = set(current or [])
        current_set |= set(folder_df["path"].tolist())
        return sorted(current_set)

    @app.callback(
        Output("age-slider", "value"),
        Input("age-input", "n_submit"),
        Input("age-input", "n_blur"),
        State("age-input", "value"),
        prevent_initial_call=True
    )
    def sync_age_input_to_slider(n_submit, n_blur, input_val):
        # Input changed (in Myr) -> convert to years for slider
        slider_value = input_val * 1e6 if input_val is not None else 0
        return slider_value

    # Update export filename dynamically via plot config (reactive)
    @app.callback(
        Output("plot", "config"),
        Input("xcol", "value"),
        Input("ycol", "value"),
        Input("picked-models", "value"),
    )
    def update_export_filename(xcol, ycol, models):
        x = xcol or "x"
        y = ycol or "y"
        n = len(models) if models else 0

        def _col_to_name(col):
            """Convert column key to a clean filename-safe label."""
            if col and ":" in col:
                # e.g. "surf:Li7" -> "Li7surf", "cent:He4" -> "He4cent"
                loc, iso = col.split(":", 1)
                return f"{iso}{loc}"
            return col

        # Special case: HR diagram
        if x == "Teff" and y == "L":
            filename = f"HRD_{n}_models"
        else:
            filename = f"{_col_to_name(y)}_vs_{_col_to_name(x)}_{n}_models"
        return {
            "doubleClick": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
            },
        }

    @app.callback(
        Output("isochrone-checkbox-container", "style"),
        Input("picked-models", "value"),
    )
    def toggle_isochrone_checkbox(models):
        """Show isochrone checkbox only when 2 or more models are selected."""
        if models and len(models) >= 2:
            return {"display": "block"}
        else:
            return {"display": "none"}

    # Callback to show/hide X age units selector
    @app.callback(
        Output("x-age-units-container", "style"),
        Input("xcol", "value"),
        Input("xscale", "value"),
        Input("ycol", "value"),
    )
    def toggle_x_age_units(xcol, xscale, ycol):
        is_kipp = str(ycol or "").startswith("__kipp")
        is_age = str(xcol or "").lower() == "age"
        is_lin = xscale == "lin"

        if (is_age and is_lin) or is_kipp:
            return {"display": "block"}
        else:
            return {"display": "none"}

    # Callback to show/hide Y age units selector
    @app.callback(
        Output("y-age-units-container", "style"),
        Input("ycol", "value"),
        Input("yscale", "value"),
    )
    def toggle_y_age_units(ycol, yscale):
        is_age = str(ycol or "").lower() == "age"
        is_lin = yscale == "lin"

        if is_age and is_lin:
            return {"display": "block"}
        else:
            return {"display": "none"}

    # Callback to show/hide the Kippenhahn M/R coordinate-type selector
    # (only the Y axis can host a Kippenhahn diagram — see auto_set_x_for_kipp).
    @app.callback(
        Output("y-kipp-coord-container", "style"),
        Input("ycol", "value"),
    )
    def toggle_y_kipp_coord(ycol):
        if str(ycol or "") == "__kipp_env":
            return {"display": "block"}
        return {"display": "none"}

    # --- Abundance format callbacks ---

    def _is_abund_col(col):
        """True if col is a virtual abundance key (surf:X or cent:X)."""
        return bool(col) and ":" in col and col.split(":")[0] in ("surf", "cent")

    @app.callback(
        Output("x-abund-format-container", "style"),
        Input("xcol", "value"),
    )
    def toggle_x_abund_format(xcol):
        if _is_abund_col(xcol):
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("y-abund-format-container", "style"),
        Input("ycol", "value"),
    )
    def toggle_y_abund_format(ycol):
        if _is_abund_col(ycol):
            return {"display": "block"}
        return {"display": "none"}

    def _abund_format_options_and_value(col, current_fmt):
        """Return (options, value) for an abundance format RadioItems given a column key."""
        _, isotope = _parse_abundance_key(col or "")
        is_hydrogen = isotope in HYDROGEN_ISOTOPES
        options = [
            {"label": "X(X)", "value": "XX"},
            {"label": "A(X)", "value": "AX", "disabled": is_hydrogen},
            {"label": "[X/H]", "value": "XH", "disabled": is_hydrogen},
            {"label": "[X/M]", "value": "XM", "disabled": is_hydrogen},
        ]
        value = current_fmt if (current_fmt and not (is_hydrogen and current_fmt != "XX")) else "XX"
        return options, value

    @app.callback(
        Output("x-abund-format", "options"),
        Output("x-abund-format", "value"),
        Input("xcol", "value"),
        State("x-abund-format", "value"),
    )
    def update_x_abund_format_options(xcol, current_fmt):
        return _abund_format_options_and_value(xcol, current_fmt)

    @app.callback(
        Output("y-abund-format", "options"),
        Output("y-abund-format", "value"),
        Input("ycol", "value"),
        State("y-abund-format", "value"),
    )
    def update_y_abund_format_options(ycol, current_fmt):
        return _abund_format_options_and_value(ycol, current_fmt)

    def _make_log_ticks(r0: float, r1: float, multipliers) -> tuple:
        """Generate (tickvals, ticktext) for a log-scale axis given log10 range [r0, r1].

        For narrow ranges (< 1 decade) uses linear-spaced ticks.
        For wide ranges uses 1/2/5 × 10^N ticks per decade.
        Returns (tickvals, ticktext) — both lists.
        """
        log_min, log_max = min(r0, r1), max(r0, r1)
        range_magnitude = abs(r1 - r0)

        if range_magnitude < 1.0:
            val_min = 10 ** log_min
            val_max = 10 ** log_max
            span = val_max - val_min
            base_order = int(np.floor(np.log10(val_min)))
            base = 10 ** base_order

            if span <= base * 2:
                step = base * 0.5
            elif span <= base * 5:
                step = base
            else:
                step = base * 2

            tickvals = []
            current = np.ceil(val_min / step) * step
            while current <= val_max:
                if current >= val_min:
                    tickvals.append(float(current))
                current += step

            if len(tickvals) < 3:
                step = step / 2
                tickvals = []
                current = np.ceil(val_min / step) * step
                while current <= val_max:
                    if current >= val_min:
                        tickvals.append(float(current))
                    current += step

            ticktext = [f"{int(v)}" if v >= 1 and v == int(v) else f"{v:.4g}" for v in tickvals]
            return tickvals, ticktext
        else:
            tickvals = []
            ticktext = []
            start_order = int(np.floor(log_min))
            end_order = int(np.ceil(log_max))
            for order in range(start_order, end_order + 1):
                base = 10 ** order
                for mult in multipliers:
                    val = mult * base
                    log_val = np.log10(val)
                    if log_min <= log_val <= log_max:
                        tickvals.append(val)
                        if mult == 1:
                            ticktext.append(f"10<sup>{order}</sup>")
                        else:
                            _m = int(mult) if mult == int(mult) else f"{mult:.4g}"
                            ticktext.append(f"<span style='font-size: 0.75em'>{_m}</span>")
            return tickvals, ticktext

    @app.callback(
        Output("reload-signal", "data"),
        Input("reload-idx-input", "value"),
        prevent_initial_call=True,
    )
    def reload_model_cache(value):
        import time
        if not value:
            return dash.no_update
        try:
            path = value.rsplit(":", 1)[0]
        except (ValueError, AttributeError):
            return dash.no_update
        if not path:
            return dash.no_update
        HR_CACHE.pop(path, None)
        FULL_CACHE.pop(path, None)
        FULL_FUTURES.pop(path, None)
        KIPP_CACHE.pop(path, None)
        KIPP_FUTURES.pop(path, None)
        SURF_CACHE.pop(path, None)
        SURF_FUTURES.pop(path, None)
        AGE_MARKER_CACHE.clear()
        return time.time()

    @app.callback(
        Output("plot", "figure"),
        Output("load-status", "children"),
        Output("poll-full-load", "disabled"),
        Output("age-slider", "max"),
        Output("age-slider", "marks"),
        Output("age-input", "value"),
        Input("picked-models", "value"),
        Input("xcol", "value"),
        Input("ycol", "value"),
        Input("xscale", "value"),
        Input("yscale", "value"),
        Input("y-kipp-coord", "value"),
        Input("x-age-units", "value"),
        Input("y-age-units", "value"),
        Input("show-phase", "value"),
        Input("phase-min", "value"),
        Input("phase-max", "value"),
        Input("age-slider", "value"),
        Input("show-isochrone", "value"),
        Input("poll-full-load", "n_intervals"),
        Input("plot", "relayoutData"),
        Input("fast-hr-mode", "value"),
        Input("x-abund-format", "value"),
        Input("y-abund-format", "value"),
        Input("reload-signal", "data"),
        State("frozen-images", "data"),
        Input("frozen-composite", "data"),
    )
    def render(models, xcol, ycol, xscale, yscale, kipp_coord, x_age_units, y_age_units, show_phase_value, phase_min_in,
               phase_max_in,
               selected_age, show_isochrone_value, _poll, relayout, fast_hr_mode, x_abund_fmt, y_abund_fmt,
               reload_signal, frozen_images, frozen_composite):

        # Set default values if None (happens on initial load)
        if xcol is None:
            xcol = "Teff"
        if ycol is None:
            ycol = "L"

        # Discard stale zoom/pan state (relayoutData reflects the LAST relayout
        # event on the plot, regardless of what triggered this render) whenever
        # the user changes WHAT is being plotted rather than interacting with
        # the plot itself -- otherwise e.g. switching Kippenhahn M<->R or
        # toggling its Y-scale would reapply the previous diagram's zoomed
        # range to the new one. uirevision below mirrors this so the client
        # doesn't try to restore the old view either.
        _trigger_id = callback_context.triggered_id if callback_context.triggered else None
        if _trigger_id in ("xcol", "ycol", "xscale", "yscale", "y-kipp-coord", "picked-models"):
            relayout = None

        def _get_zoom_range(relayout, axis: str):
            if not relayout or not isinstance(relayout, dict):
                return None
            r0 = relayout.get(f"{axis}.range[0]")
            r1 = relayout.get(f"{axis}.range[1]")
            if relayout.get(f"{axis}.autorange") in (True, "reversed"):
                return None
            if r0 is not None and r1 is not None:
                return [float(r0), float(r1)]
            return None

        step = 1
        fig = go.Figure()

        show_phase = bool(show_phase_value)  # checklist: [] or ["on"]

        def _sanitize_phase_range(pmin, pmax):
            # Defaults
            lo, hi = 1, 5

            # Coerce to int if possible
            try:
                if pmin is not None and pmin != "":
                    lo = int(pmin)
            except Exception:
                lo = 1

            try:
                if pmax is not None and pmax != "":
                    hi = int(pmax)
            except Exception:
                hi = 5

            # Clamp to [1..5]
            lo = max(1, min(5, lo))
            hi = max(1, min(5, hi))

            # If inverted, swap (default action)
            if lo > hi:
                lo, hi = hi, lo

            return lo, hi

        phase_min, phase_max = _sanitize_phase_range(phase_min_in, phase_max_in)

        dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

        # phase colors
        PHASE_COLORS = cfg.phase_colors

        base_line_width = 2.0

        xcol = xcol or "Teff"
        ycol = ycol or "L"
        kipp_env_mode = (str(ycol) == "__kipp_env")
        # The dropdown offers a single "Kippenhahn diagram" entry; the M/R radio
        # button next to it (default "M") picks the mass- or radius-coordinate
        # variant -- see y-kipp-coord-container in the layout and sync_yscale.
        kipp_radius_mode = (kipp_env_mode and str(kipp_coord or "M") == "R")

        # Effective scale modes (forced modes override user selection)
        xscale_eff = get_effective_axis_scale(xcol, xscale)
        yscale_eff = get_effective_axis_scale(ycol, yscale)
        if kipp_env_mode and not kipp_radius_mode:
            # Mass-coordinate Kippenhahn is lin-only (sync_yscale already locks
            # the radio to "lin", but force it here too in case of stale state).
            yscale_eff = "lin"

        def _wrap_title_with_log10(title: str) -> str:
            """Wrap an axis title into log10(...), preserving LaTeX $...$ when present."""
            if not title:
                return title
            t = str(title)
            # If the title is already LaTeX wrapped, keep it LaTeX.
            if t.startswith("$") and t.endswith("$") and len(t) >= 2:
                inner = t[1:-1]
                return f"$\\log~{inner}$"
            return f"log10({t})"

        def _axis_title(col: str, is_age: bool, scale_mode: str, age_units: str = None,
                        abund_fmt: str = None) -> str:
            # For Age, get base label and replace units if needed
            if is_age:
                base = display_label_plot("Age")  # Get the formatted label (e.g., LaTeX)

                # For Age in linear scale, replace units in the title
                if scale_mode == "lin" and age_units:
                    unit_map = {"gyr": "Gyr", "myr": "Myr", "yr": "yr"}
                    unit_label = unit_map.get(age_units, "Gyr")

                    # If base is LaTeX format ($...$), replace units within LaTeX
                    if isinstance(base, str) and base.startswith("$") and base.endswith("$"):
                        # Remove existing units (yr, Myr, Gyr) from LaTeX
                        import re
                        # Match patterns like "\mathrm{yr}", "\mathrm{Myr}", "\mathrm{Gyr}", or just "yr", "Myr", "Gyr"
                        base = re.sub(r',?\s*\\mathrm\{(yr|Myr|Gyr)}', '', base)
                        base = re.sub(r',?\s*(yr|Myr|Gyr)', '', base)
                        # Add new units before closing $
                        base = base.rstrip('$') + r',\ \mathrm{' + unit_label + '}$'
                    else:
                        # Plain text: remove old units and add new
                        import re
                        base = re.sub(r',?\s*(yr|Myr|Gyr)$', '', base)
                        base = f"{base}, {unit_label}"
            elif _is_abund_col(col) and abund_fmt:
                # Virtual abundance key: build title from isotope + format
                base = _abundance_axis_title(col, abund_fmt)
            else:
                base = display_label_plot(col)

            if scale_mode == "log(x)":
                # If parameter starts with "log" (e.g. logg, logL, logTeff),
                # data are already logarithmized — return label as-is, no log10(...) wrapper
                if col and str(col).startswith("log") and len(str(col)) > 3:
                    return base

                # A(X) / Ac(X) — spectroscopic abundances, already logarithmic.
                _col = str(col) if col else ""
                if _col.startswith("A_") or _col.startswith("Ac_"):
                    return base

                # Virtual abundance in dex format — already log, no wrapping
                if _is_abund_col(_col) and abund_fmt in ("AX", "XH", "XM"):
                    return base

                # [M/H], [α/H], [α/M] — also already logarithms, do not add "log"
                log_abund_ratios = ["MH", "alphaH_ONeMgSiS", "alphaH_MgSi", "alphaM_ONeMgSiS", "alphaM_MgSi"]
                if _col in log_abund_ratios:
                    return base

                # Regular parameter — wrap with log10(...)
                return _wrap_title_with_log10(base)

            return base

        def _apply_axis_scale(s: pd.Series, mode: str, col_name: str = None,
                              abund_fmt: str = None) -> pd.Series:
            """
            Applies axis data transformation depending on the scale mode.

            Args:
                s: Series with data
                mode: scale mode ("lin", "log", "log(x)")
                col_name: column name (used to determine whether logarithmization is needed)
                abund_fmt: abundance format ("XX", "AX", "XH", "XM") for virtual keys

            Important: if col_name starts with "log" (e.g. logg, logL, logTeff),
            the data are ALREADY logarithmized in files, and for mode "log(x)" we must NOT
            take their logarithm again!
            """
            s = pd.to_numeric(s, errors="coerce")

            if mode in ("log", "log(x)"):
                _col = str(col_name) if col_name else ""
                # Abundance ratios that are already logarithms
                log_abund_ratios = ["MH", "alphaH_ONeMgSiS", "alphaH_MgSi", "alphaM_ONeMgSiS", "alphaM_MgSi"]
                # Virtual abundance keys in dex-style formats are already logarithmic
                _is_virt_abund_log = _is_abund_col(_col) and (abund_fmt in ("AX", "XH", "XM"))
                already_log = (
                        (_col.startswith("log") and len(_col) > 3)
                        or _col.startswith("A_")
                        or _col.startswith("Ac_")
                        or _col in log_abund_ratios
                        or _is_virt_abund_log
                )

                if mode == "log(x)":
                    if already_log:
                        # Data are ALREADY logarithmized (logg, logL, A(X), Ac(X), [M/H], [α/H], [α/M]) —
                        # do not apply positivity mask and do not re-logarithmize
                        return s.where(np.isfinite(s))
                    else:
                        # Regular parameter: mask non-physical values and logarithmize
                        TINY_POSITIVE = 1e-250
                        s = s.mask((s <= 0) | (~np.isfinite(s)) | (s < TINY_POSITIVE))
                        return np.log10(s)

                # mode == "log" (Plotly renders log axis internally)
                TINY_POSITIVE = 1e-250
                s = s.mask((s <= 0) | (~np.isfinite(s)) | (s < TINY_POSITIVE))
                return s

            return s

        def _compute_axis_series(col, is_age, scale, age_factor, abund_fmt, is_abund, df):
            """Resolve a column key to a pd.Series, handling Age, derived ratios, abundance, and raw columns."""
            if is_age and "Age" in df.columns:
                s = pd.to_numeric(df["Age"], errors="coerce")
                if scale == "lin":
                    s = s * age_factor
                else:
                    s = s * 1_000_000.0
                return s.mask(s <= 0)
            if col == "MH":
                return compute_mh_surf(df)
            elif col == "alphaH_ONeMgSiS":
                return compute_alpha_h_surf(df, elements=("O", "Ne", "Mg", "Si", "S"))
            elif col == "alphaH_MgSi":
                return compute_alpha_h_surf(df, elements=("Mg", "Si"))
            elif col == "alphaM_ONeMgSiS":
                return compute_alpha_h_surf(df, elements=("O", "Ne", "Mg", "Si", "S")) - compute_mh_surf(df)
            elif col == "alphaM_MgSi":
                return compute_alpha_h_surf(df, elements=("Mg", "Si")) - compute_mh_surf(df)
            elif col == "Li6_Li7":
                return compute_number_ratio(df, "Li6", "Li7")
            elif col == "C12_C13":
                return compute_number_ratio(df, "C12", "C13")
            elif col == "C12_N14":
                return compute_number_ratio(df, "C12", "N14")
            elif col == "N14_N15":
                return compute_number_ratio(df, "N14", "N15")
            elif col == "O16_O17":
                return compute_number_ratio(df, "O16", "O17")
            elif col == "O16_O18":
                return compute_number_ratio(df, "O16", "O18")
            elif is_abund:
                return resolve_abundance_col(col, abund_fmt or "XX", df)
            else:
                return pd.to_numeric(df[col], errors="coerce")

        y_axis_pool = []

        x_is_age = str(xcol or "").lower() == "age"
        y_is_age = str(ycol or "").lower() == "age"
        x_is_abund = _is_abund_col(xcol)
        y_is_abund = _is_abund_col(ycol)
        # Normalize format values (None → "XX")
        x_abund_fmt = x_abund_fmt or "XX"
        y_abund_fmt = y_abund_fmt or "XX"

        # Age unit conversion factors (from Myr in file to target units)
        def get_age_scale_factor(units):
            """Returns multiplier to convert Age from Myr (file data) to selected units"""
            if units == "gyr":
                return 1e-3  # Myr to Gyr
            elif units == "myr":
                return 1.0  # Myr to Myr (no change)
            elif units == "yr":
                return 1e6  # Myr to yr
            else:
                return 1e-3  # default to Gyr

        # Get scale factors for Age axes
        # For kipp mode x_is_age will be forced False below, but x_age_factor is still needed
        x_age_factor = get_age_scale_factor(x_age_units) if (x_is_age or kipp_env_mode) else 1e-3
        y_age_factor = get_age_scale_factor(y_age_units) if y_is_age and yscale == "lin" else 1e-3  # default Gyr

        if kipp_env_mode:
            x_is_age = False
            y_is_age = False

        # If overlaying on a frozen Kippenhahn diagram with Age on X:
        # force linear X scale and same age units as the frozen kipp used.
        frozen_has_kipp = bool(
            frozen_images and any(isinstance(e, dict) and e.get("is_kipp") for e in frozen_images)
        )
        if frozen_has_kipp and x_is_age:
            xscale_eff = "lin"
            kipp_entry = next((e for e in frozen_images if isinstance(e, dict) and e.get("is_kipp")), None)
            if kipp_entry and kipp_entry.get("x_age_units"):
                x_age_units = kipp_entry["x_age_units"]
                x_age_factor = get_age_scale_factor(x_age_units)

        Trace = go.Scatter
        step = int(step or 1)
        if step < 1:
            step = 1

        loaded = 0
        loading = 0
        total = len(models or [])

        max_age_tracker = [0.0]
        age_markers_data = []

        def _finite_min_max(seq):
            arr = pd.to_numeric(pd.Series(seq), errors="coerce").dropna().values
            if arr.size == 0:
                return None, None
            return float(np.min(arr)), float(np.max(arr))

        def _interp_age_marker(interp_df, selected_age,
                               x_is_age_flag, y_is_age_flag,
                               xscale_eff, yscale_eff,
                               x_age_factor, y_age_factor,
                               extra_cols=()):
            """
            Interpolate (marker_x, marker_y, extras_dict) for a given selected_age
            using binary search (np.searchsorted) on a pre-sorted interp_df.

            Parameters
            ----------
            interp_df     : DataFrame with columns 'age', 'x', 'y' (+ any extra_cols),
                            sorted by 'age' ascending, NaNs already dropped.
            selected_age  : float, target age in years.
            x_is_age_flag : bool – x axis is the age column itself.
            y_is_age_flag : bool – y axis is the age column itself.
            xscale_eff    : str  – effective x scale ('lin', 'log', 'log(x)').
            yscale_eff    : str  – effective y scale.
            x_age_factor  : float – unit conversion factor for age on x axis.
            y_age_factor  : float – unit conversion factor for age on y axis.
            extra_cols    : iterable of column names to interpolate additionally.

            Returns
            -------
            (marker_x, marker_y, extras) or None if interpolation is impossible.
            extras is a dict {col: value} for each name in extra_cols.
            """
            ages = interp_df['age'].values  # sorted numpy array

            # Binary search: pos is the first index where age >= selected_age
            pos = np.searchsorted(ages, selected_age, side='left')

            if pos >= len(ages):
                return None  # selected_age is beyond the track end

            def _age_to_axis(age_val, scale, factor):
                if scale == 'lin':
                    return age_val * factor / 1e6
                elif scale == 'log(x)':
                    return np.log10(age_val) if age_val > 0 else None
                else:  # 'log' – Plotly handles log itself, pass raw years
                    return age_val

            if pos == 0:
                # selected_age exactly matches (or is before) the first point
                row = interp_df.iloc[0]
                marker_x = _age_to_axis(selected_age, xscale_eff, x_age_factor) if x_is_age_flag else row['x']
                marker_y = _age_to_axis(selected_age, yscale_eff, y_age_factor) if y_is_age_flag else row['y']
                extras = {c: row[c] for c in extra_cols if c in interp_df.columns}
            else:
                row_before = interp_df.iloc[pos - 1]
                row_after = interp_df.iloc[pos]

                age_before = row_before['age']
                age_after = row_after['age']

                t = ((selected_age - age_before) / (age_after - age_before)
                     if age_after != age_before else 0.0)

                if x_is_age_flag:
                    marker_x = _age_to_axis(selected_age, xscale_eff, x_age_factor)
                else:
                    marker_x = row_before['x'] + t * (row_after['x'] - row_before['x'])

                if y_is_age_flag:
                    marker_y = _age_to_axis(selected_age, yscale_eff, y_age_factor)
                else:
                    marker_y = row_before['y'] + t * (row_after['y'] - row_before['y'])

                extras = {}
                for c in extra_cols:
                    if c in interp_df.columns:
                        extras[c] = row_before[c] + t * (row_after[c] - row_before[c])

            return marker_x, marker_y, extras

        def _compute_age_marker_position(marker_data, selected_age,
                                          xscale_eff, yscale_eff,
                                          x_age_factor, y_age_factor,
                                          x_scale, y_scale):
            """
            Compute (marker_x, marker_y, is_at_max_age) for the age-tracker dot
            of one model at the given age, or (None, None, False) when it
            shouldn't be shown (age out of range / not enough data).

            Pure function of (marker_data, selected_age, axis settings) — used both
            for the full figure rebuild and for the fast Patch()-only update path,
            so the two stay perfectly in sync. Caches the cleaned/sorted interpolation
            frame on `marker_data` itself (keyed '_interp_df') so repeated calls across
            slider ticks skip the dropna/sort/reset_index work.
            """
            if selected_age <= 0:
                return None, None, False

            x_plot = marker_data['x_plot']
            y_plot = marker_data['y_plot']
            age_array = marker_data['age_array']
            x_is_age_flag = marker_data['x_is_age']
            y_is_age_flag = marker_data['y_is_age']

            if age_array.isna().all():
                return None, None, False

            max_age_for_model = age_array.max()
            is_at_max_age = False

            def _age_to_axis(age_val, scale, factor):
                if scale == 'lin':
                    return age_val * factor / 1e6
                elif scale == 'log(x)':
                    return np.log10(age_val) if age_val > 0 else None
                else:
                    return age_val

            if selected_age >= max_age_for_model - 1.0:
                last_idx = age_array.last_valid_index()
                if pd.isna(last_idx):
                    return None, None, False

                if x_is_age_flag:
                    marker_x = _age_to_axis(max_age_for_model, xscale_eff, x_age_factor)
                else:
                    marker_x = x_plot.iloc[x_plot.index.get_loc(last_idx)]

                if y_is_age_flag:
                    marker_y = _age_to_axis(max_age_for_model, yscale_eff, y_age_factor)
                else:
                    marker_y = y_plot.iloc[y_plot.index.get_loc(last_idx)]

                is_at_max_age = True
            else:
                interp_df = marker_data.get('_interp_df')
                if interp_df is None:
                    interp_df = pd.DataFrame({
                        'age': age_array, 'x': x_plot, 'y': y_plot
                    }).dropna().sort_values('age').reset_index(drop=True)
                    marker_data['_interp_df'] = interp_df

                if len(interp_df) < 2:
                    return None, None, False
                if selected_age < interp_df['age'].iloc[0] or selected_age > interp_df['age'].iloc[-1]:
                    return None, None, False

                result = _interp_age_marker(
                    interp_df, selected_age,
                    x_is_age_flag, y_is_age_flag,
                    xscale_eff, yscale_eff,
                    x_age_factor, y_age_factor,
                )
                if result is None:
                    return None, None, False
                marker_x, marker_y, _ = result

            if not (pd.notna(marker_x) and pd.notna(marker_y)):
                return None, None, False

            if x_scale is not None and not x_is_age_flag:
                marker_x = marker_x / x_scale
            if y_scale is not None and not y_is_age_flag:
                marker_y = marker_y / y_scale

            return marker_x, marker_y, is_at_max_age

        def _compute_isochrone_point(marker_data, selected_age,
                                      xscale_eff, yscale_eff,
                                      x_age_factor, y_age_factor,
                                      x_scale, y_scale):
            """
            Compute (marker_x, marker_y, marker_logg, iso_sort_val) for one model's
            contribution to the isochrone overlay at the given age, or None when
            the model has no usable point at this age.

            Mirrors _compute_age_marker_position's caching/sharing strategy so the
            isochrone overlay can also be refreshed via the fast Patch()-only path.
            """
            if selected_age <= 0:
                return None

            x_plot = marker_data['x_plot']
            y_plot = marker_data['y_plot']
            age_array = marker_data['age_array']
            logg_array = marker_data.get('logg_array')
            iso_sort_array = marker_data.get('iso_sort_array')
            x_is_age_flag = marker_data['x_is_age']
            y_is_age_flag = marker_data['y_is_age']

            if age_array.isna().all():
                return None

            max_age_for_model = age_array.max()
            marker_x = marker_y = marker_logg = None
            iso_sort_val = None

            def _age_to_axis(age_val, scale, factor):
                if scale == 'lin':
                    return age_val * factor / 1e6
                elif scale == 'log(x)':
                    return np.log10(age_val) if age_val > 0 else None
                else:
                    return age_val

            if selected_age >= max_age_for_model * 0.999:
                last_idx = age_array.idxmax()

                if x_is_age_flag:
                    marker_x = _age_to_axis(max_age_for_model, xscale_eff, x_age_factor)
                else:
                    marker_x = x_plot.iloc[x_plot.index.get_loc(last_idx)]

                if y_is_age_flag:
                    marker_y = _age_to_axis(max_age_for_model, yscale_eff, y_age_factor)
                else:
                    marker_y = y_plot.iloc[y_plot.index.get_loc(last_idx)]

                if logg_array is not None:
                    marker_logg = logg_array.iloc[logg_array.index.get_loc(last_idx)]
                if iso_sort_array is not None:
                    iso_sort_val = iso_sort_array.iloc[iso_sort_array.index.get_loc(last_idx)]
            else:
                interp_df = marker_data.get('_iso_interp_df')
                extra_cols = marker_data.get('_iso_extra_cols')
                if interp_df is None:
                    extra_cols = []
                    df_kwargs = {'age': age_array, 'x': x_plot, 'y': y_plot}
                    dropna_subset = ['age', 'x', 'y']

                    if logg_array is not None:
                        df_kwargs['logg'] = logg_array
                        extra_cols.append('logg')
                    if iso_sort_array is not None:
                        df_kwargs['iso_sort'] = iso_sort_array
                        if logg_array is None:
                            df_kwargs['logg'] = iso_sort_array
                            if 'logg' not in extra_cols:
                                extra_cols.append('logg')
                        extra_cols.append('iso_sort')

                    interp_df = (pd.DataFrame(df_kwargs)
                                 .dropna(subset=dropna_subset)
                                 .sort_values('age')
                                 .reset_index(drop=True))
                    marker_data['_iso_interp_df'] = interp_df
                    marker_data['_iso_extra_cols'] = extra_cols

                if len(interp_df) < 2:
                    return None
                if selected_age < interp_df['age'].iloc[0] or selected_age > interp_df['age'].iloc[-1]:
                    return None

                result = _interp_age_marker(
                    interp_df, selected_age,
                    x_is_age_flag, y_is_age_flag,
                    xscale_eff, yscale_eff,
                    x_age_factor, y_age_factor,
                    extra_cols=extra_cols,
                )
                if result is None:
                    return None
                marker_x, marker_y, extras = result

                if 'logg' in extras:
                    marker_logg = extras['logg']
                if 'iso_sort' in extras:
                    iso_sort_val = float(extras['iso_sort'])

            if not (pd.notna(marker_x) and pd.notna(marker_y)):
                return None

            if x_scale is not None and not x_is_age_flag:
                marker_x = marker_x / x_scale
            if y_scale is not None and not y_is_age_flag:
                marker_y = marker_y / y_scale

            if iso_sort_val is None:
                iso_sort_val = marker_logg

            return marker_x, marker_y, marker_logg, iso_sort_val

        # --- Fast path: the age slider (or the linked age-input box) is the ONLY
        # thing that changed. Instead of rebuilding the whole figure (re-loading
        # dataframes, redrawing every track/phase trace, re-shipping everything to
        # the browser) just to move one small dot, patch the existing age-marker
        # and isochrone traces in place via Patch(). The cache is populated by every
        # full render below and mirrors exactly the trace layout it just produced,
        # so trace indices stay valid as long as nothing else has changed — which is
        # guaranteed here since every other Input would itself trigger a full render.
        # (Placed after the two helpers above so it can call them directly.) ---
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        if trigger_id == "age-slider" and AGE_MARKER_CACHE:
            cache = AGE_MARKER_CACHE
            patched = Patch()

            if not cache['kipp_env_mode']:
                for i, marker_data in enumerate(cache['age_markers_data']):
                    mx, my, is_max = _compute_age_marker_position(
                        marker_data, selected_age,
                        cache['xscale_eff'], cache['yscale_eff'],
                        cache['x_age_factor'], cache['y_age_factor'],
                        cache['x_scale'], cache['y_scale'],
                    )
                    idx = cache['marker_start_idx'] + i
                    patched['data'][idx]['x'] = [mx] if mx is not None else []
                    patched['data'][idx]['y'] = [my] if my is not None else []
                    patched['data'][idx]['marker']['symbol'] = 'square' if is_max else 'circle'

            if cache['isochrone_idx'] is not None:
                points = []
                for marker_data in cache['age_markers_data']:
                    point = _compute_isochrone_point(
                        marker_data, selected_age,
                        cache['xscale_eff'], cache['yscale_eff'],
                        cache['x_age_factor'], cache['y_age_factor'],
                        cache['x_scale'], cache['y_scale'],
                    )
                    if point is not None:
                        points.append(point)

                iso_x, iso_y = [], []
                if len(points) >= 2:
                    asc = cfg.isochrone_sort_ascending
                    if all(p[3] is not None and pd.notna(p[3]) for p in points):
                        points.sort(key=lambda p: p[3], reverse=(not asc))
                    else:
                        points.sort(key=lambda p: p[0], reverse=(not asc))
                    iso_x = [p[0] for p in points]
                    iso_y = [p[1] for p in points]

                idx = cache['isochrone_idx']
                patched['data'][idx]['x'] = iso_x
                patched['data'][idx]['y'] = iso_y

            corrected_age_input = selected_age / 1e6
            if selected_age > cache['max_age_among_models']:
                corrected_age_input = cache['max_age_among_models'] / 1e6

            return (patched, no_update, no_update,
                    no_update, no_update, corrected_age_input)

        def _axis_scale_factor(vmin, vmax):
            """
            Power-of-10 scale factor (e.g. 1e-5) for matplotlib-like offset '×10^k'.
            Returns None when scaling is not desired.
            """
            if vmin is None or vmax is None:
                return None
            m = max(abs(float(vmin)), abs(float(vmax)))
            if not np.isfinite(m) or m == 0.0:
                return None

            exp = int(np.floor(np.log10(m)))

            # Enable only for milli and smaller (m, µ, n, p...)
            if exp <= -3:
                return 10.0 ** exp

            return None

        has_models = models and len(models) > 0

        if not has_models:
            x_title = _axis_title(xcol, x_is_age, xscale_eff,
                                  x_age_units if (x_is_age and xscale_eff == "lin") else None,
                                  abund_fmt=x_abund_fmt if x_is_abund else None)
            y_title = _axis_title(ycol, y_is_age, yscale_eff,
                                  y_age_units if (y_is_age and yscale_eff == "lin") else None,
                                  abund_fmt=y_abund_fmt if y_is_abund else None)

            x_tick_size_static = cfg.axis_tick_font_size
            y_tick_size_static = cfg.axis_tick_font_size

            xaxis_settings = dict(
                tickfont=dict(size=x_tick_size_static, family="Times New Roman, STIX Two Text, serif"),
                title_font=dict(size=cfg.axis_title_font_size, family="Times New Roman, STIX Two Text, serif"),
                exponentformat="power",
                showgrid=True,
                gridcolor=cfg.grid_rgba_str,
                gridwidth=1,
                showline=True,
                linecolor="#a6a6a6",
                linewidth=1.5,
                mirror=True,
                layer="above traces",
                automargin=False,
                type="log" if xscale_eff == "log" else "linear",
            )

            if x_is_age and xscale_eff == "lin":
                xaxis_settings["exponentformat"] = "none"
                xaxis_settings["separatethousands"] = True

            yaxis_settings = dict(
                tickfont=dict(size=y_tick_size_static, family="Times New Roman, STIX Two Text, serif"),
                title_font=dict(size=cfg.axis_title_font_size, family="Times New Roman, STIX Two Text, serif"),
                exponentformat="power",
                showgrid=True,
                gridcolor=cfg.grid_rgba_str,
                gridwidth=1,
                showline=True,
                linecolor="#a6a6a6",
                linewidth=1.5,
                mirror=True,
                layer="above traces",
                automargin=False,
                type="log" if yscale_eff == "log" else "linear",
            )

            # For linear Age axis: disable exponents and add thousands separators
            if y_is_age and yscale_eff == "lin":
                yaxis_settings["exponentformat"] = "none"
                yaxis_settings["separatethousands"] = True

            fig.update_layout(
                uirevision=f"{xcol}|{ycol}|empty",
                xaxis_title=x_title,
                yaxis_title=y_title,
                title=None,
                margin=dict(t=2, r=620, b=100, l=90),
                xaxis=xaxis_settings,
                yaxis=yaxis_settings,
                template="plotly_white",
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend_font=dict(size=cfg.legend_font_size),
                legend=dict(
                    x=1.02,
                    xanchor="left",
                    y=1,
                    yanchor="top",
                    itemwidth=50,
                    tracegroupgap=0,
                    font=dict(
                        family="Times New Roman",
                        size=cfg.legend_font_size,
                    )
                ),
                modebar=dict(
                    orientation='v',
                    bgcolor='rgba(255,255,255,0.8)',
                    remove=['autoScale2d', 'lasso2d', 'select2d'],
                )
            )

            if frozen_images:
                bg = [dict(source=e["src"], xref="paper", yref="paper",
                           x=0, y=1, sizex=1, sizey=1,
                           xanchor="left", yanchor="top", layer="below", sizing="stretch", opacity=1.0)
                      for e in frozen_images if isinstance(e, dict) and e.get("src")]
                if bg: fig.update_layout(images=bg)
            return fig, "", True, 1e7, {0: '0', 2.5e6: '2.5', 5e6: '5.0', 7.5e6: '7.5', 1e7: '10.0'}, 0

        def _track_conv_zones(zones_data):
            """
            STAREVOL doesn't keep a stable identity for its numbered convective
            zones: which slot (conv2, conv3, ...) describes a given physical
            layer can hand off from one saved timestep to the next -- most
            visibly during marginal/borderline ("flickering") instability near
            the surface on the giant branch, where two very close, thin
            sub-layers swap conv2/conv3 labels every few timesteps (sometimes
            even merging into one reported interval and splitting again).
            Drawing "conv2's polygon" and "conv3's polygon" verbatim then means
            each polygon zigzags between the two physical layers' positions, so
            their translucent fills (identical colour/opacity) cross and stack
            into a visibly darker, jagged "duplicated zone" -- not a real
            second zone.

            Re-derive stable tracks by greedily matching each row's active
            interval(s) to whichever previously-seen interval they sit closest
            to (merging same-row overlaps first), instead of trusting
            STAREVOL's hand-off-prone raw label. A track that finds no match in
            a row is retired immediately -- so a genuinely new/relocated layer
            never gets stitched onto an unrelated old one (it simply starts a
            fresh, separate segment in the first free output slot, exactly like
            today's "zone with two disjoint lifetimes" case) -- while a layer
            that reappears a few rows later naturally re-attaches to its own
            recent position. This keeps each output slot following one physical
            layer smoothly, so coincident/overlapping/handed-off zones render
            as correctly separated (or correctly merged, when STAREVOL itself
            reports them merged) fills. Everything downstream (fills, rings,
            contours, hover) keeps consuming conv_zones_data exactly as before.
            """
            if len(zones_data) <= 1:
                return zones_data

            raw = [(mt.to_numpy(dtype=float), mb.to_numpy(dtype=float)) for _, mt, mb in zones_data]
            n_rows = raw[0][0].size
            n_slots = len(raw)
            out_mt = [np.zeros(n_rows) for _ in range(n_slots)]
            out_mb = [np.zeros(n_rows) for _ in range(n_slots)]
            track_lo = [None] * n_slots
            track_hi = [None] * n_slots

            for r in range(n_rows):
                intervals = []
                for mt_arr, mb_arr in raw:
                    mt_v = mt_arr[r]
                    mb_v = mb_arr[r]
                    if mt_v != 0.0 or mb_v != 0.0:
                        lo, hi = (mb_v, mt_v) if mb_v <= mt_v else (mt_v, mb_v)
                        intervals.append([lo, hi])

                if not intervals:
                    track_lo = [None] * n_slots
                    track_hi = [None] * n_slots
                    continue

                intervals.sort()
                merged = []
                for lo, hi in intervals:
                    if merged and lo <= merged[-1][1]:
                        if hi > merged[-1][1]:
                            merged[-1][1] = hi
                    else:
                        merged.append([lo, hi])

                # Greedily pair each (track, candidate-interval) by closeness,
                # globally smallest distance first, so the clearest matches win
                # before any ambiguous ones are forced into a decision.
                used_tracks, used_intervals = set(), set()
                candidates = []
                for ti in range(n_slots):
                    if track_lo[ti] is None:
                        continue
                    for ii, (lo, hi) in enumerate(merged):
                        dist = abs(lo - track_lo[ti]) + abs(hi - track_hi[ti])
                        candidates.append((dist, ti, ii))
                candidates.sort(key=lambda c: c[0])
                for _dist, ti, ii in candidates:
                    if ti in used_tracks or ii in used_intervals:
                        continue
                    used_tracks.add(ti)
                    used_intervals.add(ii)
                    lo, hi = merged[ii]
                    out_mb[ti][r] = lo
                    out_mt[ti][r] = hi
                    track_lo[ti], track_hi[ti] = lo, hi

                # Unmatched intervals are new/relocated layers -- start a fresh
                # track in the first free slot.
                for ii, (lo, hi) in enumerate(merged):
                    if ii in used_intervals:
                        continue
                    free = next((ti for ti in range(n_slots)
                                 if track_lo[ti] is None and ti not in used_tracks), None)
                    if free is None:
                        continue  # impossible: at most n_slots intervals can coexist
                    out_mb[free][r] = lo
                    out_mt[free][r] = hi
                    track_lo[free], track_hi[free] = lo, hi
                    used_tracks.add(free)

                # Tracks with no match this row go quiet -- retiring them
                # immediately (rather than keeping them "warm") guarantees a
                # visible gap separates any two genuinely different episodes
                # that later reuse the same output slot.
                for ti in range(n_slots):
                    if ti not in used_tracks:
                        track_lo[ti] = None
                        track_hi[ti] = None

            index = zones_data[0][1].index
            return [
                (i + 1, pd.Series(out_mt[i], index=index), pd.Series(out_mb[i], index=index))
                for i in range(n_slots)
            ]

        # Radius-mode + log Y: tracks the smallest/largest positive boundary
        # value across ALL overlaid models, so the y-axis range (computed once,
        # after the loop, from these globals -- see kipp_radius_mode branch
        # near "Y-axis" below) covers every model's data, not just the last one.
        kipp_y_floor_global = None
        kipp_y_max_global = None

        for i, path in enumerate(models or []):
            # Check if fast HR mode is enabled
            use_fast_mode = "fast" in (fast_hr_mode or [])

            is_kipp_mode = (str(ycol) == "__kipp_env")
            kipp_radius_mode = (is_kipp_mode and str(kipp_coord or "M") == "R")
            # Radius-mode + log Y: y=0 (stellar centre, R=0 / m_base=0) is -Infinity
            # in log space. Plotly then renders any "toself" polygon edge that touches
            # such a point as a spurious diagonal fold instead of a clean fill down to
            # the plot's bottom -- see _floor_y_log below for the fix.
            y_log_mode = bool(kipp_radius_mode and yscale_eff == "log")

            # 0) If full load finished — promote to FULL_CACHE, evict intermediate caches
            fut = FULL_FUTURES.get(path)
            if fut is not None and fut.done():
                try:
                    FULL_CACHE[path] = fut.result()
                    KIPP_CACHE.pop(path, None)   # redundant once full data is available
                    SURF_CACHE.pop(path, None)   # redundant once full data is available
                except Exception:
                    pass
                FULL_FUTURES.pop(path, None)

            # 0b) If kipp load finished — promote to KIPP_CACHE
            kfut = KIPP_FUTURES.get(path)
            if kfut is not None and kfut.done():
                try:
                    KIPP_CACHE[path] = kfut.result()
                except Exception:
                    pass
                KIPP_FUTURES.pop(path, None)

            # 0c) If surf load finished — promote to SURF_CACHE
            sfut = SURF_FUTURES.get(path)
            if sfut is not None and sfut.done():
                try:
                    SURF_CACHE[path] = sfut.result()
                except Exception:
                    pass
                SURF_FUTURES.pop(path, None)

            # 1) Choose dataframe to plot now
            _model_folder, _model_stem = parse_model_path(path)
            if use_fast_mode:
                # Fast HR mode: only load .hr files, never kick off full load
                if path not in HR_CACHE:
                    try:
                        HR_CACHE[path] = load_hr_only(_model_folder, stem=_model_stem)
                    except Exception:
                        continue
                df = HR_CACHE[path].copy()
            elif path in FULL_CACHE:
                # Full mode: use full cache if available
                df = FULL_CACHE[path].copy()
                loaded += 1
            else:
                # Full mode: start with HR, upgrade to SURF if ready
                if path not in HR_CACHE:
                    try:
                        HR_CACHE[path] = load_hr_only(_model_folder, stem=_model_stem)
                    except Exception:
                        continue

                if path in SURF_CACHE:
                    df = SURF_CACHE[path].copy()
                else:
                    df = HR_CACHE[path].copy()

                # Kick off background full load (only once)
                if path not in FULL_FUTURES:
                    try:
                        FULL_FUTURES[path] = executor.submit(registry.load_model, path)
                    except Exception:
                        pass
                else:
                    if not FULL_FUTURES[path].done():
                        loading += 1

                # Kick off surf load proactively (only once)
                if path not in SURF_FUTURES:
                    try:
                        SURF_FUTURES[path] = executor.submit(
                            load_surf_model, _model_folder, _model_stem)
                    except Exception:
                        pass

                # Kipp mode: upgrade df to KIPP_CACHE if available, else launch kipp load
                if is_kipp_mode:
                    if path in KIPP_CACHE:
                        df = KIPP_CACHE[path].copy()
                    elif path not in KIPP_FUTURES:
                        try:
                            KIPP_FUTURES[path] = executor.submit(
                                load_kipp_model, _model_folder, _model_stem)
                        except Exception:
                            pass

            abund_ratio_arr = ["MH", "alphaH_ONeMgSiS", "alphaH_MgSi", "alphaM_ONeMgSiS", "alphaM_MgSi",
                               "Li6_Li7", "C12_C13", "C12_N14", "N14_N15", "O16_O17", "O16_O18"]
            if not is_kipp_mode:
                missing_x = (xcol and (xcol not in df.columns) and (
                        xcol not in abund_ratio_arr) and not _is_abund_col(xcol))
                missing_y = (ycol and (ycol not in df.columns) and (
                        ycol not in abund_ratio_arr) and not _is_abund_col(ycol))

                if missing_x or missing_y:
                    continue
            else:
                # Kippenhahn needs hr+v3+v4(+v12) columns (from KIPP_CACHE or FULL_CACHE)
                if kipp_radius_mode:
                    required_cols = {"Age", "env_Rb", "R"}
                else:
                    required_cols = {"Age", "env_Mb", "M"}
                if not required_cols.issubset(set(df.columns)):
                    continue

            if "Teff" in df.columns and "logTeff" not in df.columns:
                df["logTeff"] = np.log10(pd.to_numeric(df["Teff"], errors="coerce"))
            if "logL" in df.columns and "L" not in df.columns:
                df["L"] = 10 ** pd.to_numeric(df["logL"], errors="coerce")

            # --- Special plot: Kippenhahn diagram (convective envelope only) ---
            if is_kipp_mode:
                # Age is stored in Myr in STAREVOL outputs -> convert to selected units
                age_raw = pd.to_numeric(df["Age"], errors="coerce")
                age_max = float(np.nanmax(age_raw.to_numpy())) if age_raw.notna().any() else np.nan
                age_gyr = age_raw * x_age_factor

                if kipp_radius_mode:
                    # Radius coordinate: R is the absolute total radius (R_sun, from .hr).
                    # env_Rb/conv1_Rb/conv1_Rt (from .v3) are normalized fractions (R/R*) —
                    # convert to absolute R_sun by multiplying by R. conv2-6_Rb/Rt (from
                    # .v12) are already absolute R_sun.
                    m_tot = pd.to_numeric(df["R"], errors="coerce")
                    m_base = pd.to_numeric(df["env_Rb"], errors="coerce") * m_tot

                    conv_zones_data = []
                    for i in range(1, 6):
                        mt_col = f"conv{i}_Rt"
                        mb_col = f"conv{i}_Rb"
                        if mt_col in df.columns and mb_col in df.columns:
                            mt_vals = pd.to_numeric(df[mt_col], errors="coerce")
                            mb_vals = pd.to_numeric(df[mb_col], errors="coerce")
                            if i == 1:
                                # .v3 values are normalized fractions (R/R*)
                                mt_vals = mt_vals * m_tot
                                mb_vals = mb_vals * m_tot
                            if np.any(mt_vals.to_numpy() != 0) or np.any(mb_vals.to_numpy() != 0):
                                conv_zones_data.append((i, mt_vals, mb_vals))
                else:
                    m_tot = pd.to_numeric(df["M"], errors="coerce")
                    m_base = pd.to_numeric(df["env_Mb"], errors="coerce")

                    conv_zones_data = []
                    for i in range(1, 6):
                        mt_col = f"conv{i}_Mt"
                        mb_col = f"conv{i}_Mb"
                        if mt_col in df.columns and mb_col in df.columns:
                            mt_vals = df[mt_col].to_numpy()
                            mb_vals = df[mb_col].to_numpy()
                            if np.any(mt_vals != 0) or np.any(mb_vals != 0):
                                conv_zones_data.append((i, pd.to_numeric(df[mt_col], errors="coerce"),
                                                        pd.to_numeric(df[mb_col], errors="coerce")))

                conv_zones_data = _track_conv_zones(conv_zones_data)

                valid_env = age_gyr.notna() & m_tot.notna() & m_base.notna() & (m_base > 0) & (m_base < m_tot)
                if not valid_env.any():
                    missing_label = "env_Rb" if kipp_radius_mode else "env_Mb"
                    fig.add_annotation(
                        text=f"{Path(path).name}: no convective envelope ({missing_label} missing or invalid)",
                        xref="paper", yref="paper", x=0.01, y=0.98, showarrow=False, align="left"
                    )
                    continue

                # Background evolutionary-stage bands (approx., using phase if available)
                if "__phase" in df.columns:
                    ph = pd.to_numeric(df["__phase"], errors="coerce")
                elif "phase" in df.columns:
                    ph = pd.to_numeric(df["phase"], errors="coerce")
                else:
                    ph = None

                # Thermohaline mixing zone (conv6_* in *.v4/*.v12): draw as semi-transparent reddish area
                therm_mt = None
                therm_mb = None
                if cfg.kipp.show_thermo:
                    if kipp_radius_mode:
                        if "conv6_Rt" in df.columns and "conv6_Rb" in df.columns:
                            therm_mt = pd.to_numeric(df["conv6_Rt"], errors="coerce")
                            therm_mb = pd.to_numeric(df["conv6_Rb"], errors="coerce")
                    else:
                        if "conv6_Mt" in df.columns and "conv6_Mb" in df.columns:
                            therm_mt = pd.to_numeric(df["conv6_Mt"], errors="coerce")
                            therm_mb = pd.to_numeric(df["conv6_Mb"], errors="coerce")

                # Nuclear burning zones (H/He/C/Ne, from *.v5-*.v8): same idea as the
                # thermohaline zone above -- a translucent fill spanning [Xburn_*b, Xburn_*t],
                # drawn wherever the zone is active (Xburn_*t != 0). Each element is
                # independently toggled and coloured via cfg.kipp.show_*burn / *burn_fill / *burn_line.
                burn_zones_data = []
                for _elem, _show, _fill, _line in (
                    ("H",  cfg.kipp.show_Hburn,  cfg.kipp.Hburn_fill,  cfg.kipp.Hburn_line),
                    ("He", cfg.kipp.show_Heburn, cfg.kipp.Heburn_fill, cfg.kipp.Heburn_line),
                    ("C",  cfg.kipp.show_Cburn,  cfg.kipp.Cburn_fill,  cfg.kipp.Cburn_line),
                    ("Ne", cfg.kipp.show_Neburn, cfg.kipp.Neburn_fill, cfg.kipp.Neburn_line),
                ):
                    if not _show:
                        continue
                    if kipp_radius_mode:
                        _mt_col, _mb_col = f"{_elem}burn_Rt", f"{_elem}burn_Rb"
                    else:
                        _mt_col, _mb_col = f"{_elem}burn_Mt", f"{_elem}burn_Mb"
                    if _mt_col not in df.columns or _mb_col not in df.columns:
                        continue
                    _mt_vals = pd.to_numeric(df[_mt_col], errors="coerce")
                    _mb_vals = pd.to_numeric(df[_mb_col], errors="coerce")
                    if kipp_radius_mode:
                        # *burn_Rb/*burn_Rt are normalized fractions (R/R*) -- convert to absolute R_sun
                        _mt_vals = _mt_vals * m_tot
                        _mb_vals = _mb_vals * m_tot
                    if np.any(_mt_vals.to_numpy(dtype=float) != 0.0):
                        burn_zones_data.append((_mt_vals, _mb_vals, _fill, _line))

                # In radius mode with a log Y-axis, the stellar centre (R=0 / m_base=0)
                # is -Infinity in log space. "toself" polygon fills (convective zones,
                # radiative-stage masks, the envelope fill) legitimately reach down to
                # that centre, and Plotly renders any such edge as a spurious diagonal
                # fold instead of a clean fill to the bottom of the plot (reproduced in
                # isolation: a "toself" trace with y=0 vertices on a log axis always
                # draws this fold, regardless of zoom). Replacing those y<=0 vertices
                # with the smallest *positive* value that already occurs in this
                # model's own boundary data sidesteps the rendering quirk without
                # perturbing autorange -- that value was already going to be the
                # axis's effective minimum, so substituting it for y<=0 introduces no
                # new extreme and the fill simply clips flush with the plot's bottom.
                y_floor = None
                y_data_max = None
                if y_log_mode:
                    _floor_candidates = []
                    for _ser in (m_tot, m_base, therm_mt, therm_mb):
                        if _ser is not None:
                            _arr = _ser.to_numpy(dtype=float)
                            _pos = _arr[np.isfinite(_arr) & (_arr > 0)]
                            if _pos.size:
                                _floor_candidates.append(float(_pos.min()))
                    for _, _mt_ser, _mb_ser in conv_zones_data:
                        for _ser in (_mt_ser, _mb_ser):
                            _arr = _ser.to_numpy(dtype=float)
                            _pos = _arr[np.isfinite(_arr) & (_arr > 0)]
                            if _pos.size:
                                _floor_candidates.append(float(_pos.min()))
                    for _mt_ser, _mb_ser, _, _ in burn_zones_data:
                        for _ser in (_mt_ser, _mb_ser):
                            _arr = _ser.to_numpy(dtype=float)
                            _pos = _arr[np.isfinite(_arr) & (_arr > 0)]
                            if _pos.size:
                                _floor_candidates.append(float(_pos.min()))
                    if _floor_candidates:
                        y_floor = min(_floor_candidates)
                        kipp_y_floor_global = (y_floor if kipp_y_floor_global is None
                                               else min(kipp_y_floor_global, y_floor))

                    _y_max = float(np.nanmax(m_tot.to_numpy()))
                    if np.isfinite(_y_max):
                        y_data_max = _y_max
                        kipp_y_max_global = (_y_max if kipp_y_max_global is None
                                             else max(kipp_y_max_global, _y_max))

                def _floor_y_log(arr):
                    """Clamp non-positive/invalid y-vertices to y_floor when on a log
                    radius axis, so "toself" polygon fills never carry a y<=0 vertex
                    (see y_floor comment above for why this is safe for autorange)."""
                    if not y_log_mode or y_floor is None:
                        return arr
                    arr = np.asarray(arr, dtype=float)
                    return np.where(np.isfinite(arr) & (arr > 0), arr, y_floor)

                def _interp_boundary(x_query, x_data, y_data):
                    """Interpolate a zone boundary onto the rings grid the SAME way
                    Plotly draws the line between those exact two data points: on a
                    linear y-axis that's a straight line in (x, y); on a log y-axis
                    the renderer is linear in (x, log10(y)) screen space, so raw-y
                    interpolation would bow the rings' inclusion band away from the
                    actual rendered fill/contour edge. The mismatch is invisible when
                    consecutive samples have similar y, but glaring across a "grows
                    from zero" edge, where y spans orders of magnitude in one step --
                    rings would then cover a visibly different sliver than the fill.
                    Floor non-positive values first so log10 never sees zero (mirrors
                    how the fill polygon itself floors that same vertex)."""
                    if y_log_mode:
                        return 10.0 ** np.interp(x_query, x_data, np.log10(_floor_y_log(y_data)))
                    return np.interp(x_query, x_data, y_data)

                # Fill the convective envelope region between m_base and m_tot (segment-wise to avoid gaps)
                v = valid_env.to_numpy()
                idxs = np.where(v)[0]
                cuts = np.where(np.diff(idxs) > 1)[0] + 1
                chunks = np.split(idxs, cuts)

                dash = dash_styles[i % 6]
                _folder, _stem = parse_model_path(path)
                model_name = _stem if _stem else _folder.name
                if cfg.legend_max_chars is not None:
                    model_name = model_name[:cfg.legend_max_chars]

                prev_chunk_end = -1
                for ch in chunks:
                    if ch.size < 2:
                        continue

                    # Extended index range used ONLY for per-zone (conv_zones_data)
                    # drawing: absorbs any "this zone is active but env_Mb isn't
                    # valid yet" stretch immediately preceding this chunk -- e.g.
                    # an early fully-convective phase described solely by conv1
                    # (env_Mb == 0, excluded by valid_env's m_base > 0 test) before
                    # a radiative core forms and env_Mb becomes meaningfully
                    # positive. Those rows fall in the gap before chunks[0] (or
                    # between chunks) and would otherwise never be visited by any
                    # conv_zones_data sub-loop below, leaving that zone's
                    # fill/rings/contour entirely undrawn -- a "white hole" where
                    # the star genuinely was convective. Consecutive chunks' own
                    # ch_zones never overlap.
                    ch_zones = np.arange(prev_chunk_end + 1, int(ch[-1]) + 1)
                    prev_chunk_end = int(ch[-1])

                    # Whether some conv_zones_data zone already covers the gap
                    # right before this chunk (only possible for chunks[0], since
                    # that's the only place such a gap was found in practice) --
                    # if so, the main envelope fill/boundary lines below must NOT
                    # invent their own (age=0, mb=0) taper there: that would
                    # double-paint on top of the zone's own fill (drawn via
                    # ch_zones, further down).
                    lead_gap_covered = False
                    if (ch is chunks[0]) and (ch[0] > 0):
                        _lead = np.arange(0, int(ch[0]))
                        for _zn, _mts, _mbs in conv_zones_data:
                            _mt_l = _mts.iloc[_lead].to_numpy(dtype=float)
                            _mb_l = _mbs.iloc[_lead].to_numpy(dtype=float)
                            if np.any((_mt_l != 0.0) | (_mb_l != 0.0)):
                                lead_gap_covered = True
                                break

                    x_seg = age_gyr.iloc[ch].to_numpy()
                    mb_seg = m_base.iloc[ch].to_numpy()
                    mt_seg = m_tot.iloc[ch].to_numpy()

                    # Add initial point at x=0 if data doesn't start exactly at 0
                    # -- but only when no conv_zones_data zone already covers that
                    # lead-in gap (see lead_gap_covered above); otherwise this
                    # taper would draw a bogus diagonal on top of that zone's own
                    # (correct) fill and double-paint the overlap.
                    if (ch is chunks[0]) and (x_seg.size > 0) and not lead_gap_covered:
                        first_x = float(x_seg[0])
                        if abs(first_x) > 1e-10:  # Not exactly zero
                            x_seg = np.concatenate([[0.0], x_seg])
                            mb_seg = np.concatenate([[0.0], mb_seg])  # Start from y=0 at t=0
                            mt_seg = np.concatenate([[float(mt_seg[0])], mt_seg])

                    # Polygon: upper boundary (surface) + lower boundary (base) reversed
                    x_poly = np.concatenate([x_seg, x_seg[::-1]])
                    y_poly = np.concatenate([mt_seg, mb_seg[::-1]])

                    fig.add_trace(go.Scatter(
                        x=x_poly,
                        y=_floor_y_log(y_poly),
                        mode="lines",
                        name=model_name,
                        legendgroup=model_name,
                        showlegend=(ch is chunks[0]),  # legend once per model
                        line={"color": cfg.kipp.conv_line, "dash": dash, "width": 0.6},
                        fill="toself",
                        fillcolor=cfg.kipp.conv_fill,
                        hoverinfo="skip",
                    ))

                    for zone_num, mt_series, mb_series in conv_zones_data:
                        mt_zone = mt_series.iloc[ch_zones].to_numpy(dtype=float)
                        mb_zone = mb_series.iloc[ch_zones].to_numpy(dtype=float)
                        x_zone = age_gyr.iloc[ch_zones].to_numpy(dtype=float)
                        # Envelope bounds at the same rows, used below to drop
                        # samples where this numbered zone is just a spurious
                        # duplicate of (part of) the envelope -- see "nested"
                        # comment near `present`.
                        env_mb_zone = m_base.iloc[ch_zones].to_numpy(dtype=float)
                        env_mt_zone = m_tot.iloc[ch_zones].to_numpy(dtype=float)
                        # Parallel array of absolute dataframe row indices, so we
                        # can still recognise "the row right before chunks[0]"
                        # after the x=0 prepend below shifts everything by one
                        # (-1 marks the artificial prepended point, which can
                        # never equal a real row index).
                        abs_idx = ch_zones.copy()

                        # Extend to the plot's left edge exactly like the main
                        # envelope fill does for chunks[0] above -- otherwise a
                        # zone that is active right from the start of the data
                        # would leave a sliver gap between x=0 and its first
                        # real sample.
                        if (ch_zones[0] == 0) and (x_zone.size > 0) and (abs(float(x_zone[0])) > 1e-10):
                            x_zone = np.concatenate([[0.0], x_zone])
                            mt_zone = np.concatenate([[float(mt_zone[0])], mt_zone])
                            mb_zone = np.concatenate([[float(mb_zone[0])], mb_zone])
                            env_mb_zone = np.concatenate([[float(env_mb_zone[0])], env_mb_zone])
                            env_mt_zone = np.concatenate([[float(env_mt_zone[0])], env_mt_zone])
                            abs_idx = np.concatenate([[-1], abs_idx])

                        present = (mt_zone != 0.0) | (mb_zone != 0.0)
                        # Some STAREVOL timesteps report a numbered zone (e.g.
                        # conv2) whose entire interval sits inside the
                        # envelope's interval -- the same convective region
                        # described twice, rendered as a spurious nested wedge
                        # on top of the envelope fill. Drop those samples: a
                        # zone is a duplicate-of-envelope sample when its base
                        # is at or above the envelope's base AND its top is at
                        # or below the envelope's top (>= / <= so that the edge
                        # case where the zone's base coincides exactly with the
                        # envelope's base is also excluded -- otherwise a
                        # zero-thickness zone still renders as a bare contour
                        # line on top of the envelope boundary).
                        nested_in_env = (env_mb_zone > 0) & (mb_zone >= env_mb_zone) & (mt_zone <= env_mt_zone)
                        present = present & ~nested_in_env
                        if not np.any(present):
                            continue

                        idxs = np.where(present)[0]
                        cuts = np.where(np.diff(idxs) > 1)[0] + 1
                        segs = np.split(idxs, cuts)

                        for seg in segs:
                            if seg.size < 2:
                                continue

                            s = int(seg[0])
                            e = int(seg[-1])

                            x_part = x_zone[s:e + 1]
                            mt_part = mt_zone[s:e + 1]
                            mb_part = mb_zone[s:e + 1]

                            if (mb_part[0] == 0.0) and (mt_part[0] != 0.0) and (s > 0):
                                x_part = np.concatenate([[x_zone[s - 1]], x_part])
                                mt_part = np.concatenate([[0.0], mt_part])
                                mb_part = np.concatenate([[0.0], mb_part])

                            # Close the seam onto the envelope fill's leading
                            # edge: if this zone was active right up through the
                            # row immediately preceding where valid_env (and
                            # hence the main env_Mb/M descriptor) takes over,
                            # extend the polygon to (age[ch[0]], env_Mb[ch[0]] /
                            # M[ch[0]]) -- the SAME physical convective-zone
                            # boundary, just described by a different column at
                            # that row -- so the two fills meet with no white
                            # sliver between them (STAREVOL hands the
                            # description off between consecutive samples; conv1
                            # and env_* never overlap).
                            if int(abs_idx[e]) == int(ch[0]) - 1:
                                seam_age_g = float(age_gyr.iloc[ch[0]])
                                seam_mt_g = float(m_tot.iloc[ch[0]])
                                seam_mb_g = float(m_base.iloc[ch[0]])
                                prev_age_g = float(x_part[-1])
                                prev_mb_g = float(mb_part[-1])
                                x_part = np.concatenate([x_part, [seam_age_g]])
                                mt_part = np.concatenate([mt_part, [seam_mt_g]])
                                mb_part = np.concatenate([mb_part, [seam_mb_g]])

                                # The lower-boundary transition above is a diagonal
                                # ramp from (prev_age, prev_mb≈0) -- the zone's last
                                # known extent, reaching the centre -- to (seam_age,
                                # seam_mb): the radiative core's first measured size.
                                # That diagonal is also this zone's fill-polygon edge,
                                # but the radiative-mask polygons (drawn elsewhere,
                                # tinted by evolutionary stage) only start AT the seam
                                # row, so the thin triangle between the diagonal, the
                                # seam's vertical line and y=prev_mb is left unpainted
                                # -- a "white triangle" sliver (negligible in mass mode
                                # where the jump is tiny, glaring in radius mode where
                                # it spans a real fraction of R). Paint that sliver
                                # here with the same stage colour the mask would use,
                                # so the seam is gap-free on both sides.
                                if (seam_mb_g > prev_mb_g) and (ph is not None):
                                    _seam_phase = ph.iloc[ch[0]]
                                    if np.isfinite(_seam_phase):
                                        _seam_phase_i = int(_seam_phase)
                                        for _ph_set, _mask_c in cfg.kipp.stage_specs():
                                            if _seam_phase_i in _ph_set:
                                                fig.add_trace(go.Scatter(
                                                    x=[prev_age_g, seam_age_g, seam_age_g],
                                                    y=_floor_y_log([prev_mb_g, seam_mb_g, prev_mb_g]),
                                                    mode="lines",
                                                    showlegend=False,
                                                    legendgroup=model_name,
                                                    line={"width": 0},
                                                    fill="toself",
                                                    fillcolor=_mask_c,
                                                    hoverinfo="skip",
                                                ))
                                                break

                            x_poly_zone = np.concatenate([x_part, x_part[::-1]])
                            y_poly_zone = np.concatenate([mt_part, mb_part[::-1]])

                            fig.add_trace(go.Scatter(
                                x=x_poly_zone,
                                y=_floor_y_log(y_poly_zone),
                                mode="lines",
                                showlegend=False,
                                legendgroup=model_name,
                                line={"color": cfg.kipp.conv_line, "dash": dash, "width": 0.6},
                                fill="toself",
                                fillcolor=cfg.kipp.conv_fill,
                                hoverinfo="skip",
                            ))

                    # --- Thermohaline mixing zone (conv6_Mb/conv6_Mt) ---
                    # Draw ONLY where conv6_Mt != 0, as a reddish semi-transparent fill WITHOUT boundary line.
                    if therm_mt is not None and therm_mb is not None:
                        mt_th = therm_mt.iloc[ch].to_numpy(dtype=float)
                        mb_th = therm_mb.iloc[ch].to_numpy(dtype=float)
                        x_th = age_gyr.iloc[ch].to_numpy(dtype=float)

                        phase_arr = df["phase"].iloc[ch].to_numpy(dtype=int)

                        # lower boundary of convective envelope (mass or radius coordinate)
                        env_mb_arr = m_base.iloc[ch].to_numpy(dtype=float)

                        present_th = (mt_th != 0.0) & (phase_arr == 3) & (mb_th < env_mb_arr)
                        if np.any(present_th):
                            idxs_th = np.where(present_th)[0]
                            cuts_th = np.where(np.diff(idxs_th) > 1)[0] + 1
                            segs_th = np.split(idxs_th, cuts_th)

                            for seg_th in segs_th:
                                if seg_th.size < 2:
                                    continue

                                s_th = int(seg_th[0])
                                e_th = int(seg_th[-1])

                                x_part = x_th[s_th:e_th + 1]
                                mt_part = mt_th[s_th:e_th + 1]
                                mb_part = mb_th[s_th:e_th + 1]

                                # If Mb is 0 at start but Mt already exists, prepend one point at previous time with 0..0
                                if (mb_part[0] == 0.0) and (mt_part[0] != 0.0) and (s_th > 0):
                                    x_part = np.concatenate([[x_th[s_th - 1]], x_part])
                                    mt_part = np.concatenate([[0.0], mt_part])
                                    mb_part = np.concatenate([[0.0], mb_part])

                                x_poly_th = np.concatenate([x_part, x_part[::-1]])
                                y_poly_th = np.concatenate([mt_part, mb_part[::-1]])

                                fig.add_trace(go.Scatter(
                                    x=x_poly_th,
                                    y=_floor_y_log(y_poly_th),
                                    mode="lines",
                                    showlegend=False,
                                    legendgroup=model_name,
                                    line={"color": cfg.kipp.thermo_line, "width": 1},  # no border
                                    fill="toself",
                                    fillcolor=cfg.kipp.thermo_fill,  # reddish semi-transparent
                                    hoverinfo="skip",
                                ))

                    # --- Nuclear burning zones (H/He/C/Ne) ---
                    # Same drawing recipe as the thermohaline zone above, but drawn simply
                    # wherever the zone is active (Mt != 0) -- no phase/envelope constraints,
                    # since burning zones can sit anywhere in the star (core or shell).
                    # Rows where base == top exactly carry zero thickness -- the "fill"
                    # there would be a zero-area sliver that Plotly still strokes as a
                    # bare line, so they're excluded from "present" entirely (this also
                    # naturally splits the zone into separate segments around such a
                    # pinch, rather than drawing a flat zero-height stretch through it).
                    for burn_mt, burn_mb, burn_fill, burn_line in burn_zones_data:
                        mt_b = burn_mt.iloc[ch].to_numpy(dtype=float)
                        mb_b = burn_mb.iloc[ch].to_numpy(dtype=float)
                        x_b = age_gyr.iloc[ch].to_numpy(dtype=float)

                        present_b = (mt_b != 0.0) & (mt_b != mb_b)
                        if not np.any(present_b):
                            continue

                        idxs_b = np.where(present_b)[0]
                        cuts_b = np.where(np.diff(idxs_b) > 1)[0] + 1
                        segs_b = np.split(idxs_b, cuts_b)

                        for seg_b in segs_b:
                            if seg_b.size < 2:
                                continue

                            s_b = int(seg_b[0])
                            e_b = int(seg_b[-1])

                            x_part = x_b[s_b:e_b + 1]
                            mt_part = mt_b[s_b:e_b + 1]
                            mb_part = mb_b[s_b:e_b + 1]

                            if (mb_part[0] == 0.0) and (mt_part[0] != 0.0) and (s_b > 0):
                                x_part = np.concatenate([[x_b[s_b - 1]], x_part])
                                mt_part = np.concatenate([[0.0], mt_part])
                                mb_part = np.concatenate([[0.0], mb_part])

                            x_poly_b = np.concatenate([x_part, x_part[::-1]])
                            y_poly_b = np.concatenate([mt_part, mb_part[::-1]])

                            fig.add_trace(go.Scatter(
                                x=x_poly_b,
                                y=_floor_y_log(y_poly_b),
                                mode="lines",
                                showlegend=False,
                                legendgroup=model_name,
                                line={"color": burn_line, "width": 1},  # no border
                                fill="toself",
                                fillcolor=burn_fill,
                                hoverinfo="skip",
                            ))

                    # "rings" stipple fill as a deterministic grid (no randomness).
                    # Spacing is defined in fractions of the shown x-range and of envelope thickness.
                    NX = 60  # columns across x-range
                    NY = cfg.kipp.rings_ny
                    ring_size = cfg.kipp.rings_size

                    x_seg = x_seg.astype(float)
                    mb_seg = mb_seg.astype(float)
                    mt_seg = mt_seg.astype(float)

                    order = np.argsort(x_seg)
                    xs = x_seg[order]
                    mbs = mb_seg[order]
                    mts = mt_seg[order]

                    good = np.isfinite(xs) & np.isfinite(mbs) & np.isfinite(mts) & (mts > mbs)
                    xs = xs[good]
                    mbs = mbs[good]
                    mts = mts[good]

                    if xs.size >= 2:
                        # Hard clamp of the global age domain: [0, max_age]
                        x_rng = _get_zoom_range(relayout, "xaxis")
                        y_rng = _get_zoom_range(relayout, "yaxis")

                        max_age = float(np.nanmax(age_gyr.to_numpy()))
                        if not np.isfinite(max_age):
                            max_age = float(xs[-1])

                        x0 = 0.0
                        x1 = max_age
                        if x_rng is not None:
                            x0 = max(0.0, float(x_rng[0]))
                            x1 = min(max_age, float(x_rng[1]))
                            if x1 < x0:
                                x0, x1 = x1, x0
                            x0 = max(0.0, x0)
                            x1 = min(max_age, x1)

                        # Grid across CURRENT visible x-range (but never outside [0, max_age])
                        fx = np.linspace(0.0, 1.0, NX)
                        xg = x0 + fx * (x1 - x0)

                        # Interpolate boundaries; clamp x into data range to avoid extrap artifacts
                        xg_clamped = np.clip(xg, float(xs[0]), float(xs[-1]))
                        mb_g = _interp_boundary(xg_clamped, xs, mbs)
                        mt_g = _interp_boundary(xg_clamped, xs, mts)

                        # y-grid bounds: span the CURRENT visible y-range (full plot height).
                        if y_rng is not None:
                            if y_log_mode:
                                # Plotly reports a log-axis zoom range in log10 units.
                                y0, y1 = 10.0 ** float(y_rng[0]), 10.0 ** float(y_rng[1])
                            else:
                                y0, y1 = float(y_rng[0]), float(y_rng[1])
                            if y1 < y0:
                                y0, y1 = y1, y0
                        elif y_log_mode:
                            # A log axis can't include zero -- span the full
                            # positive extent of ALL zones (envelope + every
                            # conv zone + thermohaline; the same data y_floor
                            # is derived from), not just the envelope's own
                            # range. A zone whose radius band sits well below
                            # the envelope's (e.g. a deep convective core) would
                            # otherwise get NO grid rows landing inside its
                            # [mb, mt] band whenever the two ranges don't
                            # overlap -- silently skipping its rings entirely.
                            if y_floor is not None and y_data_max is not None and y_data_max > y_floor:
                                y0 = y_floor
                                y1 = 1.02 * y_data_max
                            else:
                                pos_b = mb_g[mb_g > 0]
                                pos_t = mt_g[mt_g > 0]
                                y0 = float(np.nanmin(pos_b)) if pos_b.size else (
                                    float(np.nanmin(pos_t)) if pos_t.size else 1e-3)
                                y1 = float(np.nanmax(pos_t)) if pos_t.size else y0 * 10.0
                                y1 = 1.02 * y1
                        else:
                            y0 = 0.0
                            y1 = float(np.nanmax(m_tot.to_numpy())) if m_tot.notna().any() else float(np.nanmax(mt_g))
                            y1 = 1.02 * y1

                        def _y_at(frac):
                            """Map a fraction of the plot height [0,1] to a data
                            y-value -- geometric spacing on a log axis, linear
                            spacing on a linear one -- so points laid out at
                            evenly-spaced fractions appear evenly spaced on screen."""
                            if y_log_mode:
                                return y0 * (y1 / y0) ** frac
                            return y0 + frac * (y1 - y0)

                        def _shift(value, frac):
                            """Shift a y boundary "up" by `frac` of the plot height
                            (negative shifts down): additive on a linear axis,
                            multiplicative on a log one, since a fixed pixel gap is
                            a fixed *ratio* in data space when the axis is log."""
                            if y_log_mode:
                                return value * (y1 / y0) ** frac
                            return value + frac * (y1 - y0)

                        fy = np.linspace(0.0, 1, NY)  # fractions of full plot height
                        y_grid = _y_at(fy)

                        if NY >= 2:
                            d_frac = 1.0 / (NY - 1)
                            col_idx_base = np.arange(xg.size)
                        else:
                            d_frac = 0.0
                            col_idx_base = np.arange(xg.size)

                        def _grid_points(xg_local):
                            """Build the (x, y) stipple grid: evenly-spaced fractions
                            of plot height per column, with every other column
                            offset by half a row for a brick-like stipple pattern."""
                            xp = np.repeat(xg_local, NY)
                            frac_p = np.tile(fy, xg_local.size)
                            if NY >= 2:
                                col_idx = np.repeat(col_idx_base[:xg_local.size], NY)
                                frac_p = np.clip(frac_p + (col_idx % 2) * (0.5 * d_frac), 0.0, 1.0)
                            return xp, _y_at(frac_p)

                        # Build full x-y grid
                        x_pts, y_pts = _grid_points(xg)

                        # Keep only points that lie inside the envelope at each x
                        mb_rep = np.repeat(mb_g, NY)
                        mt_rep = np.repeat(mt_g, NY)

                        # Padding so marker radius doesn't cross the boundary visually
                        plot_h = fig.layout.height or 700  # fallback if height is not set
                        eps_frac = max(1e-9, 0.55 * (ring_size + 2) / float(plot_h))

                        inside = (y_pts >= _shift(mb_rep, eps_frac)) & (y_pts <= _shift(mt_rep, -eps_frac))

                        x_pts = x_pts[inside]
                        y_pts = y_pts[inside]

                        fig.add_trace(go.Scatter(
                            x=x_pts,
                            y=y_pts,
                            mode="markers",
                            showlegend=False,
                            legendgroup=model_name,
                            cliponaxis=True,
                            marker=dict(
                                symbol="circle-open",
                                size=ring_size,
                                color=cfg.kipp.rings,
                                line=dict(width=1),
                            ),
                            hoverinfo="skip",
                        ))

                        for zone_num, mt_series, mb_series in conv_zones_data:
                            mt_zone_raw = mt_series.iloc[ch_zones].to_numpy(dtype=float)
                            mb_zone_raw = mb_series.iloc[ch_zones].to_numpy(dtype=float)
                            x_zone_raw = age_gyr.iloc[ch_zones].to_numpy(dtype=float)

                            present_zone = (mt_zone_raw != 0.0) | (mb_zone_raw != 0.0)
                            _env_mb_z = m_base.iloc[ch_zones].to_numpy(dtype=float)
                            _env_mt_z = m_tot.iloc[ch_zones].to_numpy(dtype=float)
                            present_zone = present_zone & ~(
                                (_env_mb_z > 0) & (mb_zone_raw >= _env_mb_z) & (mt_zone_raw <= _env_mt_z)
                            )
                            if not np.any(present_zone):
                                continue

                            # Grid of points for this zone (shared by all its segments)
                            x_pts_zone, y_pts_zone = _grid_points(xg)

                            inside_zone = np.zeros(x_pts_zone.shape, dtype=bool)

                            # Walk contiguous "exists" runs separately — exactly the
                            # same segmentation the filled zone polygon and the
                            # boundary contour use — so a zone with two disjoint
                            # lifetimes (e.g. an early fully-convective phase and a
                            # much-later separate core-convection phase) never gets
                            # bridged by a single phantom np.interp band spanning the
                            # gap between them.
                            idxs_zone = np.where(present_zone)[0]
                            cuts_zone = np.where(np.diff(idxs_zone) > 1)[0] + 1
                            for seg in np.split(idxs_zone, cuts_zone):
                                if seg.size < 2:
                                    continue
                                s, e = int(seg[0]), int(seg[-1])
                                xs_seg = x_zone_raw[s:e + 1]
                                mts_seg = mt_zone_raw[s:e + 1]
                                mbs_seg = mb_zone_raw[s:e + 1]

                                # Same "grows from zero" diagonal taper as the filled
                                # zone polygon's "prepend" trick (see ~line 2478) and
                                # the boundary contour's _capped_segment — so the rings
                                # follow the exact same outline as the fill/contour
                                # instead of stopping abruptly at the first "real"
                                # sample and leaving the tapered sliver bare.
                                if (mbs_seg[0] == 0.0) and (mts_seg[0] != 0.0) and (s > 0):
                                    xs_seg = np.concatenate([[x_zone_raw[s - 1]], xs_seg])
                                    mts_seg = np.concatenate([[0.0], mts_seg])
                                    mbs_seg = np.concatenate([[0.0], mbs_seg])

                                # Same seam-closing extension as the fill polygon
                                # (see ~line 2510): if this zone's presence run
                                # ends on the row right before chunks[0] starts,
                                # extend the interpolation domain to (age[ch[0]],
                                # env_Mb[ch[0]] / M[ch[0]]) -- the env descriptor's
                                # first point, i.e. the same physical boundary --
                                # so rings are drawn across the connecting sliver
                                # too instead of stopping bare at the seam.
                                if int(ch_zones[e]) == int(ch[0]) - 1:
                                    xs_seg = np.concatenate([xs_seg, [float(age_gyr.iloc[ch[0]])]])
                                    mts_seg = np.concatenate([mts_seg, [float(m_tot.iloc[ch[0]])]])
                                    mbs_seg = np.concatenate([mbs_seg, [float(m_base.iloc[ch[0]])]])

                                if xs_seg.size < 2:
                                    continue

                                # Interpolate this segment's boundaries onto the grid;
                                # clamp x into the segment's own range to avoid
                                # extrapolation, and exclude grid columns outside it
                                # (otherwise rings would leak into a phantom constant-
                                # width band beyond where this segment actually ends).
                                xg_clamped_seg = np.clip(xg, float(xs_seg[0]), float(xs_seg[-1]))
                                mb_g_seg = _interp_boundary(xg_clamped_seg, xs_seg, mbs_seg)
                                mt_g_seg = _interp_boundary(xg_clamped_seg, xs_seg, mts_seg)
                                in_seg_x = (xg >= float(xs_seg[0])) & (xg <= float(xs_seg[-1]))

                                mb_rep_seg = np.repeat(mb_g_seg, NY)
                                mt_rep_seg = np.repeat(mt_g_seg, NY)
                                in_seg_x_rep = np.repeat(in_seg_x, NY)

                                inside_zone |= (
                                    in_seg_x_rep
                                    & (y_pts_zone >= _shift(mb_rep_seg, eps_frac))
                                    & (y_pts_zone <= _shift(mt_rep_seg, -eps_frac))
                                )

                            x_pts_sel = x_pts_zone[inside_zone]
                            y_pts_sel = y_pts_zone[inside_zone]

                            # Draw rings for this zone
                            if x_pts_sel.size > 0:
                                fig.add_trace(go.Scatter(
                                    x=x_pts_sel,
                                    y=y_pts_sel,
                                    mode="markers",
                                    showlegend=False,
                                    legendgroup=model_name,
                                    cliponaxis=True,
                                    marker=dict(
                                        symbol="circle-open",
                                        size=ring_size,
                                        color=cfg.kipp.rings,
                                        line=dict(width=1),
                                    ),
                                    hoverinfo="skip",
                                ))

                        # --- "Clip" rings by overpainting radiative zone above them ---
                        # We overpaint everything BELOW env_Mb with the same stage color (PMS/MS/SGB...),
                        # so rings outside convective envelope are hidden.
                        # Phase transitions are rendered with a smooth gradient (10 interpolated polygons).
                        if ph is not None:
                            ph_seg = ph.iloc[ch].to_numpy()
                            age_seg = age_gyr.iloc[ch].to_numpy()

                            # radiative mask color per phase (must match your vrect colors)
                            stage_specs = cfg.kipp.stage_specs()

                            # --- Color helpers ---
                            import re as _re_color

                            def _parse_plotly_color(c):
                                """Parse 'rgba(r,g,b,a)' or '#rrggbb[aa]' -> (r,g,b,a) floats 0-255/0-1."""
                                c = c.strip()
                                m = _re_color.match(
                                    r'rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)',
                                    c)
                                if m:
                                    r, g, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
                                    a = float(m.group(4)) if m.group(4) is not None else 1.0
                                    return r, g, b, a
                                if c.startswith('#'):
                                    hx = c[1:]
                                    if len(hx) == 3:
                                        hx = ''.join(x * 2 for x in hx)
                                    r = int(hx[0:2], 16)
                                    g = int(hx[2:4], 16)
                                    b = int(hx[4:6], 16)
                                    a = int(hx[6:8], 16) / 255.0 if len(hx) >= 8 else 1.0
                                    return float(r), float(g), float(b), a
                                return 200.0, 200.0, 200.0, 0.5  # fallback grey

                            def _to_rgba(r, g, b, a):
                                return f'rgba({r:.0f},{g:.0f},{b:.0f},{a:.3f})'

                            # --- Build per-index stage color map for transition detection ---
                            # stage_color_idx[k] = index into stage_specs, or -1 if no match
                            stage_color_idx = np.full(ph_seg.shape, -1, dtype=int)
                            for spec_i, (ph_set, _mc) in enumerate(stage_specs):
                                stage_color_idx[np.isin(ph_seg, list(ph_set))] = spec_i

                            # --- Find phase boundary positions (in local ch indices) ---
                            # A boundary between index k and k+1 exists when stage_color_idx changes.
                            # Transition width = 1% of the LEFT phase duration IN AGE UNITS (Gyr),
                            # applied symmetrically on both sides of the boundary.
                            # Set TRANSITION_STEPS = 0 to disable smooth transitions entirely
                            # (sharp boundary, original behaviour).
                            TRANSITION_STEPS = cfg.kipp.transition_steps

                            if TRANSITION_STEPS == 0:
                                # Sharp transitions: draw all phase polygons without any masking
                                for ph_set, mask_color in stage_specs:
                                    mask_sharp = np.isin(ph_seg, list(ph_set)) & np.isfinite(age_seg)
                                    if not mask_sharp.any():
                                        continue
                                    idxs_sh = np.where(mask_sharp)[0]
                                    cuts_sh = np.where(np.diff(idxs_sh) > 1)[0] + 1
                                    for ch_sh in np.split(idxs_sh, cuts_sh):
                                        if ch_sh.size < 2:
                                            continue
                                        idx_g = ch[ch_sh]
                                        right_e = int(ch_sh[-1]) + 1
                                        if right_e < ch.size:
                                            idx_g = np.concatenate([idx_g, [ch[right_e]]])
                                        x_m = age_gyr.iloc[idx_g].to_numpy()
                                        mb_m = m_base.iloc[idx_g].to_numpy()
                                        x_poly_sh = np.concatenate([x_m, x_m[::-1]])
                                        y_poly_sh = np.concatenate([mb_m, np.zeros(mb_m.size)])
                                        fig.add_trace(go.Scatter(
                                            x=x_poly_sh, y=_floor_y_log(y_poly_sh),
                                            mode="lines", showlegend=False,
                                            legendgroup=model_name,
                                            line={"width": 0}, fill="toself",
                                            fillcolor=mask_color, hoverinfo="skip",
                                        ))
                            else:

                                # Boundaries: list of (bk, left_spec_i, right_spec_i, boundary_age, half_age)
                                # bk: last ch-local index of the LEFT phase
                                # boundary_age: age at bk (Gyr) — the transition is centred here
                                # half_age: half-width of the transition zone in Gyr (= 1% of left phase duration)
                                boundaries = []
                                n_seg = len(ph_seg)
                                if n_seg >= 2:
                                    left_start = 0
                                    for k in range(n_seg - 1):
                                        if stage_color_idx[k] != stage_color_idx[k + 1]:
                                            left_spec_i = stage_color_idx[k]
                                            right_spec_i = stage_color_idx[k + 1]
                                            # Duration of left phase in age units
                                            age_left_start = float(age_seg[left_start]) if np.isfinite(
                                                age_seg[left_start]) else float(age_seg[k])
                                            age_left_end = float(age_seg[k]) if np.isfinite(
                                                age_seg[k]) else age_left_start
                                            phase_duration_age = max(0.0, age_left_end - age_left_start)
                                            # Transition fraction from config; PMS→MS uses its own value
                                            left_ph_val = int(ph_seg[k]) if np.isfinite(ph_seg[k]) else -1
                                            right_ph_val = int(ph_seg[k + 1]) if np.isfinite(ph_seg[k + 1]) else -1
                                            if left_ph_val == 1 and right_ph_val == 2:
                                                transition_frac = cfg.kipp.transition_frac_pms
                                            else:
                                                transition_frac = cfg.kipp.transition_frac
                                            half_age = transition_frac * phase_duration_age
                                            boundary_age = age_left_end  # transition centred on last point of left phase
                                            boundaries.append((k, left_spec_i, right_spec_i, boundary_age, half_age))
                                            left_start = k + 1

                                # For each local index, mark it as inside a transition zone if its age
                                # falls within [boundary_age - half_age, boundary_age + half_age].
                                # Also always include the last point of the left phase (bk) and the
                                # first point of the right phase (bk+1) so there is no gap at either edge.
                                transition_mask = np.zeros(n_seg, dtype=bool)
                                for (bk, _li, _ri, boundary_age, half_age) in boundaries:
                                    if half_age <= 0.0:
                                        continue
                                    age_left_pt = float(age_seg[bk]) if np.isfinite(age_seg[bk]) else boundary_age
                                    age_right_base = float(age_seg[bk + 1]) if (
                                                bk + 1 < n_seg and np.isfinite(age_seg[bk + 1])) else boundary_age
                                    raw_lo = min(boundary_age - half_age, age_left_pt)
                                    raw_hi = max(boundary_age + half_age, age_right_base)
                                    next_solid_age = raw_hi
                                    for _fwd in range(bk + 1, n_seg):
                                        a = float(age_seg[_fwd]) if np.isfinite(age_seg[_fwd]) else None
                                        if a is not None and a > raw_hi:
                                            next_solid_age = a
                                            break
                                    transition_mask |= (
                                            np.isfinite(age_seg) &
                                            (age_seg >= raw_lo) &
                                            (age_seg < next_solid_age)
                                    )

                                # --- Helper: draw one radiative-mask polygon strip ---
                                # idx_global_strip: global df indices for this strip
                                # color_str: opaque fillcolor to use
                                def _draw_radiative_strip(idx_global_strip, color_str, added_initial_point_strip=False):
                                    x_m = age_gyr.iloc[idx_global_strip].to_numpy()
                                    mb_m = m_base.iloc[idx_global_strip].to_numpy()

                                    mb_env = mb_m.astype(float)

                                    zone_bounds = []
                                    for _zone_num, _mt_series, _mb_series in conv_zones_data:
                                        mtz = pd.to_numeric(_mt_series.iloc[idx_global_strip],
                                                            errors="coerce").to_numpy(
                                            dtype=float)
                                        mbz = pd.to_numeric(_mb_series.iloc[idx_global_strip],
                                                            errors="coerce").to_numpy(
                                            dtype=float)
                                        if added_initial_point_strip:
                                            mtz = np.concatenate([[float(mtz[0]) if mtz.size > 0 else 0.0], mtz])
                                            mbz = np.concatenate([[float(mbz[0]) if mbz.size > 0 else 0.0], mbz])
                                        valid_z = (np.isfinite(mtz) & np.isfinite(mbz)
                                                   & ((mtz != 0) | (mbz != 0)) & (mtz > mbz))
                                        mtz = np.where(valid_z, mtz, np.nan)
                                        mbz = np.where(valid_z, mbz, np.nan)
                                        zone_bounds.append((mbz, mtz))

                                    MAX_BANDS = len(zone_bounds) + 1
                                    low_bands = np.full((MAX_BANDS, mb_env.size), np.nan, dtype=float)
                                    high_bands = np.full((MAX_BANDS, mb_env.size), np.nan, dtype=float)

                                    for j in range(mb_env.size):
                                        top = mb_env[j]
                                        if not np.isfinite(top) or top < 0:
                                            continue
                                        intervals = []
                                        for (mbz, mtz) in zone_bounds:
                                            lo = mbz[j]
                                            hi = mtz[j]
                                            if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
                                                lo2 = max(0.0, lo)
                                                hi2 = min(top, hi)
                                                if hi2 > lo2:
                                                    intervals.append((lo2, hi2))
                                        intervals.sort(key=lambda t: t[0])
                                        merged = []
                                        for lo, hi in intervals:
                                            if not merged or lo > merged[-1][1]:
                                                merged.append([lo, hi])
                                            else:
                                                merged[-1][1] = max(merged[-1][1], hi)
                                        cur = 0.0
                                        band_idx = 0
                                        for lo, hi in merged:
                                            if lo > cur and band_idx < MAX_BANDS:
                                                low_bands[band_idx, j] = cur
                                                high_bands[band_idx, j] = lo
                                                band_idx += 1
                                            cur = max(cur, hi)
                                        if cur < top and band_idx < MAX_BANDS:
                                            low_bands[band_idx, j] = cur
                                            high_bands[band_idx, j] = top

                                    for b in range(MAX_BANDS):
                                        lo = low_bands[b]
                                        hi = high_bands[b]
                                        valid_b = np.isfinite(lo) & np.isfinite(hi) & (hi > lo) & np.isfinite(x_m)
                                        if not valid_b.any():
                                            continue
                                        idxb = np.where(valid_b)[0]
                                        cutsb = np.where(np.diff(idxb) > 1)[0] + 1
                                        chunksb = np.split(idxb, cutsb)
                                        for cb in chunksb:
                                            if cb.size < 2:
                                                continue
                                            x_cb = x_m[cb]
                                            lo_cb = lo[cb]
                                            hi_cb = hi[cb]
                                            x_poly_m = np.concatenate([x_cb, x_cb[::-1]])
                                            y_poly_m = np.concatenate([hi_cb, lo_cb[::-1]])
                                            fig.add_trace(go.Scatter(
                                                x=x_poly_m,
                                                y=_floor_y_log(y_poly_m),
                                                mode="lines",
                                                showlegend=False,
                                                legendgroup=model_name,
                                                line={"width": 0},
                                                fill="toself",
                                                fillcolor=color_str,
                                                hoverinfo="skip",
                                            ))

                                # === MAIN PHASE POLYGONS (excluding transition zones) ===
                                for ph_set, mask_color in stage_specs:
                                    mask = np.isin(ph_seg, list(ph_set)) & np.isfinite(age_seg) & ~transition_mask
                                    if not mask.any():
                                        continue

                                    idxs2 = np.where(mask)[0]
                                    cuts2 = np.where(np.diff(idxs2) > 1)[0] + 1
                                    chunks2 = np.split(idxs2, cuts2)

                                    for ch2 in chunks2:
                                        if ch2.size < 2:
                                            continue

                                        idx_global = ch[ch2]

                                        # Extend to the next point to avoid white gaps —
                                        # but only if it is also NOT in a transition zone.
                                        # If the next point is in a transition zone, we skip the extension
                                        # (the transition polygons will cover that gap instead).
                                        right_edge = int(ch2[-1]) + 1
                                        if right_edge < ch.size and not transition_mask[right_edge]:
                                            idx_global = np.concatenate([idx_global, [ch[right_edge]]])
                                        elif right_edge < ch.size and transition_mask[right_edge]:
                                            # Include the immediate next point (start of transition) so
                                            # the solid polygon reaches exactly to the transition edge —
                                            # the transition polygons will start there too, no gap.
                                            idx_global = np.concatenate([idx_global, [ch[right_edge]]])

                                        x_m = age_gyr.iloc[idx_global].to_numpy()
                                        mb_m = m_base.iloc[idx_global].to_numpy()

                                        added_initial_point = False
                                        if (ch is chunks[0]) and (ch2[0] == 0) and (x_m.size > 0) and (
                                                float(x_m[0]) > 0.0):
                                            x_m = np.concatenate([[0.0], x_m])
                                            mb_m = np.concatenate([[0.0], mb_m])
                                            added_initial_point = True

                                        mb_env = mb_m.astype(float)

                                        zone_bounds = []
                                        for _zone_num, _mt_series, _mb_series in conv_zones_data:
                                            mtz = pd.to_numeric(_mt_series.iloc[idx_global], errors="coerce").to_numpy(
                                                dtype=float)
                                            mbz = pd.to_numeric(_mb_series.iloc[idx_global], errors="coerce").to_numpy(
                                                dtype=float)
                                            if added_initial_point:
                                                mtz = np.concatenate([[float(mtz[0]) if mtz.size > 0 else 0.0], mtz])
                                                mbz = np.concatenate([[float(mbz[0]) if mbz.size > 0 else 0.0], mbz])
                                            valid_z = (np.isfinite(mtz) & np.isfinite(mbz)
                                                       & ((mtz != 0) | (mbz != 0)) & (mtz > mbz))
                                            mtz = np.where(valid_z, mtz, np.nan)
                                            mbz = np.where(valid_z, mbz, np.nan)
                                            zone_bounds.append((mbz, mtz))

                                        MAX_BANDS = len(zone_bounds) + 1
                                        low_bands = np.full((MAX_BANDS, mb_env.size), np.nan, dtype=float)
                                        high_bands = np.full((MAX_BANDS, mb_env.size), np.nan, dtype=float)

                                        for j in range(mb_env.size):
                                            top = mb_env[j]
                                            if not np.isfinite(top) or top < 0:
                                                continue
                                            intervals = []
                                            for (mbz, mtz) in zone_bounds:
                                                lo = mbz[j]
                                                hi = mtz[j]
                                                if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
                                                    lo2 = max(0.0, lo)
                                                    hi2 = min(top, hi)
                                                    if hi2 > lo2:
                                                        intervals.append((lo2, hi2))
                                            intervals.sort(key=lambda t: t[0])
                                            merged = []
                                            for lo, hi in intervals:
                                                if not merged or lo > merged[-1][1]:
                                                    merged.append([lo, hi])
                                                else:
                                                    merged[-1][1] = max(merged[-1][1], hi)
                                            cur = 0.0
                                            band_idx = 0
                                            for lo, hi in merged:
                                                if lo > cur and band_idx < MAX_BANDS:
                                                    low_bands[band_idx, j] = cur
                                                    high_bands[band_idx, j] = lo
                                                    band_idx += 1
                                                cur = max(cur, hi)
                                            if cur < top and band_idx < MAX_BANDS:
                                                low_bands[band_idx, j] = cur
                                                high_bands[band_idx, j] = top

                                        for b in range(MAX_BANDS):
                                            lo = low_bands[b]
                                            hi = high_bands[b]
                                            valid_b = np.isfinite(lo) & np.isfinite(hi) & (hi > lo) & np.isfinite(x_m)
                                            if not valid_b.any():
                                                continue
                                            idxb = np.where(valid_b)[0]
                                            cutsb = np.where(np.diff(idxb) > 1)[0] + 1
                                            chunksb = np.split(idxb, cutsb)
                                            for cb in chunksb:
                                                if cb.size < 2:
                                                    continue
                                                x_cb = x_m[cb]
                                                lo_cb = lo[cb]
                                                hi_cb = hi[cb]
                                                x_poly_m = np.concatenate([x_cb, x_cb[::-1]])
                                                y_poly_m = np.concatenate([hi_cb, lo_cb[::-1]])
                                                fig.add_trace(go.Scatter(
                                                    x=x_poly_m,
                                                    y=_floor_y_log(y_poly_m),
                                                    mode="lines",
                                                    showlegend=False,
                                                    legendgroup=model_name,
                                                    line={"width": 0},
                                                    fill="toself",
                                                    fillcolor=mask_color,
                                                    hoverinfo="skip",
                                                ))

                                # === TRANSITION ZONE POLYGONS (smooth gradient between phases) ===
                                # Each boundary produces TRANSITION_STEPS vertical rectangle strips.
                                # Strip X-edges are evenly spaced in age — NOT tied to the data grid.
                                # Y-boundaries (mb_env, conv zone bounds) are interpolated via np.interp
                                # onto a dense x-grid within each strip, so strip edges are strictly
                                # vertical and there are zero gaps between adjacent strips.
                                for (bk, left_spec_i, right_spec_i, boundary_age, half_age) in boundaries:
                                    if half_age <= 0.0:
                                        continue

                                    left_color_str = stage_specs[left_spec_i][
                                        1] if left_spec_i >= 0 else 'rgba(200,200,200,0.5)'
                                    right_color_str = stage_specs[right_spec_i][
                                        1] if right_spec_i >= 0 else 'rgba(200,200,200,0.5)'
                                    lr, lg, lb, la = _parse_plotly_color(left_color_str)
                                    rr, rg, rb, ra = _parse_plotly_color(right_color_str)

                                    # Clamp zone edges to actual data points so no gap appears between
                                    # the transition zone and the solid phase polygons on either side.
                                    # Left edge: go back to the last point of the left phase (= bk)
                                    # Right edge: go forward to the first point of the right phase (= bk+1)
                                    age_left_pt = float(age_seg[bk]) if np.isfinite(age_seg[bk]) else boundary_age
                                    # Find the age of the first right-phase point that will NOT be consumed
                                    # by the transition zone, i.e. the first point where the right solid
                                    # polygon will start. We need transition zone to reach exactly there.
                                    # Strategy: extend age_zone_hi past bk+1 to the next point still in
                                    # right phase, so the solid right polygon starts at a point that is
                                    # contiguous with age_zone_hi (no gap between last strip and solid polygon).
                                    age_right_base = float(age_seg[bk + 1]) if (
                                                bk + 1 < n_seg and np.isfinite(age_seg[bk + 1])) else boundary_age
                                    raw_hi = max(boundary_age + half_age, age_right_base)
                                    # Walk forward from bk+1 to find first point beyond raw_hi — that
                                    # is where the solid right polygon will start; transition must reach it.
                                    next_solid_age = raw_hi
                                    for _fwd in range(bk + 1, n_seg):
                                        a = float(age_seg[_fwd]) if np.isfinite(age_seg[_fwd]) else None
                                        if a is not None and a > raw_hi:
                                            next_solid_age = a
                                            break
                                    age_zone_lo = min(boundary_age - half_age, age_left_pt)
                                    age_zone_hi = next_solid_age

                                    # Build interpolation arrays from data points in the zone plus
                                    # immediate neighbours. Always include bk and bk+1 explicitly
                                    # (the boundary points) so the zone is never empty even when
                                    # there are no grid points between boundary_age-half_age and
                                    # boundary_age+half_age (large timestep right after boundary).
                                    in_zone_ext = (
                                            np.isfinite(age_seg) &
                                            (age_seg >= age_zone_lo - 1e-12) &
                                            (age_seg <= age_zone_hi + 1e-12)
                                    )
                                    zone_ext_local = np.where(in_zone_ext)[0]

                                    # Always include bk and bk+1 as anchor points
                                    anchor_pts = [bk]
                                    if bk + 1 < n_seg:
                                        anchor_pts.append(bk + 1)
                                    zone_ext_local = np.unique(np.concatenate([zone_ext_local, anchor_pts]))

                                    # Extend by one point on each side for better interp at edges
                                    ext_lo = max(0, int(zone_ext_local[0]) - 1)
                                    ext_hi = min(n_seg - 1, int(zone_ext_local[-1]) + 1)
                                    interp_local = np.arange(ext_lo, ext_hi + 1)
                                    idx_interp = ch[interp_local]

                                    x_interp = age_gyr.iloc[idx_interp].to_numpy(dtype=float)
                                    mb_interp = m_base.iloc[idx_interp].to_numpy(dtype=float)

                                    # Conv zone bounds for interpolation
                                    zone_interp_bounds = []
                                    for _zone_num, _mt_series, _mb_series in conv_zones_data:
                                        mtz_i = pd.to_numeric(_mt_series.iloc[idx_interp], errors="coerce").to_numpy(
                                            dtype=float)
                                        mbz_i = pd.to_numeric(_mb_series.iloc[idx_interp], errors="coerce").to_numpy(
                                            dtype=float)
                                        valid_z = (np.isfinite(mtz_i) & np.isfinite(mbz_i)
                                                   & ((mtz_i != 0) | (mbz_i != 0)) & (mtz_i > mbz_i))
                                        mtz_i = np.where(valid_z, mtz_i, np.nan)
                                        mbz_i = np.where(valid_z, mbz_i, np.nan)
                                        zone_interp_bounds.append((mbz_i, mtz_i))

                                    # Sort interp arrays by age (should already be sorted, but be safe)
                                    order_i = np.argsort(x_interp)
                                    x_s = x_interp[order_i]
                                    mb_s = mb_interp[order_i]
                                    zone_s = [(mbz[order_i], mtz[order_i]) for (mbz, mtz) in zone_interp_bounds]

                                    # Strip x-edges: TRANSITION_STEPS+1 evenly spaced age values
                                    strip_x_edges = np.linspace(age_zone_lo, age_zone_hi, TRANSITION_STEPS + 1)

                                    for s_idx in range(TRANSITION_STEPS):
                                        x0_s = strip_x_edges[s_idx]
                                        x1_s = strip_x_edges[s_idx + 1]

                                        # Evaluate mb_env and conv zone bounds at exactly x0_s and x1_s
                                        # (the strip edges) — this guarantees strictly vertical strip edges.
                                        x_eval = np.array([x0_s, x1_s])
                                        x_eval_c = np.clip(x_eval, x_s[0], x_s[-1])

                                        mb_eval = np.interp(x_eval_c, x_s, mb_s)

                                        zone_eval = []
                                        for (mbz_s, mtz_s) in zone_s:
                                            valid_mask = np.isfinite(mbz_s) & np.isfinite(mtz_s)
                                            if valid_mask.sum() >= 2:
                                                x_v = x_s[valid_mask]
                                                mbz_v = mbz_s[valid_mask]
                                                mtz_v = mtz_s[valid_mask]
                                                mbz_e = np.interp(x_eval_c, x_v, mbz_v)
                                                mtz_e = np.interp(x_eval_c, x_v, mtz_v)
                                            else:
                                                mbz_e = np.full(2, np.nan)
                                                mtz_e = np.full(2, np.nan)
                                            zone_eval.append((mbz_e, mtz_e))

                                        # Build radiative bands at the 2 eval points
                                        MAX_BANDS = len(zone_eval) + 1
                                        low_d = np.full((MAX_BANDS, 2), np.nan)
                                        high_d = np.full((MAX_BANDS, 2), np.nan)

                                        for j in range(2):
                                            top = mb_eval[j]
                                            if not np.isfinite(top) or top <= 0:
                                                continue
                                            intervals = []
                                            for (mbz_e, mtz_e) in zone_eval:
                                                lo = mbz_e[j]
                                                hi = mtz_e[j]
                                                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                                                    lo2 = max(0.0, lo)
                                                    hi2 = min(top, hi)
                                                    if hi2 > lo2:
                                                        intervals.append((lo2, hi2))
                                            intervals.sort(key=lambda tt: tt[0])
                                            merged = []
                                            for lo, hi in intervals:
                                                if not merged or lo > merged[-1][1]:
                                                    merged.append([lo, hi])
                                                else:
                                                    merged[-1][1] = max(merged[-1][1], hi)
                                            cur = 0.0
                                            bidx = 0
                                            for lo, hi in merged:
                                                if lo > cur and bidx < MAX_BANDS:
                                                    low_d[bidx, j] = cur
                                                    high_d[bidx, j] = lo
                                                    bidx += 1
                                                cur = max(cur, hi)
                                            if cur < top and bidx < MAX_BANDS:
                                                low_d[bidx, j] = cur
                                                high_d[bidx, j] = top

                                        # t at midpoint of this strip → interpolated color
                                        t = (s_idx + 0.5) / TRANSITION_STEPS
                                        ir = lr + t * (rr - lr)
                                        ig = lg + t * (rg - lg)
                                        ib = lb + t * (rb - lb)
                                        ia = la + t * (ra - la)
                                        strip_color = _to_rgba(ir, ig, ib, ia)

                                        # Draw each radiative band as a trapezoid with exactly 4 corners:
                                        # (x0,lo0), (x1,lo1), (x1,hi1), (x0,hi0)
                                        # This guarantees vertical left/right edges and zero inter-strip gaps.
                                        for b in range(MAX_BANDS):
                                            lo0, lo1 = low_d[b, 0], low_d[b, 1]
                                            hi0, hi1 = high_d[b, 0], high_d[b, 1]
                                            if not (np.isfinite(lo0) and np.isfinite(lo1) and
                                                    np.isfinite(hi0) and np.isfinite(hi1)):
                                                continue
                                            if hi0 <= lo0 and hi1 <= lo1:
                                                continue
                                            # Trapezoid: go clockwise
                                            x_poly = [x0_s, x1_s, x1_s, x0_s, x0_s]
                                            y_poly = [lo0, lo1, hi1, hi0, lo0]
                                            fig.add_trace(go.Scatter(
                                                x=x_poly,
                                                y=_floor_y_log(y_poly),
                                                mode="lines",
                                                showlegend=False,
                                                legendgroup=model_name,
                                                line={"width": 0},
                                                fill="toself",
                                                fillcolor=strip_color,
                                                hoverinfo="skip",
                                            ))

                        # Re-draw the envelope base boundary on top (clean line)
                        x_base_line = age_gyr.iloc[ch].to_numpy()
                        y_base_line = m_base.iloc[ch].to_numpy()

                        # Find first point where env_Mb becomes non-zero
                        threshold = 1e-9
                        nonzero_mask = y_base_line > threshold

                        if (ch is chunks[0]) and (not lead_gap_covered) and nonzero_mask.any():
                            first_nonzero_idx = np.where(nonzero_mask)[0][0]
                            x_first = x_base_line[first_nonzero_idx]
                            y_first = y_base_line[first_nonzero_idx]

                            # Add vertical line from (x_first, 0) to (x_first, y_first)
                            # and horizontal line from (0, 0) to (x_first, 0)
                            x_prepend = [0.0, x_first, x_first]
                            y_prepend = [0.0, 0.0, y_first]

                            # Add these points before the rest of the line
                            x_base_line = np.concatenate([x_prepend, x_base_line[first_nonzero_idx:]])
                            y_base_line = np.concatenate([y_prepend, y_base_line[first_nonzero_idx:]])
                        elif (ch is chunks[0]) and (not lead_gap_covered) and (x_base_line.size > 0) and (float(x_base_line[0]) > 0.0):
                            # Fallback: just add point at (0,0) if all values are near zero
                            x_base_line = np.concatenate([[0.0], x_base_line])
                            y_base_line = np.concatenate([[0.0], y_base_line])

                        fig.add_trace(go.Scatter(
                            x=x_base_line,
                            y=y_base_line,
                            mode="lines",
                            showlegend=False,
                            legendgroup=model_name,
                            line={"color": cfg.kipp.boundary_line, "width": 1.2},
                            hoverinfo="skip",
                        ))

                        # Re-draw the stellar surface boundary on top (clean line)
                        x_surf_line = age_gyr.iloc[ch].to_numpy()
                        y_surf_line = m_tot.iloc[ch].to_numpy()

                        # Add initial point at x=0 if needed (same logic as polygon).
                        # Gated on (not lead_gap_covered): when a conv zone already
                        # traces this leading gap with its own (correct, varying-R)
                        # contour, prepending a flat (0, m_tot[ch[0]]) segment here
                        # would overlay a spurious horizontal chord across it -- in
                        # mass mode this is invisible (m_tot is ~constant, so the flat
                        # line hugs the zone's own surface-coincident edge), but in
                        # radius mode R varies sharply with age and the chord cuts
                        # visibly across the zone's sloped boundary.
                        if (ch is chunks[0]) and (not lead_gap_covered) and (x_surf_line.size > 0) and (float(x_surf_line[0]) > 0.0):
                            x_surf_line = np.concatenate([[0.0], x_surf_line])
                            y_surf_line = np.concatenate([[float(y_surf_line[0])], y_surf_line])

                        fig.add_trace(go.Scatter(
                            x=x_surf_line,
                            y=y_surf_line,
                            mode="lines",
                            showlegend=False,
                            legendgroup=model_name,
                            line={"color": cfg.kipp.boundary_line, "width": 1.2},
                            hoverinfo="skip",
                        ))

                        # Redraw boundaries of additional convective zones on top of masks
                        for zone_num, mt_series, mb_series in conv_zones_data:
                            mts_zone = mt_series.iloc[ch_zones].to_numpy()
                            mbs_zone = mb_series.iloc[ch_zones].to_numpy()

                            # Check zone existence
                            if not (np.any(mts_zone != 0) or np.any(mbs_zone != 0)):
                                continue

                            age_zone = age_gyr.iloc[ch_zones].to_numpy()
                            abs_idx_zone = ch_zones.copy()
                            env_mb_ctr = m_base.iloc[ch_zones].to_numpy(dtype=float)
                            env_mt_ctr = m_tot.iloc[ch_zones].to_numpy(dtype=float)

                            # Add initial point at x=0 if needed (same logic as polygon).
                            # Gate on ch_zones[0] == 0 (not "ch is chunks[0]"): with the
                            # extended ch_zones range, the chunk that needs this isn't
                            # necessarily chunks[0] -- it's whichever iteration's
                            # ch_zones happens to start at the very first data row.
                            if (ch_zones[0] == 0) and (age_zone.size > 0) and (float(age_zone[0]) > 0.0):
                                age_zone = np.concatenate([[0.0], age_zone])
                                mts_zone = np.concatenate([[float(mts_zone[0])], mts_zone])
                                mbs_zone = np.concatenate([[float(mbs_zone[0])], mbs_zone])
                                env_mb_ctr = np.concatenate([[float(env_mb_ctr[0])], env_mb_ctr])
                                env_mt_ctr = np.concatenate([[float(env_mt_ctr[0])], env_mt_ctr])
                                abs_idx_zone = np.concatenate([[-1], abs_idx_zone])

                            m_surf = m_tot.iloc[ch_zones].to_numpy()
                            if (ch_zones[0] == 0) and (m_surf.size > 0) and (float(m_surf[0]) > 0.0):
                                # Add the same initial point for comparison
                                m_surf = np.concatenate([[float(m_surf[0])], m_surf])

                            # Canonical env values at the seam row (chunks[0][0]) -- the
                            # same physical boundary this zone's sentinel-adjacent last
                            # row is "handed off" to. Used to extend a segment ending at
                            # chunks[0][0]-1 so the contour matches the fill polygon's
                            # seam-closing edge (~line 2510) instead of the old short
                            # cap onto this row's own (0,0)-sentinel-adjacent values.
                            seam_age = float(age_gyr.iloc[ch[0]])
                            seam_mt = float(m_tot.iloc[ch[0]])
                            seam_mb = float(m_base.iloc[ch[0]])

                            # Exclude timesteps where this zone is fully nested inside the
                            # envelope (same condition as the polygon-fill loop above).
                            nested_ctr = (env_mb_ctr > 0) & (mbs_zone >= env_mb_ctr) & (mts_zone <= env_mt_ctr)

                            tolerance = 1e-6
                            not_surface = np.abs(mts_zone - m_surf) > tolerance
                            valid_mt = (mts_zone != 0) & not_surface & ~nested_ctr

                            # A zone genuinely appears/disappears (as opposed to merging
                            # with the surface, which "not_surface" already excludes) at
                            # samples where BOTH mt and mb are recorded as the file's
                            # "this zone slot is inactive" sentinel (mt == mb == 0).
                            zone_absent = (mts_zone == 0) & (mbs_zone == 0)

                            def _capped_segment(idxs, ys_self, ys_other, canon_self, canon_other):
                                """Build (x, y) for a contiguous run, closing it onto
                                the filled zone polygon's actual outline at birth/death
                                — which is itself asymmetric AND conditional:

                                * At the START, the fill polygon's "prepend" block
                                  inserts a (previous-age, 0) vertex — producing a
                                  DIAGONAL ramp from the neighbouring age up to the
                                  first real point — but ONLY when the zone is born
                                  right at the stellar centre (mb == 0 and mt != 0).
                                  For zones/segments that appear away from the centre
                                  (e.g. internal zones, or brief blips), the fill has
                                  no such prepend and simply closes with a short
                                  VERTICAL edge at the same age. We must branch the
                                  same way, or else those segments grow a long bogus
                                  diagonal spike all the way down to y=0.
                                * At the END there is never a prepend EXCEPT for the
                                  one seam case where this run ends at chunks[0][0]-1
                                  — there the fill polygon extends its edge out to
                                  (seam_age, env's canonical values) instead of
                                  capping at this row's own values (canon_self/
                                  canon_other carry those env values, pre-selected by
                                  the caller to match ys_self/ys_other's roles). In
                                  every other case the polygon still closes with a
                                  plain VERTICAL edge straight down (or up) from the
                                  last real point to the opposite boundary's value at
                                  that same age — we add the same vertical cap here.

                                Matching this exactly ensures the contour and rings
                                agree with the green fill's actual outline instead of
                                drawing spurious bold near-vertical lines where the
                                fill only has a thin natural wall."""
                                seg_x = age_zone[idxs]
                                seg_y = ys_self[idxs]
                                i0, i1 = int(idxs[0]), int(idxs[-1])
                                if i0 > 0 and zone_absent[i0 - 1]:
                                    if (mbs_zone[i0] == 0.0) and (mts_zone[i0] != 0.0):
                                        seg_x = np.concatenate([[age_zone[i0 - 1]], seg_x])
                                        seg_y = np.concatenate([[0.0], seg_y])
                                    else:
                                        seg_x = np.concatenate([[age_zone[i0]], seg_x])
                                        seg_y = np.concatenate([[ys_other[i0]], seg_y])
                                if i1 < zone_absent.size - 1 and zone_absent[i1 + 1]:
                                    if int(abs_idx_zone[i1]) == int(ch[0]) - 1:
                                        # Seam case: STAREVOL hands the description off in a
                                        # single timestep (no intermediate samples), so the
                                        # *fill* polygon must stretch its edge out to (seam_age,
                                        # canon_*) to stay gap-free -- but redrawing that long
                                        # edge here too, in the bold boundary_line style, makes
                                        # an honest one-step data jump look like a rendering
                                        # "spike". Leave it to the fill polygon's own thin,
                                        # semi-transparent border to close the seam quietly.
                                        pass
                                    else:
                                        seg_x = np.concatenate([seg_x, [age_zone[i1]]])
                                        seg_y = np.concatenate([seg_y, [ys_other[i1]]])

                                # A "born/dies at the centre" cap (and any raw
                                # zone-boundary sample of exactly 0) carries a
                                # y<=0 vertex -- impossible on a log radius axis.
                                # Plotly can't draw a segment to/from such a point,
                                # so the line appears to spring up out of nowhere at
                                # its first valid neighbour, looking like a bogus
                                # vertical edge. Floor it to y_floor exactly like the
                                # fill polygon already does (_floor_y_log above) so
                                # the contour traces the SAME diagonal ramp the fill
                                # actually shows, instead of an invalid jump.
                                return seg_x, _floor_y_log(seg_y)

                            # Draw upper boundary segment-wise (contiguous runs only) —
                            # connecting non-adjacent valid points with a straight line
                            # would draw a spurious chord across the gap between them
                            # (e.g. between an early near-surface stretch of this zone
                            # and a much-later deep-interior stretch).
                            idxs_mt = np.where(valid_mt)[0]
                            cuts_mt = np.where(np.diff(idxs_mt) > 1)[0] + 1
                            for seg_mt in np.split(idxs_mt, cuts_mt):
                                if seg_mt.size < 2:
                                    continue
                                seg_x, seg_y = _capped_segment(seg_mt, mts_zone, mbs_zone, seam_mt, seam_mb)
                                fig.add_trace(go.Scatter(
                                    x=seg_x,
                                    y=seg_y,
                                    mode="lines",
                                    showlegend=False,
                                    legendgroup=model_name,
                                    line={"color": cfg.kipp.boundary_line, "width": 1.2},
                                    hoverinfo="skip",
                                ))

                            # Draw lower boundary of the zone (same segment-wise approach)
                            valid_mb = (mbs_zone != 0) & ~nested_ctr
                            idxs_mb = np.where(valid_mb)[0]
                            cuts_mb = np.where(np.diff(idxs_mb) > 1)[0] + 1
                            for seg_mb in np.split(idxs_mb, cuts_mb):
                                if seg_mb.size < 2:
                                    continue
                                seg_x, seg_y = _capped_segment(seg_mb, mbs_zone, mts_zone, seam_mb, seam_mt)
                                fig.add_trace(go.Scatter(
                                    x=seg_x,
                                    y=seg_y,
                                    mode="lines",
                                    showlegend=False,
                                    legendgroup=model_name,
                                    line={"color": cfg.kipp.boundary_line, "width": 1.2},
                                    hoverinfo="skip",
                                ))

                # Axes styling (paper-like look)
                fig.update_layout(
                    xaxis_title=r"$\mathrm{Age},\ \mathrm{Gyr}$",
                    yaxis_title=(r"$\mathrm{r},\ \mathrm{R}_{\odot}$" if kipp_radius_mode
                                 else r"$\mathrm{m},\ \mathrm{M}_{\odot}$"),
                )
                fig.update_yaxes(type="log" if (kipp_radius_mode and yscale_eff == "log") else "linear")
                fig.update_yaxes(showgrid=True, gridcolor=cfg.grid_rgba_str, zeroline=False, ticks="outside",
                                 mirror=True,
                                 linecolor="#c0c0c0", layer="above traces", )

                # Force x-range to stay within [0, max_age] even after zoom-out
                max_age = float(np.nanmax(age_gyr.to_numpy()))
                if np.isfinite(max_age):
                    x_rng = _get_zoom_range(relayout, "xaxis")
                    if x_rng is None:
                        fig.update_yaxes(range=[0.0, max_age], autorange=False)
                    else:
                        x0 = max(0.0, float(x_rng[0]))
                        x1 = min(max_age, float(x_rng[1]))
                        if x1 < x0:
                            x0, x1 = x1, x0
                        fig.update_yaxes(range=[x0, x1], autorange=False)

                m_arr = m_tot.to_numpy()
                m_init = float(m_arr[np.isfinite(m_arr)][0]) if np.isfinite(m_arr).any() else float(np.nanmax(m_arr))
                if np.isfinite(m_init):
                    fig.update_yaxes(range=[0.0, 1.50 * m_init], autorange=False)

                continue

            DERIVED_COLS = {
                "MH",
                "alphaH_ONeMgSiS", "alphaH_MgSi",
                "alphaM_ONeMgSiS", "alphaM_MgSi",
                "Li6_Li7",
                "C12_C13",
                "C12_N14",
                "N14_N15",
                "O16_O17",
                "O16_O18",
            }

            x_is_abund = _is_abund_col(xcol)
            y_is_abund = _is_abund_col(ycol)

            if (xcol not in df.columns and xcol not in DERIVED_COLS and not x_is_abund) or (
                    ycol not in df.columns and ycol not in DERIVED_COLS and not y_is_abund):
                fig.add_annotation(
                    text=f"{Path(path).name}: missing columns — x='{xcol}' in={xcol in df.columns}, y='{ycol}' in={ycol in df.columns}",
                    xref="paper", yref="paper", x=0.01, y=0.98, showarrow=False, align="left"
                )
                continue

            x = _compute_axis_series(xcol, x_is_age, xscale, x_age_factor, x_abund_fmt, x_is_abund, df)
            y = _compute_axis_series(ycol, y_is_age, yscale, y_age_factor, y_abund_fmt, y_is_abund, df)

            # Apply selected/forced axis scale transforms
            x = _apply_axis_scale(x, xscale_eff, xcol, abund_fmt=x_abund_fmt if x_is_abund else None)
            y = _apply_axis_scale(y, yscale_eff, ycol, abund_fmt=y_abund_fmt if y_is_abund else None)

            y_axis_pool.extend(pd.to_numeric(y, errors="coerce").dropna().tolist())

            dff = pd.DataFrame({"__x": x, "__y": y})
            valid = dff.dropna()

            # attach phase if requested (do NOT drop points if phase is missing)
            if "phase" in df.columns:
                phase_s = pd.to_numeric(df["phase"], errors="coerce")
                valid = valid.join(phase_s.rename("__phase"))
            elif "Phase" in df.columns:
                phase_s = pd.to_numeric(df["Phase"], errors="coerce")
                valid = valid.join(phase_s.rename("__phase"))

                # apply phase window ALWAYS if phase exists
            if "__phase" in valid.columns:
                valid = valid[(valid["__phase"] >= phase_min) & (valid["__phase"] <= phase_max)]

            if valid.empty:
                fig.add_annotation(
                    text=f"{Path(path).name}: no points after dropna for x='{xcol}', y='{ycol}'",
                    xref="paper", yref="paper", x=0.01, y=0.95, showarrow=False, align="left"
                )
                continue

            # 3) Subsample by step
            valid = valid.iloc[::step]
            if valid.empty:
                fig.add_annotation(
                    text=f"{Path(path).name}: empty after subsample (step={step})",
                    xref="paper", yref="paper", x=0.01, y=0.92, showarrow=False, align="left"
                )
                continue

            # --- matplotlib-like exponent offset: scale Y values to show ticks like 2..7 and put ×10^k in axis title ---
            x_plot = valid["__x"]
            y_plot = valid["__y"]

            # linestyle: cycle dash styles per model when show_phase + multicolor
            # (phase color occupies hue, so dash distinguishes models)
            dash = dash_styles[i % len(dash_styles)] if (show_phase and cfg.track_multicolor) else "solid"
            width = base_line_width + cfg.track_line_width_step * (i // 10)
            # track color: None = let Plotly cycle automatically
            track_color = None if cfg.track_multicolor else cfg.track_single_color

            _folder, _stem = parse_model_path(path)
            model_name = _stem if _stem else _folder.name
            if cfg.legend_max_chars is not None:
                model_name = model_name[:cfg.legend_max_chars]

            if show_phase and "__phase" in valid.columns:
                # draw 5 colored segments (phase=1..5) but keep gaps (no fake connections)
                legend_group = model_name
                legend_shown = False  # show legend entry only for the first phase that actually exists

                # Extract numpy arrays once outside loop — avoids repeated 338k-row conversions
                phase_arr = valid["__phase"].to_numpy()
                x_arr = x_plot.to_numpy(dtype=float, na_value=np.nan)
                y_arr = y_plot.to_numpy(dtype=float, na_value=np.nan)

                for ph in range(phase_min, phase_max + 1):
                    mask_arr = (phase_arr == ph)
                    if not mask_arr.any():
                        continue

                    # bridge: include the first point AFTER each run of this phase
                    # so the segment line visually ends at the phase boundary
                    bridge_arr = (~mask_arr) & np.concatenate(([False], mask_arr[:-1]))
                    combined = mask_arr | bridge_arr

                    # Build compact float64 segments with NaN separators.
                    # For 338k rows, this sends only the ~N/5 real phase points instead of
                    # a 338k-element object array — much faster JSON serialisation in Plotly.
                    transitions = np.diff(combined.astype(np.int8), prepend=0, append=0)
                    run_starts = np.where(transitions == 1)[0]
                    run_ends   = np.where(transitions == -1)[0]

                    parts_x: list[np.ndarray] = []
                    parts_y: list[np.ndarray] = []
                    nan1 = np.array([np.nan])
                    for s, e in zip(run_starts, run_ends):
                        if parts_x:
                            parts_x.append(nan1)
                            parts_y.append(nan1)
                        parts_x.append(x_arr[s:e])
                        parts_y.append(y_arr[s:e])

                    if not parts_x:
                        continue

                    x_ph = np.concatenate(parts_x)
                    y_ph = np.concatenate(parts_y)

                    # show_phase always uses phase colors regardless of track_multicolor
                    line_color = PHASE_COLORS[ph]
                    fig.add_trace(Trace(
                        x=x_ph,
                        y=y_ph,
                        mode="lines",
                        name=model_name,
                        legendgroup=legend_group,
                        showlegend=not legend_shown,  # show legend only for first present phase
                        line={"color": line_color, "dash": dash, "width": width},
                    ))
                    legend_shown = True
            else:
                # normal single-color line
                line_spec = {"dash": dash, "width": width}
                if track_color is not None:
                    line_spec["color"] = track_color
                fig.add_trace(Trace(
                    x=x_plot, y=y_plot,
                    mode="lines", name=model_name,
                    line=line_spec,
                ))

            if not kipp_env_mode and "Age" in df.columns:
                age_full = pd.to_numeric(df["Age"], errors="coerce") * 1_000_000.0  #
                age_full = age_full.mask(age_full <= 0)

                age_max = age_full.max()
                if pd.notna(age_max) and float(age_max) > max_age_tracker[0]:
                    max_age_tracker[0] = float(age_max)

                age_for_valid = age_full.loc[valid.index]

                # Extract logg for isochrone sorting
                logg_for_valid = None
                if "logg" in df.columns:
                    logg_full = pd.to_numeric(df["logg"], errors="coerce")
                    logg_for_valid = logg_full.loc[valid.index]

                # Isochrone sort column (from config; fallback to logg)
                iso_sort_col = cfg.isochrone_sort_by
                # Case-insensitive column lookup
                _col_map = {c.lower(): c for c in df.columns}
                _resolved = _col_map.get(iso_sort_col.lower())
                if _resolved is not None:
                    iso_sort_full = pd.to_numeric(df[_resolved], errors="coerce")
                    iso_sort_for_valid = iso_sort_full.loc[valid.index]
                else:
                    iso_sort_for_valid = logg_for_valid  # fallback: logg

                age_markers_data.append({
                    'model_name': model_name,
                    'x_plot': x_plot.copy(),
                    'y_plot': y_plot.copy(),
                    'age_array': age_for_valid.copy(),
                    'logg_array': logg_for_valid.copy() if logg_for_valid is not None else None,
                    'iso_sort_array': iso_sort_for_valid.copy() if iso_sort_for_valid is not None else None,
                    'x_is_age': x_is_age,
                    'y_is_age': y_is_age
                })

        if xcol.lower() in ("teff", "logteff") and ycol.lower() in ("logl", "l", "log_l"):
            fig.update_yaxes(autorange="reversed")

        # Kiel — invert Y (log g decreases upward)
        if ycol and ycol.lower() in ("logg", "log_g", "g", "loggsurf", "logg_surf"):
            fig.update_yaxes(autorange="reversed")

        def _needs_power_exponent(vmin, vmax):
            """Return True when axis values are so small that Plotly tends to switch to SI prefixes (p, n, µ, m)."""
            if vmin is None or vmax is None:
                return False
            m = max(abs(float(vmin)), abs(float(vmax)))
            if not np.isfinite(m) or m == 0.0:
                return False
            # We only care about tiny magnitudes (milli and smaller).
            # Plotly usually starts SI-prefix formatting below ~1e-3.
            return m < 1e-3

        def _nice_dtick_linear(span: float, target_ticks: int = 7) -> float:
            """Pick a 'nice' linear tick step (1-2-5 * 10^k) close to span/target_ticks."""
            if span is None:
                return 1.0
            span = float(span)
            if (not np.isfinite(span)) or span <= 0.0:
                return 1.0

            raw = span / float(max(1, target_ticks))
            if raw <= 0 or (not np.isfinite(raw)):
                return 1.0

            exp = 10 ** np.floor(np.log10(raw))
            for m in (1.0, 2.0, 5.0, 10.0):
                step = m * exp
                if raw <= step:
                    return float(step)
            return float(10.0 * exp)

        def _set_log_major_ticks(fig, axis: str, vmin: float, vmax: float):
            """Force major log ticks as HTML '10^k' so tickfont.size matches other axes."""
            if vmin is None or vmax is None:
                return
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return
            lo, hi = (vmin, vmax) if vmin < vmax else (vmax, vmin)
            if lo <= 0:
                return

            k_min = int(np.floor(np.log10(lo)))
            k_max = int(np.ceil(np.log10(hi)))

            tickvals = [10 ** k for k in range(k_min, k_max + 1)]
            ticktext = [f"10<sup>{k}</sup>" for k in range(k_min, k_max + 1)]

            if axis == "x":
                fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            else:
                fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

        x_all, y_all = [], []
        for tr in fig.data:
            if getattr(tr, "x", None) is not None:
                x_all.extend(tr.x)

        x_min, x_max = _finite_min_max(x_all)

        # compute Y min/max from independent pool to decouple from X filtering
        if y_axis_pool:
            y_min, y_max = _finite_min_max(y_axis_pool)
        else:
            # fallback: traces
            for tr in fig.data:
                if getattr(tr, "y", None) is not None:
                    y_all.extend(tr.y)
            y_min, y_max = _finite_min_max(y_all)

        phantom_x_min, phantom_x_max = x_min, x_max
        phantom_y_min, phantom_y_max = y_min, y_max

        x_scale = _axis_scale_factor(x_min, x_max) if (xscale_eff != "log" and not x_is_age) else None
        y_scale = _axis_scale_factor(y_min, y_max) if (yscale_eff != "log" and not y_is_age) else None

        # For Age with yr units, apply scaling to avoid mantissa < 1 (like 0.2×10^10)
        # For Myr, values are reasonable (~thousands) so no scaling needed
        if x_is_age and xscale_eff == "lin" and x_age_units == "yr":
            if x_max is not None and x_max > 0:
                # Find appropriate scale to keep mantissa >= 1
                exp = int(np.floor(np.log10(abs(x_max))))
                x_scale = 10.0 ** exp

        if y_is_age and yscale_eff == "lin" and y_age_units == "yr":
            if y_max is not None and y_max > 0:
                # Find appropriate scale to keep mantissa >= 1
                exp = int(np.floor(np.log10(abs(y_max))))
                y_scale = 10.0 ** exp

        # Apply scaling to all traces so axis ticks are in scaled units
        if x_scale is not None or y_scale is not None:
            for tr in fig.data:
                if x_scale is not None and getattr(tr, "x", None) is not None:
                    tr.x = np.asarray(tr.x, dtype=float) / x_scale
                if y_scale is not None and getattr(tr, "y", None) is not None:
                    tr.y = np.asarray(tr.y, dtype=float) / y_scale

        # Scale global ranges too (so padding/phantom points stay consistent with scaled traces)
        if x_scale is not None:
            if x_min is not None: x_min = x_min / x_scale
            if x_max is not None: x_max = x_max / x_scale
            if phantom_x_min is not None: phantom_x_min = phantom_x_min / x_scale
            if phantom_x_max is not None: phantom_x_max = phantom_x_max / x_scale

        if y_scale is not None:
            if y_min is not None: y_min = y_min / y_scale
            if y_max is not None: y_max = y_max / y_scale
            if phantom_y_min is not None: phantom_y_min = phantom_y_min / y_scale
            if phantom_y_max is not None: phantom_y_max = phantom_y_max / y_scale

        def _append_times10(title: str, exp: int) -> str:
            # If title is LaTeX ($...$), append LaTeX \times 10^{k}
            if isinstance(title, str) and title.startswith("$") and title.endswith("$"):
                return title[:-1] + rf"\times 10^{{{exp}}}$"
            # Otherwise (plain/HTML title) use HTML superscript
            return f"{title} ×10<sup>{exp}</sup>"

        # Update axis titles with ×10^k (once) and make tick labels "normal"
        x_title = _axis_title(xcol, x_is_age, xscale_eff, x_age_units if (x_is_age and xscale_eff == "lin") else None,
                              abund_fmt=x_abund_fmt if x_is_abund else None)
        y_title = _axis_title(ycol, y_is_age, yscale_eff, y_age_units if (y_is_age and yscale_eff == "lin") else None,
                              abund_fmt=y_abund_fmt if y_is_abund else None)

        if x_scale is not None:
            exp = int(np.log10(x_scale))
            x_title = _append_times10(x_title, exp)
            fig.update_yaxes(tickformat="")

        if y_scale is not None:
            exp = int(np.log10(y_scale))
            y_title = _append_times10(y_title, exp)
            fig.update_yaxes(tickformat="")

        if kipp_env_mode:
            _kipp_unit_label = {"gyr": "Gyr", "myr": "Myr", "yr": "yr"}.get(x_age_units, "Gyr")
            _kipp_x_title = r"$\mathrm{Age},\ \mathrm{" + _kipp_unit_label + r"}$"
            _kipp_y_title = (r"$\mathrm{r},\ \mathrm{R}_{\odot}$" if kipp_radius_mode
                             else r"$\mathrm{m},\ \mathrm{M}_{\odot}$")
            fig.update_layout(
                xaxis_title=_kipp_x_title,
                yaxis_title=_kipp_y_title,
            )
        else:
            fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

        # Universal margin helper: fixed % of the axis window width
        def _add_margin(val_min, val_max, scale_type, margin_fraction=0.02):
            """
            Adds equal visual margin regardless of scale type.

            For scale_type == "log":
              - Plotly renders axis in log scale
              - Data are NOT logarithmized (raw values)
              - Window width in log-space: log10(max) - log10(min)
              - Margin applied in log-space

            For scale_type == "log(x)":
              - Plotly renders axis linearly
              - Data are ALREADY logarithmized by _apply_axis_scale
              - Window width: max - min (values already log10)
              - Linear margin applied

            For scale_type == "lin":
              - Linear scale, linear margin
            """
            if scale_type == "log":
                # Log scale, data NOT logarithmized
                # Work in log-space for uniform visual offset
                tiny = float(np.nextafter(0.0, 1.0))
                v_min = max(val_min, tiny)
                v_max = max(val_max, v_min)

                log_min = np.log10(v_min)
                log_max = np.log10(v_max)
                log_width = log_max - log_min
                margin = margin_fraction * log_width

                new_min = 10 ** (log_min - margin)
                new_max = 10 ** (log_max + margin)
                return new_min, new_max
            else:
                # Linear scale (including log(x), where data are already logarithmized)
                width = val_max - val_min
                margin = margin_fraction * width
                new_min = val_min - margin
                new_max = val_max + margin
                return new_min, new_max

        if (
                x_min is not None and x_max is not None and x_min != x_max
                and y_min is not None and y_max is not None and y_min != y_max
        ):
            # X-axis
            if kipp_env_mode:
                phantom_x_min = 0.0
                phantom_x_max = x_max
            else:
                x_margin = 0.0 if frozen_has_kipp else 0.02
                phantom_x_min, phantom_x_max = _add_margin(x_min, x_max, xscale_eff, x_margin)

            # Y-axis
            if kipp_env_mode:
                phantom_y_min = 0.0
                phantom_y_max = y_max * 1.1
            else:
                phantom_y_min, phantom_y_max = _add_margin(y_min, y_max, yscale_eff)

            fig.add_trace(
                go.Scatter(
                    x=[phantom_x_min, phantom_x_max],
                    y=[phantom_y_min, phantom_y_max],
                    mode="markers",
                    marker=dict(size=0),
                    opacity=0.0,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        x_zoom = _get_zoom_range(relayout, "xaxis")
        y_zoom = _get_zoom_range(relayout, "yaxis")

        x_rev = (xcol and xcol.lower() in ("logteff", "teff"))
        y_rev = (ycol and ycol.lower() in ("logg", "log_g", "g", "loggsurf", "logg_surf"))

        if x_zoom is not None:
            xaxis_kwargs = dict(range=x_zoom, autorange=False)
            if xscale_eff == "log":
                xaxis_kwargs.update(type="log", exponentformat="power")

                # Add tick generation for small ranges in log scale
                r0, r1 = x_zoom[0], x_zoom[1]
                if x_rev:
                    r0, r1 = r1, r0

                tickvals, ticktext = _make_log_ticks(r0, r1, cfg.log_tick_multipliers)
                if tickvals:
                    xaxis_kwargs.update(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            else:
                if _needs_power_exponent(x_zoom[0], x_zoom[1]):
                    xaxis_kwargs.update(exponentformat="power", showexponent="last")
            fig.update_xaxes(**xaxis_kwargs)
        else:
            if kipp_env_mode and x_max is not None:
                fig.update_xaxes(range=[0.0, float(x_max)], autorange=False)
            elif x_is_age and phantom_x_min is not None and phantom_x_max is not None:
                # For Age, check the selected scale mode
                if xscale_eff == "log":
                    # Mode "log": Plotly renders log axis, data are NOT logarithmized
                    tiny = float(np.nextafter(0.0, 1.0))
                    x0 = max(phantom_x_min, tiny)
                    x1 = max(phantom_x_max, x0)
                    r0 = float(np.log10(x0))
                    r1 = float(np.log10(x1))
                    if x_rev:
                        r0, r1 = r1, r0

                    range_magnitude = abs(r1 - r0)

                    if range_magnitude < 1.0:
                        # Narrow range within one order of magnitude
                        # Use linear-style ticks but on log scale
                        tickvals, ticktext = _make_log_ticks(r0, r1, cfg.log_tick_multipliers)
                        fig.update_xaxes(
                            type="log",
                            exponentformat="power",
                            range=[r0, r1],
                            autorange=False,
                            tickmode="array",
                            tickvals=tickvals,
                            ticktext=ticktext,
                        )
                    else:
                        tickvals, ticktext = _make_log_ticks(r0, r1, cfg.log_tick_multipliers)
                        fig.update_xaxes(
                            type="log",
                            range=[r0, r1],
                            autorange=False,
                            tickmode="array",
                            tickvals=tickvals,
                            ticktext=ticktext,
                        )
                elif xscale_eff == "log(x)":
                    # Mode "log(x)": data ALREADY logarithmized, Plotly renders linearly
                    # phantom_x_min/max are already in log form, use as-is
                    r0, r1 = phantom_x_min, phantom_x_max
                    if x_rev:
                        r0, r1 = r1, r0

                    fig.update_xaxes(
                        range=[r0, r1],
                        autorange=False,
                    )
                else:
                    r0, r1 = phantom_x_min, phantom_x_max
                    if x_rev:
                        r0, r1 = r1, r0

                    fig.update_xaxes(
                        range=[r0, r1],
                        autorange=False,
                    )
            elif phantom_x_min is not None and phantom_x_max is not None and phantom_x_min != phantom_x_max:
                if xscale_eff == "log":
                    tiny = float(np.nextafter(0.0, 1.0))
                    x0 = max(float(phantom_x_min), tiny)
                    x1 = max(float(phantom_x_max), x0)
                    r0 = float(np.log10(x0))
                    r1 = float(np.log10(x1))
                    if x_rev:
                        r0, r1 = r1, r0

                    range_magnitude = abs(r1 - r0)

                    tickvals, ticktext = _make_log_ticks(r0, r1, cfg.log_tick_multipliers)
                    xupd = dict(
                        type="log",
                        range=[r0, r1],
                        autorange=False,
                        tickmode="array",
                        tickvals=tickvals,
                        ticktext=ticktext,
                    )
                    if range_magnitude < 1.0:
                        xupd["exponentformat"] = "power"
                    fig.update_xaxes(**xupd)
                else:
                    r0, r1 = phantom_x_min, phantom_x_max
                    if x_rev:
                        r0, r1 = r1, r0
                    xaxis_kwargs = dict(range=[r0, r1], autorange=False)
                    if (not x_is_age) and _needs_power_exponent(phantom_x_min, phantom_x_max):
                        xaxis_kwargs.update(exponentformat="power", showexponent="last")
                    fig.update_xaxes(**xaxis_kwargs)

        # Note: Removed special tick handling for Teff when Y is log
        # Let Plotly handle X-axis ticks automatically for consistency

        # Y-axis
        if y_zoom is not None:
            yaxis_kwargs = dict(range=y_zoom, autorange=False)
            if yscale_eff == "log":
                yaxis_kwargs.update(type="log", exponentformat="power")

                # Add tick generation for small ranges in log scale
                r0, r1 = y_zoom[0], y_zoom[1]
                if y_rev:
                    r0, r1 = r1, r0

                tickvals, ticktext = _make_log_ticks(r0, r1, cfg.log_tick_multipliers)
                if tickvals:
                    yaxis_kwargs.update(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            else:
                if _needs_power_exponent(y_zoom[0], y_zoom[1]):
                    yaxis_kwargs.update(exponentformat="power", showexponent="last")
            fig.update_yaxes(**yaxis_kwargs)
        else:
            if kipp_radius_mode and yscale_eff == "log":
                # Log scale cannot include zero (envelope base is 0 when absent),
                # so a plain [0, y_max*1.1] window is impossible -- normally we'd
                # just let Plotly autorange. But per the user's request the view
                # should be cropped at the bottom at 1e-2 R_sun (smaller radii are
                # astrophysically negligible clutter at the star's centre). Splice
                # that crop into Plotly's own autorange result by reproducing its
                # log-axis padding formula (empirically: a fixed span/18 on each
                # side) from the floor/max collected while building the traces,
                # then raising only the lower edge -- the natural top edge (and
                # thus the rest of the view) is left exactly as autorange would
                # have produced it.
                _LOG_CLIP_MIN = 1e-2
                if (kipp_y_floor_global is not None and kipp_y_max_global is not None
                        and kipp_y_max_global > kipp_y_floor_global > 0):
                    _lo = np.log10(kipp_y_floor_global)
                    _hi = np.log10(kipp_y_max_global)
                    _pad = (_hi - _lo) / 18.0
                    r0 = max(_lo - _pad, np.log10(_LOG_CLIP_MIN))
                    r1 = _hi + _pad
                    fig.update_yaxes(type="log", range=[r0, r1], autorange=False, exponentformat="power")
                else:
                    fig.update_yaxes(type="log", autorange=True, exponentformat="power")
            elif kipp_env_mode and y_max is not None:
                fig.update_yaxes(range=[0.0, float(y_max) * 1.1], autorange=False)
            elif y_is_age and phantom_y_min is not None and phantom_y_max is not None:
                # For Age, check the selected scale mode
                if yscale_eff == "log":
                    # Mode "log": Plotly renders log axis, data are NOT logarithmized
                    tiny = float(np.nextafter(0.0, 1.0))
                    y0 = max(phantom_y_min, tiny)
                    y1 = max(phantom_y_max, y0)
                    r0 = float(np.log10(y0))
                    r1 = float(np.log10(y1))
                    if y_rev:
                        r0, r1 = r1, r0

                    range_magnitude = abs(r1 - r0)

                    if range_magnitude < 1.0:
                        # Narrow range within one order of magnitude
                        # Use linear-style ticks but on log scale
                        tickvals, ticktext = _make_log_ticks(r0, r1, cfg.log_tick_multipliers)

                        fig.update_yaxes(
                            type="log",
                            exponentformat="power",
                            range=[r0, r1],
                            autorange=False,
                            tickmode="array",
                            tickvals=tickvals,
                            ticktext=ticktext,
                        )
                    else:
                        # Wide range — ticks at 1, 2, 5
                        tickvals, ticktext = _make_log_ticks(r0, r1, cfg.log_tick_multipliers)

                        fig.update_yaxes(
                            type="log",
                            range=[r0, r1],
                            autorange=False,
                            tickmode="array",
                            tickvals=tickvals,
                            ticktext=ticktext,
                        )
            elif phantom_y_min is not None and phantom_y_max is not None and phantom_y_min != phantom_y_max:
                if yscale_eff == "log":
                    tiny = float(np.nextafter(0.0, 1.0))
                    y0 = max(float(phantom_y_min), tiny)
                    y1 = max(float(phantom_y_max), y0)
                    r0 = float(np.log10(y0))
                    r1 = float(np.log10(y1))
                    if y_rev:
                        r0, r1 = r1, r0

                    range_magnitude = abs(r1 - r0)

                    if range_magnitude < 1.0:
                        # Narrow range within one order of magnitude
                        # Use linear-style ticks but on log scale
                        val_min = 10 ** min(r0, r1)
                        val_max = 10 ** max(r0, r1)

                        # Generate nice linear ticks
                        span = val_max - val_min
                        base_order = int(np.floor(np.log10(val_min)))
                        base = 10 ** base_order

                        # Choose step size based on span
                        if span <= base * 2:
                            step = base * 0.5
                        elif span <= base * 5:
                            step = base
                        else:
                            step = base * 2

                        tickvals = []
                        current = np.ceil(val_min / step) * step
                        while current <= val_max:
                            if current >= val_min:
                                tickvals.append(float(current))
                            current += step

                        # Ensure at least 3 ticks
                        if len(tickvals) < 3:
                            step = step / 2
                            tickvals = []
                            current = np.ceil(val_min / step) * step
                            while current <= val_max:
                                if current >= val_min:
                                    tickvals.append(float(current))
                                current += step

                        # Format tick labels as plain numbers
                        ticktext = [f"{int(v)}" if v >= 1 and v == int(v) else f"{v:.4g}" for v in tickvals]

                        fig.update_yaxes(
                            type="log",
                            exponentformat="power",
                            range=[r0, r1],
                            autorange=False,
                            tickmode="array",
                            tickvals=tickvals,
                            ticktext=ticktext,
                        )
                    else:
                        # Wide range — ticks at 1, 2, 5
                        log_min = min(r0, r1)
                        log_max = max(r0, r1)

                        tickvals = []
                        ticktext = []
                        start_order = int(np.floor(log_min))
                        end_order = int(np.ceil(log_max))

                        for order in range(start_order, end_order + 1):
                            base = 10 ** order
                            for mult in cfg.log_tick_multipliers:
                                val = mult * base
                                log_val = np.log10(val)
                                if log_min <= log_val <= log_max:
                                    tickvals.append(val)
                                    if mult == 1:
                                        ticktext.append(f"10<sup>{order}</sup>")
                                    else:
                                        _m = int(mult) if mult == int(mult) else f"{mult:.4g}"
                                        ticktext.append(f"<span style='font-size: 0.75em'>{_m}</span>")

                        fig.update_yaxes(
                            type="log",
                            range=[r0, r1],
                            autorange=False,
                            tickmode="array",
                            tickvals=tickvals,
                            ticktext=ticktext,
                        )
                else:
                    r0, r1 = phantom_y_min, phantom_y_max
                    if y_rev:
                        r0, r1 = r1, r0
                    yaxis_kwargs = dict(range=[r0, r1], autorange=False)
                    if (not y_is_age) and _needs_power_exponent(phantom_y_min, phantom_y_max):
                        yaxis_kwargs.update(exponentformat="power", showexponent="last")
                    fig.update_yaxes(**yaxis_kwargs)

        fig.update_layout(uirevision=f"{xcol}|{ycol}|{xscale}|{yscale}|{kipp_coord}|{tuple(models or [])}")

        x_tick_size = cfg.axis_tick_font_size
        y_tick_size = cfg.axis_tick_font_size

        xaxis_dict = dict(
            tickfont=dict(size=x_tick_size, family="Times New Roman, STIX Two Text, serif"),
            title_font=dict(size=cfg.axis_title_font_size, family="Times New Roman, STIX Two Text, serif"),
            exponentformat="power",
            zeroline=False,
            showgrid=True,
            gridcolor=cfg.grid_rgba_str,
            gridwidth=1,
            showline=True,
            linecolor="#a6a6a6",
            linewidth=1.5,
            mirror=True,
            layer="above traces",
            automargin=False,
        )

        if x_is_age and xscale_eff == "lin":
            xaxis_dict["exponentformat"] = "none"
            xaxis_dict["separatethousands"] = True

        yaxis_dict = dict(
            tickfont=dict(size=y_tick_size, family="Times New Roman, STIX Two Text, serif"),
            title_font=dict(size=cfg.axis_title_font_size, family="Times New Roman, STIX Two Text, serif"),
            exponentformat="power",
            zeroline=False,
            showgrid=True,
            gridcolor=cfg.grid_rgba_str,
            gridwidth=1,
            showline=True,
            linecolor="#a6a6a6",
            linewidth=1.5,
            mirror=True,
            layer="above traces",
            automargin=False,
        )

        # For linear Age axis: use thousands separators instead of exponents
        if y_is_age and yscale_eff == "lin":
            yaxis_dict["exponentformat"] = "none"
            yaxis_dict["separatethousands"] = True

        if xscale_eff == "log" and not x_is_age:
            if hasattr(fig.layout.xaxis, 'range') and fig.layout.xaxis.range:
                x_range = fig.layout.xaxis.range
            else:
                all_x = []
                for trace in fig.data:
                    if trace.x is not None:
                        all_x.extend([v for v in trace.x if v > 0])
                if all_x:
                    log_min_x = np.log10(min(all_x))
                    log_max_x = np.log10(max(all_x))
                else:
                    log_min_x, log_max_x = 0, 10
                x_range = [log_min_x, log_max_x]

            tickvals_x, ticktext_x = _make_log_ticks(min(x_range), max(x_range), cfg.log_tick_multipliers)
            xaxis_dict.update(dict(
                type="log",
                tickmode="array",
                tickvals=tickvals_x,
                ticktext=ticktext_x,
            ))

        if yscale_eff == "log" and not y_is_age:
            # Get Y-axis range
            if hasattr(fig.layout.yaxis, 'range') and fig.layout.yaxis.range:
                y_range = fig.layout.yaxis.range
            else:
                # Determine range from data
                all_y = []
                for trace in fig.data:
                    if trace.y is not None:
                        all_y.extend([v for v in trace.y if v > 0])
                if all_y:
                    log_min_y = np.log10(min(all_y))
                    log_max_y = np.log10(max(all_y))
                else:
                    log_min_y, log_max_y = 0, 10
                y_range = [log_min_y, log_max_y]

            tickvals_y, ticktext_y = _make_log_ticks(min(y_range), max(y_range), cfg.log_tick_multipliers)
            yaxis_dict.update(dict(
                type="log",
                tickmode="array",
                tickvals=tickvals_y,
                ticktext=ticktext_y,
            ))

        fig.update_layout(
            title=None,
            margin=dict(t=2, r=620, b=100, l=90),
            xaxis=xaxis_dict,
            yaxis=yaxis_dict,
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend_font=dict(size=cfg.legend_font_size),
            legend=dict(
                x=1.02,
                xanchor="left",
                y=1,
                yanchor="top",
                itemwidth=50,
                tracegroupgap=0,
                font=dict(
                    family="Times New Roman",
                    size=cfg.legend_font_size,
                )
            ),
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.8)',
                remove=['autoScale2d', 'lasso2d', 'select2d'],
            )
        )

        # Age-tracker dots: always emit exactly one trace per age_markers_data entry
        # (with empty x/y when the marker isn't shown for the current age) so the
        # trace layout is deterministic — this lets slider-only updates patch these
        # traces in place via Patch() without touching anything else in the figure.
        marker_start_idx = len(fig.data)
        if not kipp_env_mode:
            for marker_data in age_markers_data:
                marker_x, marker_y, is_at_max_age = _compute_age_marker_position(
                    marker_data, selected_age,
                    xscale_eff, yscale_eff,
                    x_age_factor, y_age_factor,
                    x_scale, y_scale,
                )
                marker_symbol = 'square' if is_at_max_age else 'circle'
                fig.add_trace(go.Scatter(
                    x=[marker_x] if marker_x is not None else [],
                    y=[marker_y] if marker_y is not None else [],
                    mode='markers',
                    marker=dict(
                        size=cfg.isochrone_marker_size,
                        color=cfg.isochrone_marker_color,
                        symbol=marker_symbol,
                        line=dict(width=cfg.isochrone_marker_border_width, color='black')
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                    name=marker_data['model_name'],
                ))
        marker_count = len(fig.data) - marker_start_idx

        # Draw isochrone if enabled and multiple models are selected
        show_isochrone = bool(show_isochrone_value)  # checklist: [] or ["on"]
        # Isochrone overlay: always emit the trace whenever it's eligible to exist
        # (>=2 models picked and the checkbox is on), with empty x/y when there
        # aren't enough points for the current age — same determinism rationale
        # as the age-tracker dots above, so Patch() can address it reliably.
        isochrone_idx = None
        if show_isochrone and models and len(models) >= 2:
            isochrone_points = []
            for marker_data in age_markers_data:
                point = _compute_isochrone_point(
                    marker_data, selected_age,
                    xscale_eff, yscale_eff,
                    x_age_factor, y_age_factor,
                    x_scale, y_scale,
                )
                if point is not None:
                    isochrone_points.append(point)

            iso_x, iso_y = [], []
            if len(isochrone_points) >= 2:
                # Sort by configured column (index 3 = iso_sort_val); fallback to x
                asc = cfg.isochrone_sort_ascending
                if all(p[3] is not None and pd.notna(p[3]) for p in isochrone_points):
                    isochrone_points.sort(key=lambda p: p[3], reverse=(not asc))
                else:
                    isochrone_points.sort(key=lambda p: p[0], reverse=(not asc))

                iso_x = [p[0] for p in isochrone_points]
                iso_y = [p[1] for p in isochrone_points]

            isochrone_idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=iso_x,
                y=iso_y,
                mode='lines',
                line=dict(color=cfg.isochrone_line_color, width=cfg.isochrone_line_width),
                showlegend=False,
                hoverinfo='skip',
                name='Isochrone'
            ))

        # Snapshot everything the fast age-slider Patch() path needs to recompute
        # marker/isochrone positions and address the right traces, without redoing
        # any of the data loading / track building above.
        AGE_MARKER_CACHE.clear()
        AGE_MARKER_CACHE.update({
            'age_markers_data': age_markers_data,
            'marker_start_idx': marker_start_idx,
            'marker_count': marker_count,
            'isochrone_idx': isochrone_idx,
            'kipp_env_mode': kipp_env_mode,
            'xscale_eff': xscale_eff,
            'yscale_eff': yscale_eff,
            'x_age_factor': x_age_factor,
            'y_age_factor': y_age_factor,
            'x_scale': x_scale,
            'y_scale': y_scale,
            'max_age_among_models': max_age_tracker[0],
        })

        status = ""
        # Interval should run ONLY while something is loading in background
        interval_disabled = True

        use_fast_mode = "fast" in (fast_hr_mode or [])

        if use_fast_mode:
            # In fast mode, we never load background data
            status = "Fast mode: only .hr files loaded"
            interval_disabled = True
        elif total > 0 and (loaded < total or loading > 0):
            status = f"Loading full model data in background: {loaded}/{total} ready"
            interval_disabled = False
        elif total > 0:
            status = "Full model data loaded"
            interval_disabled = True

        max_age_among_models = max_age_tracker[0]
        slider_max = max(max_age_among_models, 1e7)  #
        slider_marks = {
            0: '0',
            slider_max / 4: f'{(slider_max / 4) / 1e6:.1f}',
            slider_max / 2: f'{(slider_max / 2) / 1e6:.1f}',
            3 * slider_max / 4: f'{(3 * slider_max / 4) / 1e6:.1f}',
            slider_max: f'{slider_max / 1e6:.1f}'
        }

        corrected_age_input = selected_age / 1e6
        if selected_age > max_age_among_models:
            corrected_age_input = max_age_among_models / 1e6

        # Record that these models were opened (for last_opened sort)
        if models:
            for _m_path in models:
                record_opened(_m_path)

        # --- overlay frozen PNG snapshots (composited JS-side into frozen-composite store) ---
        if frozen_composite:
            fig.update_layout(images=[dict(
                source=frozen_composite,
                xref="paper", yref="paper",
                x=0, y=1, sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                layer="below", sizing="stretch", opacity=1.0,
            )])

        return fig, status, interval_disabled, slider_max, slider_marks, corrected_age_input

    class _Server:
        def run(self, host="127.0.0.1", port=8050, debug=False):
            app.run(host=host, port=port, debug=debug)

    return _Server(), f"http://127.0.0.1:{port}"