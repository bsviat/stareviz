# stio.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
import numpy as np
import gzip
from io import StringIO
import re

from stareviz.constants import ATOMIC_MASSES

CANDIDATES = {
    "age": ["age", "age", "t_myr", "time_myr", "t", "time"],
    "teff": ["teff", "t_eff", "teff_k", "effective_temperature"],
    "logteff": ["logteff", "log_teff", "log_t_eff"],
    "l": ["l", "lum", "luminosity", "l_lsun", "l/lsun"],
    "logl": ["logl", "log_l", "log_lum"],
    "logg": ["logg", "log_g"],
    "li_surf": ["a(li)", "a_li", "li", "li_surf", "li_surface", "li7_surf", "ali"],
    "omega_surf": ["omega_surf", "omega", "omega_surface", "surf_omega", "omega_s", "omega_eq"],
}


# Compiled once at module load — reused for every file read
_RE_EXOTIC_SPACES = re.compile(r"[\xa0\u202f\u2007]")
_RE_MARKERS    = re.compile(r"[@<\|]+")
_RE_FORTRAN_D  = re.compile(r"[dD]([+\-]?\d+)")
_RE_LEADING_WS = re.compile(r"^[ \t]+", re.MULTILINE)
_RE_MULTI_WS   = re.compile(r"[ \t]+")


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if str(path).endswith(".gz") \
        else open(path, "rt", encoding="utf-8", errors="ignore")



def _read_starevol_table(path: Path) -> pd.DataFrame:
    with _open_text(path) as f:
        text = f.read()

    # Single-pass exotic space normalization via C regex (releases GIL, unlike str.replace chains)
    text = _RE_EXOTIC_SPACES.sub(" ", text)

    # Find header by scanning only the first 100 lines with str.find() — avoids splitlines()
    # on the full file (splitlines creates 338k Python string objects and holds the GIL).
    pos = 0
    header_idx = None
    header_cols = None
    for i in range(100):
        nl = text.find("\n", pos)
        line = text[pos:nl] if nl != -1 else text[pos:]
        if re.match(r"\s*model(\s|$)", line, re.IGNORECASE):
            header_idx = i
            header_cols = re.split(r"\s+", line.strip())
            pos = nl + 1 if nl != -1 else len(text)
            break
        pos = (nl + 1) if nl != -1 else len(text)

    if header_idx is None:
        return _read_any_table_fallback(path)

    need = len(header_cols)

    # Find optional dash separator in the next few lines
    for _ in range(5):
        nl = text.find("\n", pos)
        line = text[pos:nl] if nl != -1 else text[pos:]
        stripped = line.strip()
        if stripped and set(stripped) <= {"-"}:
            pos = (nl + 1) if nl != -1 else len(text)
            break
        pos = (nl + 1) if nl != -1 else len(text)

    # Slice data block directly — one O(n) copy instead of splitlines()+join() which
    # creates and GC-s 338k intermediate string objects while holding the GIL.
    data_text = text[pos:]

    # Bulk transforms: all four regex subs release the GIL during C-level processing,
    # enabling true parallelism when multiple files are read in a thread pool.
    data_text = _RE_MARKERS.sub(" ", data_text)
    data_text = _RE_FORTRAN_D.sub(r"E\1", data_text)
    data_text = _RE_LEADING_WS.sub("", data_text)
    data_text = _RE_MULTI_WS.sub(" ", data_text)

    buf = StringIO(data_text)
    try:
        df_raw = pd.read_csv(
            buf,
            sep=" ",
            header=None,
            engine="c",
            usecols=list(range(need)),
            on_bad_lines="skip",
        ).dropna(axis=1, how="all")
    except Exception:
        buf = StringIO(data_text)
        try:
            df_raw = pd.read_csv(
                buf,
                sep=r"\s+",
                header=None,
                engine="python",
                on_bad_lines="skip",
            ).dropna(axis=1, how="all")
        except Exception:
            return _read_any_table_fallback(path)

    if df_raw.shape[1] < need:
        return _read_any_table_fallback(path)

    df = df_raw.iloc[:, :need].copy()
    df.columns = header_cols

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "model" in df.columns:
        df = df[df["model"].notna()]
        m1 = df.index[(df["model"].astype("Int64") == 1)]
        if len(m1) > 0:
            df = df.loc[m1[0]:].reset_index(drop=True)

    return df


def _norm(c: str) -> str:
    return c.lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def _find_col(df: pd.DataFrame, keys: Iterable[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for key in keys:
        if key in norm_map:
            return norm_map[key]
    return None


def _read_any_table_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, comment="#", sep=r"\s+", compression="infer", engine="python")
    except Exception:
        return pd.read_csv(path, comment="#", compression="infer")


def _read_any_table(path: Path) -> pd.DataFrame:
    name = path.name

    if re.search(r"(evolvar\d+|(\.|)(hr|as|tc\d+|s\d+|c\d+|v\d+))(\.gz)?$", name):
        return _read_starevol_table(path)

    return _read_any_table_fallback(path)


def _pick_first(paths: list[Path], patterns: list[str]) -> Optional[Path]:
    from fnmatch import fnmatch
    for pat in patterns:
        m = [p for p in paths if fnmatch(p.name, pat)]
        if m:
            m = sorted(m, key=lambda x: x.name)
            return m[-1]
    return None


def _collect_files(model_dir: Path, stem: Optional[str] = None) -> list[Path]:
    files = list(model_dir.glob("*"))
    if stem is not None:
        files = [f for f in files if f.name.startswith(stem + ".")]
    return files


def _merge_on_age(base: pd.DataFrame, extra: pd.DataFrame, tol_myr: float = 0.05) -> pd.DataFrame:
    if "Age" not in base.columns or "Age" not in extra.columns:
        return base

    q = max(tol_myr, 1e-6)
    b = base.copy()
    e = extra.copy()
    b["__ageq"] = (b["Age"] / q).round().astype(int)
    e["__ageq"] = (e["Age"] / q).round().astype(int)
    out = pd.merge(b, e.drop_duplicates("__ageq"), on="__ageq", how="left", suffixes=("", "_ext"))
    return out.drop(columns="__ageq")


def _standardize_main(df: pd.DataFrame) -> pd.DataFrame:
    if "model" in df.columns:
        df["model"] = pd.to_numeric(df["model"], errors="coerce")

    # age: t — in years → Myr
    if "Age" not in df.columns:
        if "age_yr" in df.columns:
            df["Age"] = pd.to_numeric(df["age_yr"], errors="coerce") / 1e6
        elif "t" in df.columns:
            df["Age"] = pd.to_numeric(df["t"], errors="coerce") / 1e6

    if "Teff" in df.columns and "logTeff" not in df.columns:
        df["logTeff"] = np.log10(pd.to_numeric(df["Teff"], errors="coerce"))
    if "logL" in df.columns and "L" not in df.columns:
        df["L"] = 10 ** pd.to_numeric(df["logL"], errors="coerce")
    if "logg" not in df.columns and "geff" in df.columns:
        df["logg"] = df["geff"]

    if "logg" in df.columns and "g" not in df.columns:
        df["g"] = 10 ** pd.to_numeric(df["logg"], errors="coerce")

    return df


def _extract_scalar(df: pd.DataFrame, key: str, target_name: str) -> pd.DataFrame:
    col = _find_col(df, [_norm(x) for x in CANDIDATES[key]])
    if not col:
        return pd.DataFrame()
    out = df[[c for c in ("Age", col) if c in df.columns]].copy()
    if col != target_name:
        out = out.rename(columns={col: target_name})
    return out


def load_one_model(model_dir: Path, stem: Optional[str] = None, verbose: bool = False) -> pd.DataFrame:
    files = _collect_files(model_dir, stem)
    if not files:
        raise FileNotFoundError(f"No files in {model_dir}")

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Loading model from: {model_dir}")
        print(f"Total files: {len(files)}")
        v13_files = [f.name for f in files if 'v13' in f.name.lower()]
        if v13_files:
            print(f"v13 files: {v13_files}")
        else:
            print("\u26a0 v13 files NOT found!")
        print(f"{'=' * 80}\n")

    main_file = _pick_first(files, ["*.hr.gz", "*.hr", "*.track.gz", "*.track", "*.dat.gz", "*.dat", "*.csv"])
    if not main_file:
        main_file = sorted(files)[0]

    base = _standardize_main(_read_any_table(main_file)).reset_index(drop=True)

    aux_specs = (
        [([f"*.v{i}.gz", f"*.v{i}", f"*v{i}.gz", f"*v{i}"], None, f"v{i}") for i in range(1, 14)]
        + [
            (["*.tc1.gz", "*.tc1", "*tc1.gz", "*tc1"], None, "tc1"),
            (["*.tc2.gz", "*.tc2", "*tc2.gz", "*tc2"], None, "tc2"),
            (["*.as.gz", "*.as"], None, "as"),
        ]
        + [([f"*.s{i}.gz", f"*.s{i}"], "s_", f"s{i}") for i in range(1, 5)]
        + [([f"*.c{i}.gz", f"*.c{i}"], "c_", f"c{i}") for i in range(1, 5)]
    )

    for patterns, prefix, label in aux_specs:
        f = _pick_first(files, patterns)
        if not f:
            if verbose:
                print(f"  - {label}: file not found")
            continue
        try:
            df = _read_any_table(f)
        except Exception as e:
            if verbose:
                print(f"  - {label}: read error - {e}")
            continue
        if "model" not in df.columns:
            if verbose:
                print(f"  - {label}: no 'model' column (columns: {list(df.columns)[:5]}...)")
            continue
        df.columns = [c.strip() for c in df.columns]
        if prefix:
            df = df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != "model"})
        cols_to_merge = [c for c in df.columns if c != "model"]
        if not cols_to_merge:
            if verbose:
                print(f"  - {label}: no columns to merge")
            continue
        base = base.merge(df[["model"] + cols_to_merge], on="model", how="left")
        if verbose:
            print(f"  \u2713 {label}: added {len(cols_to_merge)} columns")


    # Compute A(X) for surface abundances if s_H1 exists
    if "s_H1" in base.columns:
        H1_surf = pd.to_numeric(base["s_H1"], errors="coerce")

        for element, atomic_mass in ATOMIC_MASSES.items():
            col_name = f"s_{element}"
            if col_name in base.columns and element != "H1":
                X = pd.to_numeric(base[col_name], errors="coerce")
                m = (X > 0) & (H1_surf > 0)
                A_X = pd.Series(np.nan, index=base.index, dtype=float)
                A_X[m] = np.log10(X[m] / (H1_surf[m] * atomic_mass)) + 12.0
                base[f"A_{element}"] = A_X

    if "s_heavy" in base.columns and "s_H1" in base.columns:
        H1_surf = pd.to_numeric(base["s_H1"], errors="coerce")
        Xh = pd.to_numeric(base["s_heavy"], errors="coerce")
        m = (Xh > 0) & (H1_surf > 0)
        A_h = pd.Series(np.nan, index=base.index, dtype=float)
        A_h[m] = np.log10(Xh[m] / (H1_surf[m] * 56.0)) + 12.0
        base["A_heavy"] = A_h

    # Compute A(X) for central abundances if c_H1 exists
    if "c_H1" in base.columns:
        H1_cent = pd.to_numeric(base["c_H1"], errors="coerce")

        for element, atomic_mass in ATOMIC_MASSES.items():
            col_name = f"c_{element}"
            if col_name in base.columns and element != "H1":
                X = pd.to_numeric(base[col_name], errors="coerce")
                m = (X > 0) & (H1_cent > 0)
                A_X = pd.Series(np.nan, index=base.index, dtype=float)
                A_X[m] = np.log10(X[m] / (H1_cent[m] * atomic_mass)) + 12.0
                base[f"Ac_{element}"] = A_X

        if "c_heavy" in base.columns:
            Xh = pd.to_numeric(base["c_heavy"], errors="coerce")
            m = (Xh > 0) & (H1_cent > 0)
            A_h = pd.Series(np.nan, index=base.index, dtype=float)
            A_h[m] = np.log10(Xh[m] / (H1_cent[m] * 56.0)) + 12.0
            base["Ac_heavy"] = A_h

    if "A_Li7" in base.columns:
        base["A_Li"] = base["A_Li7"]

    if "omegas" in base.columns and "Omega_surf" not in base.columns:
        base = base.rename(columns={"omegas": "Omega_surf"})
    elif "omega_S" in base.columns and "Omega_surf" not in base.columns:
        base = base.rename(columns={"omega_S": "Omega_surf"})

    if "Age" in base.columns:
        base = base.sort_values(["Age", "model"]).reset_index(drop=True)
    elif "model" in base.columns:
        base = base.sort_values("model").reset_index(drop=True)

    return base

def load_kipp_model(model_dir: Path, stem: Optional[str] = None) -> pd.DataFrame:
    """Load only the files needed for the Kippenhahn diagram: .hr + .v3 + .v4 + .v5-.v8 + .v12.

    .v3/.v4 carry the mass-coordinate boundaries (and the conv1/env radii as
    fractions of the stellar radius); .v12 carries the absolute radii (R_sun)
    for the conv2-conv6 zones, needed for the radius-coordinate diagram.
    .v5-.v8 carry the H/He/C/Ne nuclear-burning-zone boundaries (Xburn_Mb/Mt
    in M_sun, Xburn_Rb/Rt as fractions of the stellar radius), drawn as
    optional overlays controlled by cfg.kipp.show_*burn.

    ~3-4s instead of ~29s for large models. KIPP_CACHE should be evicted once
    FULL_CACHE is ready to avoid keeping a redundant copy in memory.
    """
    files = _collect_files(model_dir, stem)
    if not files:
        raise FileNotFoundError(f"No files in {model_dir}")

    main_file = _pick_first(files, ["*.hr.gz", "*.hr", "*.track.gz", "*.track", "*.dat.gz", "*.dat", "*.csv"])
    if not main_file:
        main_file = sorted(files)[0]

    base = _standardize_main(_read_any_table(main_file)).reset_index(drop=True)

    for patterns, label in [
        (["*.v3.gz", "*.v3", "*v3.gz", "*v3"], "v3"),
        (["*.v4.gz", "*.v4", "*v4.gz", "*v4"], "v4"),
        (["*.v5.gz", "*.v5", "*v5.gz", "*v5"], "v5"),
        (["*.v6.gz", "*.v6", "*v6.gz", "*v6"], "v6"),
        (["*.v7.gz", "*.v7", "*v7.gz", "*v7"], "v7"),
        (["*.v8.gz", "*.v8", "*v8.gz", "*v8"], "v8"),
        (["*.v12.gz", "*.v12", "*v12.gz", "*v12"], "v12"),
    ]:
        f = _pick_first(files, patterns)
        if not f:
            continue
        try:
            df = _read_any_table(f)
        except Exception:
            continue
        if "model" not in df.columns:
            continue
        df.columns = [c.strip() for c in df.columns]
        cols_to_merge = [c for c in df.columns if c != "model"]
        if cols_to_merge:
            base = base.merge(df[["model"] + cols_to_merge], on="model", how="left")

    if "Age" in base.columns:
        base = base.sort_values(["Age", "model"]).reset_index(drop=True)
    elif "model" in base.columns:
        base = base.sort_values("model").reset_index(drop=True)

    return base


def load_surf_model(model_dir: Path, stem: Optional[str] = None) -> pd.DataFrame:
    """Load .hr + .s1-.s4 for surface abundance plots.

    ~4-5s instead of ~29s for large models. Includes A(X) computation.
    SURF_CACHE should be evicted once FULL_CACHE is ready.
    """
    files = _collect_files(model_dir, stem)
    if not files:
        raise FileNotFoundError(f"No files in {model_dir}")

    main_file = _pick_first(files, ["*.hr.gz", "*.hr", "*.track.gz", "*.track", "*.dat.gz", "*.dat", "*.csv"])
    if not main_file:
        main_file = sorted(files)[0]

    base = _standardize_main(_read_any_table(main_file)).reset_index(drop=True)

    for i in range(1, 5):
        f = _pick_first(files, [f"*.s{i}.gz", f"*.s{i}"])
        if not f:
            continue
        try:
            df = _read_any_table(f)
        except Exception:
            continue
        if "model" not in df.columns:
            continue
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={c: f"s_{c}" for c in df.columns if c != "model"})
        cols_to_merge = [c for c in df.columns if c != "model"]
        if cols_to_merge:
            base = base.merge(df[["model"] + cols_to_merge], on="model", how="left")

    # Compute A(X) surface abundances — same logic as load_one_model
    if "s_H1" in base.columns:
        H1_surf = pd.to_numeric(base["s_H1"], errors="coerce")
        for element, atomic_mass in ATOMIC_MASSES.items():
            col_name = f"s_{element}"
            if col_name in base.columns and element != "H1":
                X = pd.to_numeric(base[col_name], errors="coerce")
                m = (X > 0) & (H1_surf > 0)
                A_X = pd.Series(np.nan, index=base.index, dtype=float)
                A_X[m] = np.log10(X[m] / (H1_surf[m] * atomic_mass)) + 12.0
                base[f"A_{element}"] = A_X

    if "s_heavy" in base.columns and "s_H1" in base.columns:
        H1_surf = pd.to_numeric(base["s_H1"], errors="coerce")
        Xh = pd.to_numeric(base["s_heavy"], errors="coerce")
        m = (Xh > 0) & (H1_surf > 0)
        A_h = pd.Series(np.nan, index=base.index, dtype=float)
        A_h[m] = np.log10(Xh[m] / (H1_surf[m] * 56.0)) + 12.0
        base["A_heavy"] = A_h

    if "A_Li7" in base.columns:
        base["A_Li"] = base["A_Li7"]

    if "Age" in base.columns:
        base = base.sort_values(["Age", "model"]).reset_index(drop=True)
    elif "model" in base.columns:
        base = base.sort_values("model").reset_index(drop=True)

    return base


def load_hr_only(model_dir: Path, stem: Optional[str] = None) -> pd.DataFrame:
    """Fast-path loader: read only the main *.hr file (or closest equivalent).

    Returns a standardized dataframe, but WITHOUT merging auxiliary files.
    Designed for quick HR-diagram rendering.
    """
    files = _collect_files(model_dir, stem)
    if not files:
        raise FileNotFoundError(f"No files in {model_dir}")

    main_file = _pick_first(files, ["*.hr.gz", "*.hr", "*.track.gz", "*.track", "*.dat.gz", "*.dat", "*.csv"])
    if not main_file:
        main_file = sorted(files)[0]

    base = _read_any_table(main_file)
    base = _standardize_main(base).reset_index(drop=True)
    return base