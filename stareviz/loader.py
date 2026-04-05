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


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if str(path).endswith(".gz") \
        else open(path, "rt", encoding="utf-8", errors="ignore")

def _read_starevol_table(path: Path) -> pd.DataFrame:
    with _open_text(path) as f:
        lines = f.readlines()

    # normalize exotic spaces (NBSP, narrow NBSP, figure space)
    lines = [ln.replace("\xa0", " ").replace("\u202f", " ").replace("\u2007", " ") for ln in lines]

    header_idx = None
    for i, line in enumerate(lines):
        # match 'model' with any leading spaces/tabs/non-breaking spaces
        if re.match(r"\s*model(\s|$)", line, flags=re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        return _read_any_table_fallback(path)

    header_cols = re.split(r"\s+", lines[header_idx].strip())

    dash_idx = None
    for j in range(header_idx + 1, min(header_idx + 6, len(lines))):
        if set(lines[j].strip()) <= set("-"):
            dash_idx = j
            break
    data_start = (dash_idx + 1) if dash_idx is not None else (header_idx + 1)

    data_lines = []
    need = len(header_cols)

    for ln in lines[data_start:]:
        ln = ln.replace("\xa0", " ").replace("\u202f", " ").replace("\u2007", " ")
        ln = re.sub(r"[@<\|]+", " ", ln)
        ln = re.sub(r"\s+", " ", ln.strip())
        if not ln:
            continue

        parts = ln.split(" ")
        if len(parts) < need:
            continue
        parts = parts[:need]
        data_lines.append(" ".join(parts))

    buf = StringIO("\n".join(data_lines))

    df_raw = pd.read_csv(
        buf,
        sep=r"\s+",
        header=None,
        engine="python",
    ).dropna(axis=1, how="all")

    need = len(header_cols)
    if df_raw.shape[1] < need:
        return _read_any_table_fallback(path)

    df = df_raw.iloc[:, :need].copy()
    df.columns = header_cols

    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str)
            s = s.str.replace(r"[@<\|]+$", "", regex=True)  # drop trailing markers
            s = s.str.replace(r"[dD]([+\-]?\d+)", r"E\1", regex=True)  # FORTRAN exponent D→E
            s = s.str.replace(r"[^0-9\.\+\-eE]", "", regex=True)  # keep numeric tokens
            df[c] = pd.to_numeric(s, errors="coerce")

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


def _collect_files(model_dir: Path) -> list[Path]:
    return list(model_dir.glob("*"))


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


def load_one_model(model_dir: Path, verbose: bool = False) -> pd.DataFrame:
    files = _collect_files(model_dir)
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
            print("⚠ v13 files NOT found!")
        print(f"{'=' * 80}\n")

    main_file = _pick_first(files, ["*.hr.gz", "*.hr", "*.track.gz", "*.track", "*.dat.gz", "*.dat", "*.csv"])
    if not main_file:
        main_file = sorted(files)[0]

    base = _read_any_table(main_file)
    base = _standardize_main(base).reset_index(drop=True)

    # Helper function to merge additional files by model number
    def merge_file(pattern: str, prefix: str = None, debug_name: str = None):
        nonlocal base
        f = _pick_first(files, pattern if isinstance(pattern, list) else [pattern])
        if not f:
            if verbose:
                print(f"  - {debug_name or pattern}: file not found")
            return

        try:
            df = _read_any_table(f)
        except Exception as e:
            if verbose:
                print(f"  - {debug_name or f.name}: read error - {e}")
            return

        if "model" not in df.columns:
            if verbose:
                print(f"  - {debug_name or f.name}: no 'model' column (columns: {list(df.columns)[:5]}...)")
            return

        df.columns = [c.strip() for c in df.columns]

        # Add prefix to columns (except 'model')
        if prefix:
            rename_map = {c: f"{prefix}{c}" for c in df.columns if c != "model"}
            df = df.rename(columns=rename_map)

        # Remove 'model' from df before merge to avoid duplicate
        cols_to_merge = [c for c in df.columns if c != "model"]
        if cols_to_merge:
            base = base.merge(df[["model"] + cols_to_merge], on="model", how="left")
            if verbose:
                print(f"  ✓ {debug_name or f.name}: added {len(cols_to_merge)} columns")
        elif verbose:
            print(f"  - {debug_name or f.name}: no columns to merge")

    # Load all v* files (v1-v13) - no prefix
    for i in range(1, 14):
        merge_file([f"*.v{i}.gz", f"*.v{i}", f"*v{i}.gz", f"*v{i}"], debug_name=f"v{i}")

    merge_file(["*.tc1.gz", "*.tc1", "*tc1.gz", "*tc1"])
    merge_file(["*.tc2.gz", "*.tc2", "*tc2.gz", "*tc2"])

    merge_file(["*.as.gz", "*.as"])

    # Load surface abundances (s1-s4) with 's_' prefix
    for i in range(1, 5):
        merge_file([f"*.s{i}.gz", f"*.s{i}"], prefix="s_")

    # Load central abundances (c1-c4) with 'c_' prefix
    for i in range(1, 5):
        merge_file([f"*.c{i}.gz", f"*.c{i}"], prefix="c_")

    # Atomic mass numbers for abundance calculation A(X) = log10(X/H * A_H) + 12
    # where A_H is the atomic mass of the element (e.g., 7 for Li7, 4 for He4, etc.)

    # Compute A(X) for surface abundances if s_H1 exists
    if "s_H1" in base.columns:
        H1_surf = pd.to_numeric(base["s_H1"], errors="coerce")

        for element, atomic_mass in ATOMIC_MASSES.items():
            col_name = f"s_{element}"
            if col_name in base.columns and element != "H1":  # Skip H1 itself
                X = pd.to_numeric(base[col_name], errors="coerce")
                m = (X > 0) & (H1_surf > 0)
                A_X = pd.Series(np.nan, index=base.index, dtype=float)
                A_X[m] = np.log10(X[m] / (H1_surf[m] * atomic_mass)) + 12.0
                base[f"A_{element}"] = A_X

    # Compute pseudo A(heavy) if heavy mass fraction exists.
    # NOTE: "heavy" is an aggregate of all elements heavier than Cl37, so there is no unique atomic mass.
    # We use a representative atomic mass (56, iron-like) to get a consistent diagnostic quantity for plotting/debugging.
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
            if col_name in base.columns and element != "H1":  # Skip H1 itself
                X = pd.to_numeric(base[col_name], errors="coerce")
                m = (X > 0) & (H1_cent > 0)
                A_X = pd.Series(np.nan, index=base.index, dtype=float)
                A_X[m] = np.log10(X[m] / (H1_cent[m] * atomic_mass)) + 12.0
                base[f"Ac_{element}"] = A_X

        # Pseudo A(heavy) for central heavy mass fraction (see note above).
        if "c_heavy" in base.columns:
            Xh = pd.to_numeric(base["c_heavy"], errors="coerce")
            m = (Xh > 0) & (H1_cent > 0)
            A_h = pd.Series(np.nan, index=base.index, dtype=float)
            A_h[m] = np.log10(Xh[m] / (H1_cent[m] * 56.0)) + 12.0
            base["Ac_heavy"] = A_h

    # Rename A_Li7 to A_Li for backward compatibility (and keep A_Li7 as well)
    if "A_Li7" in base.columns:
        base["A_Li"] = base["A_Li7"]

    # Rename omega_S or omegas to Omega_surf for consistency
    if "omegas" in base.columns and "Omega_surf" not in base.columns:
        base = base.rename(columns={"omegas": "Omega_surf"})
    elif "omega_S" in base.columns and "Omega_surf" not in base.columns:
        base = base.rename(columns={"omega_S": "Omega_surf"})

    if "Age" in base.columns:
        base = base.sort_values(["Age", "model"]).reset_index(drop=True)
    elif "model" in base.columns:
        base = base.sort_values("model").reset_index(drop=True)

    return base


def load_hr_only(model_dir: Path) -> pd.DataFrame:
    """Fast-path loader: read only the main *.hr file (or closest equivalent).

    Returns a standardized dataframe, but WITHOUT merging auxiliary files.
    Designed for quick HR-diagram rendering.
    """
    files = _collect_files(model_dir)
    if not files:
        raise FileNotFoundError(f"No files in {model_dir}")

    main_file = _pick_first(files, ["*.hr.gz", "*.hr", "*.track.gz", "*.track", "*.dat.gz", "*.dat", "*.csv"])
    if not main_file:
        main_file = sorted(files)[0]

    base = _read_any_table(main_file)
    base = _standardize_main(base).reset_index(drop=True)
    return base