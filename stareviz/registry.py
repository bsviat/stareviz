from pathlib import Path
from functools import lru_cache
from typing import Optional
import json
import time
import pandas as pd
from stareviz.loader import load_one_model


def _find_hr_stems(dir_path: Path) -> list[str]:
    """Return sorted list of model stems found in dir_path via .hr/.hr.gz files."""
    seen: set[str] = set()
    for f in dir_path.glob("*.hr.gz"):
        seen.add(f.name[: -len(".hr.gz")])
    for f in dir_path.glob("*.hr"):
        stem = f.name[: -len(".hr")]
        seen.add(stem)
    return sorted(seen)


def parse_model_path(path: str) -> tuple[Path, Optional[str]]:
    """Parse 'folder:::stem' or plain folder path → (folder, stem_or_None)."""
    if ":::" in path:
        folder, stem = path.split(":::", 1)
        return Path(folder), stem
    return Path(path), None

# Path to the history file that tracks last-opened timestamps
HISTORY_FILE = Path.home() / ".config" / "stareviz_history.json"


def _load_history() -> dict:
    """Load {path: timestamp} from history file. Returns {} on any error."""
    try:
        if HISTORY_FILE.is_file():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def record_opened(path: str) -> None:
    """Record that a model at *path* was opened right now."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = _load_history()
        history[path] = time.time()
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        # Non-fatal — just warn
        import warnings
        warnings.warn(f"[stareviz] Could not write history file: {e}")


class ModelRegistry:
    def __init__(self, roots: list[str]):
        self.roots = [Path(r).resolve() for r in roots]
        self._index = None  # DataFrame: columns=[name, path]

    def build_index(self) -> pd.DataFrame:
        patterns = ("*.hr", "*.hr.gz", "*.v*", "*.v*.gz",
                    "*.s*", "*.s*.gz", "*.c*", "*.c*.gz",
                    "*.as", "*.as.gz", "*.tc*", "*.tc*.gz",
                    "*.track", "*.track.gz", "*.dat", "*.dat.gz", "*.csv")
        rows = []
        for root in self.roots:
            for p in root.rglob("*"):
                if not p.is_dir():
                    continue
                if not any(p.glob(pt) for pt in patterns):
                    continue
                stems = _find_hr_stems(p)
                if len(stems) > 1:
                    ctime = _get_ctime(p)
                    for stem in stems:
                        rows.append({
                            "name":        stem,
                            "path":        str(p) + ":::" + stem,
                            "folder_path": str(p),
                            "folder_name": p.name,
                            "stem":        stem,
                            "created":     ctime,
                        })
                else:
                    stem = stems[0] if stems else None
                    rows.append({
                        "name":        stem if stem else p.name,
                        "path":        (str(p) + ":::" + stem) if stem else str(p),
                        "folder_path": str(p),
                        "folder_name": p.name,
                        "stem":        stem,
                        "created":     _get_ctime(p),
                    })

        cols = ["name", "path", "folder_path", "folder_name", "stem", "created"]
        self._index = (pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols))
        if rows:
            self._index = self._index.drop_duplicates(subset=["path"]).reset_index(drop=True)

        self._apply_sort()
        return self._index

    def _apply_sort(self) -> None:
        """Sort self._index according to the current config setting."""
        from stareviz.viz_config import cfg
        order = getattr(cfg, "model_sort_order", "name")
        asc   = getattr(cfg, "model_sort_ascending", True)

        if order == "created":
            self._index = (self._index
                           .sort_values("created", ascending=asc)
                           .reset_index(drop=True))

        elif order == "last_opened":
            history = _load_history()
            self._index["_last_opened"] = (
                self._index["path"].map(lambda p: history.get(p, 0.0))
            )
            # Models never opened get 0 → fall to the bottom (always);
            # among those, sort alphabetically as a stable tiebreaker.
            # ascending applies only to the opened models.
            self._index = (self._index
                           .sort_values(["_last_opened", "name"],
                                        ascending=[asc, True])
                           .drop(columns=["_last_opened"])
                           .reset_index(drop=True))

        else:
            # name sort — keep multi-model groups contiguous:
            # primary key = folder_name (same for all stems of one folder),
            # secondary key = stem (within group) or '' (single-model items)
            is_multi = self._index["stem"].notna()
            self._index["_pk"] = self._index["folder_name"].where(is_multi, self._index["name"])
            self._index["_sk"] = self._index["stem"].fillna("")
            self._index = (self._index
                           .sort_values(["_pk", "_sk"], ascending=[asc, True])
                           .drop(columns=["_pk", "_sk"])
                           .reset_index(drop=True))

    def resort(self) -> None:
        """Re-apply sort without rebuilding the full index (e.g. after history update)."""
        if self._index is not None:
            self._apply_sort()

    def search(self, needle: str) -> pd.DataFrame:
        if self._index is None:
            self.build_index()
        m = (
            self._index["name"].str.contains(needle, case=False, regex=False) |
            self._index["folder_name"].str.contains(needle, case=False, regex=False)
        )
        return self._index[m].copy()

    @lru_cache(maxsize=512)
    def load_model(self, path: str) -> pd.DataFrame:
        folder, stem = parse_model_path(path)
        return load_one_model(folder, stem=stem)


def _get_ctime(p: Path) -> float:
    """Return creation time (or mtime on Linux where ctime = inode-change time)."""
    try:
        stat = p.stat()
        # st_birthtime is available on macOS; fall back to st_ctime elsewhere
        return getattr(stat, "st_birthtime", stat.st_ctime)
    except Exception:
        return 0.0