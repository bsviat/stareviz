from pathlib import Path
from functools import lru_cache
import json
import time
import pandas as pd
from stareviz.loader import load_one_model

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
                if any(p.glob(pt) for pt in patterns):
                    rows.append({
                        "name":    p.name,
                        "path":    str(p),
                        "created": _get_ctime(p),
                    })

        self._index = (pd.DataFrame(rows)
                       .drop_duplicates(subset=["path"])
                       .reset_index(drop=True))

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
            # name sort
            self._index = (self._index
                           .sort_values("name", ascending=asc)
                           .reset_index(drop=True))

    def resort(self) -> None:
        """Re-apply sort without rebuilding the full index (e.g. after history update)."""
        if self._index is not None:
            self._apply_sort()

    def search(self, needle: str) -> pd.DataFrame:
        if self._index is None:
            self.build_index()
        m = self._index["name"].str.contains(needle, case=False, regex=False)
        return self._index[m].copy()

    @lru_cache(maxsize=512)
    def load_model(self, path: str) -> pd.DataFrame:
        return load_one_model(Path(path))


def _get_ctime(p: Path) -> float:
    """Return creation time (or mtime on Linux where ctime = inode-change time)."""
    try:
        stat = p.stat()
        # st_birthtime is available on macOS; fall back to st_ctime elsewhere
        return getattr(stat, "st_birthtime", stat.st_ctime)
    except Exception:
        return 0.0