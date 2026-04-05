from typing import Optional
from .constants import (
    FORCED_AXIS_SCALE_PREFIXES,
    FORCED_AXIS_SCALE_KEYS,
    DEFAULT_AXIS_SCALE,
    AXIS_SCALE_CHOICES,
)

def get_forced_axis_scale(param: Optional[str]) -> Optional[str]:
    """
    Determines the forced axis scale for the given parameter.

    Returns "log(x)" if:
    - parameter starts with "A_" or "Ac_" (spectroscopic abundances)
    - parameter starts with "log" (e.g. logg, logL, logTeff)
    - parameter is in FORCED_AXIS_SCALE_KEYS

    Returns "lin" if:
    - parameter is "__kipp_env" or "__kipp_env_r" (Kippenhahn diagram)

    These parameters are locked — other scale modes are unavailable.
    """
    if param is None:
        return None

    # Check A_, Ac_ prefixes
    if any(str(param).startswith(p) for p in FORCED_AXIS_SCALE_PREFIXES):
        return "log(x)"

    # Check log prefix (e.g. logg, logL, logTeff)
    # These parameters are ALREADY logarithmized in files, so we lock to "log(x)"
    param_str = str(param)
    if param_str.startswith("log") and len(param_str) > 3:
        next_char = param_str[3]
        if next_char.isupper() or param_str in ("logg", "logl", "logteff"):
            return "log(x)"

    # geff is log(g) under a different name, lock to "log(x)"
    if param_str == "geff":
        return "log(x)"

    # Kippenhahn diagram — lin only
    if param_str in ("__kipp_env", "__kipp_env_r"):
        return "lin"

    # Check explicitly defined keys
    if str(param) in FORCED_AXIS_SCALE_KEYS:
        return "log(x)"

    return None


def get_default_axis_scale(param: Optional[str]) -> str:
    """
    Returns the default axis scale for the given parameter.

    Logic:
    1. If parameter has a forced scale (A_, Ac_, MH, etc.) — return it
    2. If parameter starts with "log" and is not in DEFAULT_AXIS_SCALE — return "log(x)"
    3. Otherwise look up in DEFAULT_AXIS_SCALE or return "lin"
    """
    if param is None:
        return "lin"

    forced = get_forced_axis_scale(param)
    if forced is not None:
        return forced

    param_str = str(param)

    # Check explicitly defined defaults
    if param_str in DEFAULT_AXIS_SCALE:
        return DEFAULT_AXIS_SCALE[param_str]

    # Auto-detect for parameters with "log" prefix
    # (e.g. logReff, logRho etc. not in DEFAULT_AXIS_SCALE)
    if param_str.startswith("log") and len(param_str) > 3:
        next_char = param_str[3]
        if next_char.isupper() or next_char.islower():
            return "log(x)"

    return "lin"


def get_axis_scale_options(param: Optional[str]):
    forced = get_forced_axis_scale(param)
    if forced is None:
        return AXIS_SCALE_CHOICES

    # Disable everything except the forced one
    out = []
    for o in AXIS_SCALE_CHOICES:
        out.append({**o, "disabled": (o["value"] != forced)})
    return out


def get_effective_axis_scale(
        param: Optional[str],
        selected_scale: Optional[str],
) -> str:
    forced = get_forced_axis_scale(param)
    if forced is not None:
        return forced
    if selected_scale in ("lin", "log", "log(x)"):
        return selected_scale
    return get_default_axis_scale(param)


def get_axis_label_without_log_prefix(param: Optional[str], label: str, scale_mode: str) -> str:
    """
    Removes the 'log' prefix from the axis title if:
    1. Parameter starts with 'log' (e.g. logg, logL, logTeff)
    2. Scale mode is 'log(x)' (data are already logarithmized)

    Examples:
    - logg: "$\log \ g$" -> "$g$" (if scale_mode == "log(x)")
    - logL: "$\log L$" -> "$L$" (if scale_mode == "log(x)")
    - logTeff: "$\log T_{eff}$" -> "$T_{eff}$" (if scale_mode == "log(x)")
    """
    if param is None or scale_mode != "log(x)":
        return label

    param_str = str(param)

    # Check whether parameter starts with "log"
    if not param_str.startswith("log"):
        return label

    # Verify this is indeed a logarithmic parameter
    if len(param_str) <= 3:
        return label

    next_char = param_str[3]
    if not (next_char.isupper() or param_str in ("logg", "logl", "logteff")):
        return label

    # Remove "log" prefix from LaTeX label
    # Handle various notation styles:
    # "$\log \ g$" -> "$g$"
    # "$\log g$" -> "$g$"
    # "$\log(g)$" -> "$g$"
    # "$\mathrm{log}\ g$" -> "$g$"

    import re

    patterns = [
        (r'\$\\log\s*\\\s*', r'$'),  # "$\log \ " -> "$"
        (r'\$\\log\s+', r'$'),  # "$\log " -> "$"
        (r'\$\\log', r'$'),  # "$\log" -> "$"
        (r'\$\\mathrm\{log\}\s*\\\s*', r'$'),  # "$\mathrm{log}\ " -> "$"
        (r'\$\\mathrm\{log\}\s+', r'$'),  # "$\mathrm{log} " -> "$"
        (r'\$\\mathrm\{log\}', r'$'),  # "$\mathrm{log}" -> "$"
    ]

    result = label
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    return result