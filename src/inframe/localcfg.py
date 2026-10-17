"""Per-machine settings for the developer tools (regression gates, test composites): paths to clips that are not in the repo.

Each value comes from an environment variable, else from ~/.config/inframe/local.json (override the file with INFRAME_LOCAL_CONFIG).
Nothing here is read by the product itself; tools/regress.py, tools/browser_regress.py, tools/splice.py and tools/bench_units.py use it.

    key                 environment variable            used for
    regress_video       INFRAME_REGRESS_VIDEO           the video the regression gates run detect on
    regress_baseline    INFRAME_REGRESS_BASELINE        the accepted segments.json of that video
    splice_target_dir   INFRAME_SPLICE_TARGET_DIR       the folder of clips of the target person (*_01_*.mov, *_03_*.mov, ...), with a
                                                        target_windows.json beside them: {"targetA": [file seconds, first, last second in frame], ...}
    dev_samples         INFRAME_DEV_SAMPLES             the folder of openly licensed sample clips (docs/TESTDATA.md)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

KEYS = {
    "regress_video": "INFRAME_REGRESS_VIDEO",
    "regress_baseline": "INFRAME_REGRESS_BASELINE",
    "splice_target_dir": "INFRAME_SPLICE_TARGET_DIR",
    "dev_samples": "INFRAME_DEV_SAMPLES",
}
CONFIG_ENV = "INFRAME_LOCAL_CONFIG"


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV) or Path.home() / ".config/inframe/local.json")


def _file_values() -> dict:
    try:
        data = json.loads(config_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def get(key: str) -> Path | None:
    """The configured path for `key` (environment first, then the config file), or None when neither is set."""
    if key not in KEYS:
        raise KeyError(key)
    v = os.environ.get(KEYS[key]) or _file_values().get(key)
    return Path(str(v)).expanduser() if v else None
