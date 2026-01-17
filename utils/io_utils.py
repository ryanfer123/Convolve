from __future__ import annotations

from pathlib import Path
from typing import List


def load_master_list(path: Path | None) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Master list not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # If CSV, assume first column per line
    if path.suffix.lower() == ".csv":
        parsed = []
        for line in lines:
            parsed.append(line.split(",")[0].strip())
        return parsed

    return lines
