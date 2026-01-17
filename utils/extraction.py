from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rapidfuzz import fuzz, process


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def best_fuzzy_match(value: str, choices: List[str]) -> Tuple[str, float]:
    if not choices or not value:
        return value, 0.0
    match = process.extractOne(value, choices, scorer=fuzz.WRatio)
    if match is None:
        return value, 0.0
    return match[0], float(match[1]) / 100.0


def extract_hp(text: str) -> Tuple[int | None, float]:
    # Patterns like "50 HP", "HP 50"
    patterns = [
        r"(\d{2,3})\s*hp",
        r"hp\s*(\d{2,3})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1)), 0.8
    return None, 0.0


def extract_cost(text: str) -> Tuple[int | None, float]:
    # Match large numbers e.g. 525000 or 5,25,000
    candidates = re.findall(r"\b\d{1,3}(?:,\d{2,3}){1,3}\b|\b\d{5,8}\b", text)
    if not candidates:
        return None, 0.0
    # Pick the max numeric value as heuristic
    def to_int(s: str) -> int:
        return int(s.replace(",", ""))
    values = [(to_int(c), c) for c in candidates]
    value = max(values, key=lambda x: x[0])[0]
    return value, 0.5


def extract_dealer_name(text: str) -> Tuple[str | None, float]:
    # Simple heuristic: line containing 'dealer'
    for line in text.splitlines():
        if "dealer" in line.lower():
            name = line.split(":")[-1].strip()
            if name:
                return name, 0.6
    return None, 0.0


def extract_model_name(text: str) -> Tuple[str | None, float]:
    # Heuristic: line with 'model'
    for line in text.splitlines():
        if "model" in line.lower():
            name = line.split(":")[-1].strip()
            if name:
                return name, 0.6
    return None, 0.0


def aggregate_confidences(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def extract_fields(full_text: str, dealer_master: List[str], model_master: List[str]) -> Tuple[Dict, float]:
    text = normalize_text(full_text)

    dealer, dealer_score = extract_dealer_name(full_text)
    model, model_score = extract_model_name(full_text)
    hp, hp_score = extract_hp(text)
    cost, cost_score = extract_cost(text)

    if dealer:
        dealer, fuzz_score = best_fuzzy_match(dealer, dealer_master)
        dealer_score = max(dealer_score, fuzz_score)
    if model:
        model, fuzz_score = best_fuzzy_match(model, model_master)
        model_score = max(model_score, fuzz_score)

    fields = {
        "dealer_name": dealer,
        "model_name": model,
        "horse_power": hp,
        "asset_cost": cost,
        "signature": {"present": False, "bbox": None},
        "stamp": {"present": False, "bbox": None},
    }

    confidence = aggregate_confidences([dealer_score, model_score, hp_score, cost_score])
    return fields, confidence
