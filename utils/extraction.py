from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from rapidfuzz import fuzz, process


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _clean_number_token(token: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", token)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _find_number_tokens(text: str) -> List[int]:
    # Capture digit groups that may include spaces, commas, or punctuation
    raw_tokens = re.findall(r"\d[\d\s,./:-]{2,}\d|\d{4,9}", text)
    values = [_clean_number_token(t) for t in raw_tokens]
    return [v for v in values if v is not None]


def _find_number_tokens_loose(text: str) -> List[int]:
    raw_tokens = re.findall(r"\d{3,9}", text)
    values = [_clean_number_token(t) for t in raw_tokens]
    return [v for v in values if v is not None]


def best_fuzzy_match(value: str, choices: List[str]) -> Tuple[str, float]:
    if not choices or not value:
        return value, 0.0
    match = process.extractOne(value, choices, scorer=fuzz.WRatio)
    if match is None:
        return value, 0.0
    return match[0], float(match[1]) / 100.0


def extract_hp(text: str, lines: List[str]) -> Tuple[int | None, float]:
    # Patterns like "50 HP", "HP 50", "Horse Power: 50"
    patterns = [
        r"(\d{1,3})\s*(?:hp|h\.p\.|horse\s*power)",
        r"(?:hp|h\.p\.|horse\s*power)\s*[:\-]?\s*(\d{1,3})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1)), 0.85

    # Line-based fallback
    for line in lines:
        if re.search(r"horse\s*power|\bh\s*p\b|h\.p\.|\bhp\b", line, re.IGNORECASE):
            m = re.search(r"(\d{1,3})", line)
            if m:
                return int(m.group(1)), 0.7
        # OCR-noisy patterns like "YSHP" or "HP" separated by non-letters
        if re.search(r"h[^a-zA-Z0-9]{0,2}p", line, re.IGNORECASE):
            m = re.search(r"(\d{1,3})", line)
            if m:
                return int(m.group(1)), 0.6
    return None, 0.0


def extract_cost(text: str, lines: List[str]) -> Tuple[int | None, float]:
    min_cost = 10000
    max_cost = 10000000
    # Prefer values near cost-related keywords
    keyword_patterns = [
        r"asset\s*cost",
        r"invoice\s*value",
        r"total\s*value",
        r"grand\s*total",
        r"total\s*amount",
        r"total\s*amt",
        r"amount\s*payable",
        r"price",
        r"cost",
    ]

    for idx, line in enumerate(lines):
        if any(re.search(pat, line, re.IGNORECASE) for pat in keyword_patterns):
            window = [line]
            if idx + 1 < len(lines):
                window.append(lines[idx + 1])
            if idx + 2 < len(lines):
                window.append(lines[idx + 2])

            # Prefer first reasonable 5-7 digit amount in nearby lines
            for wline in window:
                for token in re.findall(r"\d{3,9}", wline):
                    if len(token) < 5:
                        continue
                    value = _clean_number_token(token)
                    if value is None:
                        continue
                    if min_cost <= value <= max_cost:
                        return value, 0.8

            candidates = []
            for wline in window:
                candidates.extend([v for v in _find_number_tokens(wline) if min_cost <= v <= max_cost])
            if not candidates:
                for wline in window:
                    candidates.extend([v for v in _find_number_tokens_loose(wline) if min_cost <= v <= max_cost])
            if candidates:
                # Prefer 5-7 digit amounts when available
                preferred = [v for v in candidates if 5 <= len(str(v)) <= 7]
                return max(preferred or candidates), 0.75

    # Fallback: pick the max numeric value in the document
    values = [v for v in _find_number_tokens(text) if min_cost <= v <= max_cost]
    if not values:
        return None, 0.0
    return max(values), 0.5


def _best_line_fuzzy_match(lines: Iterable[str], choices: List[str]) -> Tuple[str | None, float]:
    best_value = None
    best_score = 0.0
    for line in lines:
        line_clean = normalize_line(line)
        if not line_clean:
            continue
        match, score = best_fuzzy_match(line_clean, choices)
        if score > best_score:
            best_value = match
            best_score = score
    return best_value, best_score


def extract_dealer_name(lines: List[str], dealer_master: List[str]) -> Tuple[str | None, float]:
    # Keyword-driven extraction
    patterns = [
        r"dealer\s*name\s*[:\-]\s*(.+)",
        r"dealer\s*[:\-]\s*(.+)",
    ]
    for line in lines:
        for pat in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                name = normalize_line(m.group(1))
                if name:
                    return name, 0.7

    # If master list exists, fuzzy match against all lines
    if dealer_master:
        best_value, best_score = _best_line_fuzzy_match(lines, dealer_master)
        if best_value:
            return best_value, min(0.9, best_score)

    # Fallback: pick a plausible header line
    header_candidates = []
    header_keywords = [
        "tractor",
        "motors",
        "agro",
        "farm",
        "industries",
        "enterprise",
        "company",
        "ltd",
        "pvt",
        "dealer",
    ]
    for line in lines[:10]:
        if len(line) < 6:
            continue
        if any(k in line.lower() for k in header_keywords) and len(line) >= 10:
            header_candidates.append(line)
        # Fallback: high alpha ratio line in header region
        alpha_count = sum(ch.isalpha() for ch in line)
        if len(line) >= 10 and alpha_count / max(len(line), 1) > 0.75 and not re.search(r"\d", line):
            header_candidates.append(line)
    if header_candidates:
        best = max(header_candidates, key=len)
        return best, 0.4

    return None, 0.0


def extract_model_name(lines: List[str], model_master: List[str], full_text: str) -> Tuple[str | None, float]:
    # If master list exists, exact match (case-insensitive) in full text
    if model_master:
        lowered = full_text.lower()
        for model in model_master:
            if model and model.lower() in lowered:
                return model, 0.9

    # Keyword-based fallback
    patterns = [
        r"model\s*name\s*[:\-]\s*(.+)",
        r"model\s*[:\-]\s*(.+)",
        r"tractor\s*model\s*[:\-]\s*(.+)",
        r"description\s*[:\-]\s*(.+)",
    ]
    for line in lines:
        for pat in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                name = normalize_line(m.group(1))
                if name:
                    return name, 0.6

    # Heuristic: look for alphanumeric model-like tokens only on relevant lines
    ignore_keywords = ["gstin", "ifsc", "account", "bank", "mob", "mobile", "phone"]
    for line in lines:
        if any(k in line.lower() for k in ignore_keywords):
            continue
        if re.search(r"m.{0,3}o.{0,3}d|tractor|power", line, re.IGNORECASE):
            m = re.search(r"[:\-]\s*([A-Za-z0-9\- ]{3,})", line)
            if m:
                return normalize_line(m.group(1).lstrip("-: ")), 0.5
            m = re.search(r"(\d{3,4}\s*[A-Za-z]{1,3})", line)
            if m:
                return normalize_line(m.group(1)), 0.5
            m = re.search(r"([A-Za-z]{2,}\s*\d{2,4}[A-Za-z0-9\-]*)", line)
            if m:
                return normalize_line(m.group(1)), 0.45
    return None, 0.0


def aggregate_confidences(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def extract_fields(full_text: str, dealer_master: List[str], model_master: List[str]) -> Tuple[Dict, float]:
    text = normalize_text(full_text)
    lines = [normalize_line(line) for line in full_text.splitlines() if normalize_line(line)]

    dealer, dealer_score = extract_dealer_name(lines, dealer_master)
    model, model_score = extract_model_name(lines, model_master, text)
    hp, hp_score = extract_hp(text, lines)
    cost, cost_score = extract_cost(text, lines)

    if dealer and dealer_master:
        dealer, fuzz_score = best_fuzzy_match(dealer, dealer_master)
        dealer_score = max(dealer_score, fuzz_score)
    if model and model_master:
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
