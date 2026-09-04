#!/usr/bin/env python3
"""Check structural and numerical equivalence of the Chinese and English contracts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CJK = re.compile(r"[\u3400-\u9fff]")
ENGLISH_QUALITY_FORBIDDEN = (
    re.compile(r"\breartifaction\b", re.IGNORECASE),
    re.compile(r"\bartifaction\b", re.IGNORECASE),
    re.compile(r"\bassignd\b", re.IGNORECASE),
    re.compile(r"\bno any\b", re.IGNORECASE),
    re.compile(r"\bcan not\b", re.IGNORECASE),
    re.compile(r"\bthe the\b", re.IGNORECASE),
    re.compile(r"\bpriority note\b", re.IGNORECASE),
    re.compile(r"(?:task card|finding)”s", re.IGNORECASE),
    re.compile(r"\.(?:If|The|Score)"),
)
SCORE_ARITHMETIC = re.compile(
    r"(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value


def normalized_key(key: str) -> str:
    return key[:-3] + "_lang" if key.endswith(("_zh", "_en")) else key


def compare(
    left: Any,
    right: Any,
    errors: list[str],
    *,
    path: str = "$",
    left_key: str = "",
) -> None:
    if isinstance(left, dict):
        if not isinstance(right, dict):
            errors.append(f"{path}: type mismatch")
            return
        left_map = {normalized_key(key): (key, value) for key, value in left.items()}
        right_map = {normalized_key(key): (key, value) for key, value in right.items()}
        if set(left_map) != set(right_map):
            errors.append(f"{path}: key mismatch")
            return
        for normalized in left_map:
            left_name, left_value = left_map[normalized]
            right_name, right_value = right_map[normalized]
            if left_name.endswith("_zh") and not right_name.endswith("_en"):
                errors.append(f"{path}.{left_name}: missing English language key")
            elif not left_name.endswith("_zh") and left_name != right_name:
                errors.append(f"{path}.{left_name}: key identity mismatch")
            compare(
                left_value,
                right_value,
                errors,
                path=f"{path}.{left_name}",
                left_key=left_name,
            )
        return

    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            errors.append(f"{path}: list shape mismatch")
            return
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            compare(
                left_value,
                right_value,
                errors,
                path=f"{path}[{index}]",
                left_key=left_key,
            )
        return

    if isinstance(left, str):
        if not isinstance(right, str):
            errors.append(f"{path}: string type mismatch")
        elif CJK.search(left):
            if not right.strip() or CJK.search(right):
                errors.append(f"{path}: untranslated text")
        elif left != right:
            errors.append(f"{path}: non-language string changed")
        return

    if type(left) is not type(right) or left != right:
        errors.append(f"{path}: scalar identity mismatch")


def check_english_quality(value: Any, errors: list[str], *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            check_english_quality(child, errors, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            check_english_quality(child, errors, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    for pattern in ENGLISH_QUALITY_FORBIDDEN:
        if pattern.search(value):
            errors.append(f"{path}: malformed English fragment: {pattern.pattern}")


def check_score_arithmetic(specification: dict[str, Any], errors: list[str], label: str) -> None:
    for family in specification.get("families", []):
        for action in family.get("actions", []):
            weight = float(action.get("weight", 0))
            for level, reasoning in action.get("example_reasoning", {}).items():
                for factor, multiplier, result in SCORE_ARITHMETIC.findall(str(reasoning)):
                    factor_value = float(factor)
                    multiplier_value = float(multiplier)
                    result_value = float(result)
                    if (
                        not math.isclose(factor_value, weight)
                        or not math.isclose(multiplier_value, float(level))
                        or not math.isclose(factor_value * multiplier_value, result_value)
                    ):
                        errors.append(
                            f"{label}:{action.get('id')}:{level}: score arithmetic does not match weight"
                        )


def check_novelty_weights(specification: dict[str, Any], errors: list[str], label: str) -> None:
    actions = {
        action.get("id"): action
        for family in specification.get("families", [])
        for action in family.get("actions", [])
    }
    weights = actions.get("independent_origin.frozen_search_prior_art", {}).get(
        "claim_atom_weights", {}
    )
    expected = {"object", "scope", "relation", "direction", "boundary"}
    try:
        values = [float(value) for value in weights.values()] if weights else []
    except (TypeError, ValueError):
        values = []
    if set(weights) != expected or not values or any(value <= 0 for value in values) or not math.isclose(sum(values), 1):
        errors.append(f"{label}: novelty atom weights must be positive, complete, and sum to 1")


def main() -> int:
    chinese = load_json(HERE / "specification.json")
    english = load_json(HERE / "specification_en.json")
    errors: list[str] = []
    compare(chinese, english, errors)
    check_english_quality(english, errors)
    check_score_arithmetic(chinese, errors, "zh")
    check_score_arithmetic(english, errors, "en")
    check_novelty_weights(chinese, errors, "zh")
    check_novelty_weights(english, errors, "en")

    chinese_actions = [
        action for family in chinese.get("families", []) for action in family.get("actions", [])
    ]
    english_actions = [
        action for family in english.get("families", []) for action in family.get("actions", [])
    ]
    if len(chinese_actions) != 29 or len(english_actions) != 29:
        errors.append("action count must be 29 in both specifications")
    if (
        sum(action.get("weight", 0) for action in chinese_actions) != 100
        or sum(action.get("weight", 0) for action in english_actions) != 100
    ):
        errors.append("action weights must total 100 in both specifications")
    if [action.get("id") for action in chinese_actions] != [
        action.get("id") for action in english_actions
    ]:
        errors.append("action identity or order differs between specifications")

    receipt = {
        "release": "V1.0",
        "status": "PASS" if not errors else "FAIL",
        "family_count": len(english.get("families", [])),
        "action_count": len(english_actions),
        "weight_total": sum(action.get("weight", 0) for action in english_actions),
        "errors": errors,
    }
    print(json.dumps(receipt, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
