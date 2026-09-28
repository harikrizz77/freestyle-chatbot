"""Method selector (build plan section 8, node 4): reads `config/method_rules.yaml`
and picks the FIRST matching rule against the `DataProfile` computed by `data_agent`.

Deliberately implemented as deterministic Python, not an LLM call: rule matching here
is mechanical membership/threshold testing against facts the data_agent already
computed, and the build plan is explicit that this step "does NOT invent methods and
NEVER computes an estimate" -- pure code is strictly more auditable and testable than
an LLM reading the same YAML, with identical behavior. If a future revision wants an
LLM in the loop (e.g. to explain *why* a rule fired), that belongs in `reporter.py`,
which narrates but never re-decides.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import yaml

from config.settings import METHOD_RULES_PATH
from src.schema.tools import DataProfile, MethodSelection

_THRESHOLD_PATTERN = re.compile(r"^\s*(>=|<=|>|<|==)\s*(\w+)\s*$")


@functools.lru_cache(maxsize=1)
def _load_rules(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _rule_matches(
    when: dict[str, Any], data_profile: DataProfile, thresholds: dict[str, Any]
) -> bool:
    for field, expected in when.items():
        actual = getattr(data_profile, field, None)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif isinstance(expected, bool):
            if actual is not expected:
                return False
        elif isinstance(expected, str):
            match = _THRESHOLD_PATTERN.match(expected)
            if not match:
                raise ValueError(f"unrecognized rule condition for {field!r}: {expected!r}")
            op, threshold_key = match.groups()
            threshold_value = thresholds[threshold_key]
            if not _compare(actual, op, threshold_value):
                return False
        else:
            if actual != expected:
                return False
    return True


def _compare(actual: Any, op: str, threshold: Any) -> bool:
    if op == ">=":
        return bool(actual >= threshold)
    if op == "<=":
        return bool(actual <= threshold)
    if op == ">":
        return bool(actual > threshold)
    if op == "<":
        return bool(actual < threshold)
    if op == "==":
        return bool(actual == threshold)
    raise ValueError(f"unsupported operator {op!r}")


def select_method(
    data_profile: DataProfile, rules_path: str | Path = METHOD_RULES_PATH
) -> MethodSelection:
    """Evaluate `config/method_rules.yaml` top-to-bottom; the first matching rule
    wins. Raises `ValueError` if no rule matches (should be unreachable given the
    catch-all `sparse_bayesian_fallback` rule -- treated as a config bug if it fires).
    """
    config = _load_rules(str(rules_path))
    thresholds = config["thresholds"]

    for rule in config["rules"]:
        if _rule_matches(rule["when"], data_profile, thresholds):
            return MethodSelection(
                rule_id=rule["id"],
                primary_method=rule["primary"],
                cross_check_methods=rule.get("cross_checks", []),
                segmentation=rule.get("segmentation", "none"),
                confidence_ceiling=rule.get("confidence_ceiling"),
                notes=rule.get("notes", ""),
            )

    raise ValueError(
        f"no method_rules.yaml rule matched data_profile={data_profile!r}; "
        "this should be unreachable given the catch-all fallback rule -- check the config."
    )
