"""Relationship agent (build plan section 8, node 3): build the choice set /
substitutability scores. Thin wrapper over `src.tools.build_choice_set` -- the
attribute-embedding similarity logic itself lives in the tools layer so it stays
independently unit-tested without any agent/graph machinery.
"""

from __future__ import annotations

from src.schema.core import AnalysisRequest, Product
from src.schema.tools import ChoiceSet
from src.tools import build_choice_set


def build_relationships(
    products: list[Product], request: AnalysisRequest, min_substitutability: float = 0.05
) -> ChoiceSet:
    return build_choice_set(products, request, min_substitutability=min_substitutability)
