from __future__ import annotations

import re
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

Label = Literal["company","person","place","product","organization","concept","other"]

class ValState(TypedDict, total=False):
    entity: str
    parsed: dict
    error: str


def heuristic_label(entity: str) -> tuple[Label, float]:
    s = entity.strip()
    if not s:
        return ("other", 0.0)

    low = s.lower()

    # places
    if re.search(r"\b(city|state|country|borough|bronx|nyc|new york)\b", low):
        return ("place", 0.80)

    # org acronyms
    if re.fullmatch(r"[A-Z]{2,10}", s.replace(" ", "")):
        return ("organization", 0.70)

    # companies / brands
    if re.search(r"\b(inc|corp|llc|ltd|company|co\.)\b", low):
        return ("company", 0.85)

    # person-ish (two+ capitalized words)
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", s):
        return ("person", 0.70)

    # product-ish: common “thing” tokens; also many single-token proper nouns are brands/products
    if len(s.split()) == 1:
        # Pikachu, iPhone, PlayStation etc.
        return ("product", 0.60)

    return ("other", 0.50)


def classify_node(state: ValState) -> ValState:
    entity = (state.get("entity") or "").strip()
    if not entity:
        return {"error": "Missing entity"}

    label, conf = heuristic_label(entity)
    return {"parsed": {"label": label, "confidence": conf}, "error": ""}


def build_validator_agent():
    g = StateGraph(ValState)
    g.add_node("classify", classify_node)
    g.set_entry_point("classify")
    g.add_edge("classify", END)
    return g.compile()
