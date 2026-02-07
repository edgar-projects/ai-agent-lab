from __future__ import annotations

import json
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

from .hf_client import chat
from .agent_file import build_file_agent


class RouterState(TypedDict, total=False):
    user_text: str
    action: Literal["summarize", "todos", "rewrite", "classify_text"]
    file_path: str
    result: str
    error: str


def extract_json_object(raw: str) -> dict:
    raw = (raw or "").strip()

    # remove <think>...</think> fully (even multiline)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.replace("<think>", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in: {raw}")

    return json.loads(raw[start:end + 1])

def decide_action_node(state: RouterState) -> RouterState:
    user = (state.get("user_text") or "").strip()
    if not user:
        return {"error": "Empty input"}

    # ---- Deterministic routing (no model needed) ----
    lower = user.lower()

    # summarize:PATH
    if lower.startswith("summarize:"):
        path = user.split(":", 1)[1].strip()
        return {"action": "summarize", "file_path": path} if path else {"error": "Missing file path after summarize:"}

    # todos:PATH OR "extract todos from PATH"
    if lower.startswith("todos:"):
        path = user.split(":", 1)[1].strip()
        return {"action": "todos", "file_path": path} if path else {"error": "Missing file path after todos:"}

    if "extract todos from" in lower:
        # naive parse but works for your CLI pattern
        path = user.lower().split("extract todos from", 1)[1].strip()
        return {"action": "todos", "file_path": path} if path else {"error": "Missing file path after 'extract todos from'"}

    # rewrite:PATH
    if lower.startswith("rewrite:"):
        path = user.split(":", 1)[1].strip()
        return {"action": "rewrite", "file_path": path} if path else {"error": "Missing file path after rewrite:"}

    # ---- Fallback to model for free-form intent ----
    prompt = f"""
Decide the user's intent. Return ONLY JSON:
{{"action":"summarize|todos|rewrite|classify_text","file_path":"", "note":""}}

User message:
{user}
""".strip()

    raw = chat(prompt, max_tokens=120, temperature=0.0)

    try:
        data = extract_json_object(raw)
        action = data.get("action")
        file_path = (data.get("file_path") or "").strip()

        if action not in {"summarize", "todos", "rewrite", "classify_text"}:
            return {"error": f"Bad action from model. Raw: {raw}"}

        out: RouterState = {"action": action}
        if file_path:
            out["file_path"] = file_path
        return out

    except Exception as e:
        # Last resort: classify_text so your app still works
        return {"action": "classify_text"}

def classify_text_node(state: RouterState) -> RouterState:
    text = (state.get("user_text") or "").strip()
    if not text:
        return {"error": "Empty input"}

    prompt = f"""
Return ONLY valid JSON with:
- label: one of ["company","person","place","product","organization","concept","other"]
- confidence: a float between 0 and 1 (e.g., 0.92)

Entity: {text}
""".strip()

    raw = chat(prompt, max_tokens=120, temperature=0.0)

    # Normalize: return *parsed* JSON as a string (or store fields if you prefer)
    try:
        data = extract_json_object(raw)
        return {"result": json.dumps(data, ensure_ascii=False)}
    except Exception as e:
        return {"error": f"Classification did not return JSON: {e}. Raw: {raw}"}


def run_file_agent_node(state: RouterState) -> RouterState:
    file_path = (state.get("file_path") or "").strip()
    if not file_path:
        return {"error": "Missing file_path"}

    mode = state.get("action")  # summarize/todos/rewrite
    if mode not in {"summarize", "todos", "rewrite"}:
        return {"error": f"Bad mode for file agent: {mode}"}

    file_agent = build_file_agent()
    res = file_agent.invoke({"mode": mode, "file_path": file_path})

    if res.get("error"):
        return {"error": res["error"]}

    return {"result": res.get("result", "")}


def route_after_decide(state: RouterState) -> str:
    if state.get("error"):
        return "end"
    return state.get("action", "classify_text")


def build_router_agent():
    g = StateGraph(RouterState)

    g.add_node("decide", decide_action_node)
    g.add_node("classify_text", classify_text_node)
    g.add_node("run_file", run_file_agent_node)

    g.set_entry_point("decide")

    g.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "classify_text": "classify_text",
            "summarize": "run_file",
            "todos": "run_file",
            "rewrite": "run_file",
            "end": END,
        },
    )

    g.add_edge("classify_text", END)
    g.add_edge("run_file", END)

    return g.compile()
