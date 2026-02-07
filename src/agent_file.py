from __future__ import annotations
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from .hf_client import chat
import re

class FileState(TypedDict, total=False):
    mode: Literal["summarize", "todos", "rewrite"]
    file_path: str
    file_text: str
    result: str
    error: str

def read_file_node(state: FileState) -> FileState:
    path = state["file_path"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"file_text": f.read()}
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

def summarize_node(state: FileState) -> FileState:
    text = (state.get("file_text") or "").strip()
    if not text:
        return {"error": "File empty"}
    prompt = f"Summarize in 5 bullets:\n\n{text[:8000]}"
    return {"result": chat(prompt, max_tokens=250, temperature=0.2)}

def todos_node(state: FileState) -> FileState:
    text = (state.get("file_text") or "")
    if not text.strip():
        return {"error": "File empty"}

    # Grab literal TODO-style markers
    matches = re.findall(r"(?im)^\s*#?\s*(TODO|FIXME|XXX)\b.*$", text)

    # The above returns just the marker; use finditer to return full lines instead:
    lines = []
    for m in re.finditer(r"(?im)^\s*(?:#\s*)?(TODO|FIXME|XXX)\b.*$", text):
        lines.append(m.group(0).strip())

    if not lines:
        return {"result": "No TODOs found."}

    return {"result": "\n".join(f"- {line}" for line in lines)}

def rewrite_node(state: FileState) -> FileState:
    text = (state.get("file_text") or "").strip()
    if not text:
        return {"error": "File empty"}
    prompt = f"""
Rewrite the following text to be clearer and more concise.
Preserve meaning. Keep formatting if present.

TEXT:
{text[:8000]}
""".strip()
    return {"result": chat(prompt, max_tokens=400, temperature=0.2)}

def route_mode(state: FileState) -> str:
    if state.get("error"):
        return "end"
    return state["mode"]

def build_file_agent():
    g = StateGraph(FileState)
    g.add_node("read", read_file_node)
    g.add_node("summarize", summarize_node)
    g.add_node("todos", todos_node)
    g.add_node("rewrite", rewrite_node)

    g.set_entry_point("read")

    g.add_conditional_edges("read", route_mode, {
        "summarize": "summarize",
        "todos": "todos",
        "rewrite": "rewrite",
        "end": END,
    })

    g.add_edge("summarize", END)
    g.add_edge("todos", END)
    g.add_edge("rewrite", END)
    return g.compile()
