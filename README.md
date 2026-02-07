# AI Agent Lab

A CLI-based project demonstrating agent-based workflows using LangGraph.

This repository contains three core agents:

- File Agent – summarize files, extract TODOs, or rewrite content
- Router Agent – determine intent and route requests automatically
- Validator Agent – deterministic entity classification with no LLM usage

The project emphasizes:
- clean agent orchestration
- predictable outputs
- no exposure of model “thinking”

---

## Project Structure

src/
├── main.py              # CLI entry point
├── agent_file.py        # File agent
├── agent_router.py      # Router agent
├── agent_validator.py   # Deterministic validator
├── hf_client.py         # HuggingFace client wrapper
└── schemas.py           # Optional shared schemas

---

## Requirements

- Python 3.10+
- HuggingFace API token (for file/router agents)

---

## Setup

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/ai-agent-lab.git
cd ai-agent-lab

### 2. Create and activate a virtual environment

python -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Environment variables

Create a .env file in the repo root:

HUGGINGFACE_API_TOKEN=your_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2

Do not commit secrets. The .env file is ignored by git.

---

## Running the CLI

From the project root:

python -m src.main

You will see:

Choose: 1=file agent  2=router agent  3=validator agent

---

## Option 1 — File Agent

Use this for direct file operations.

Supported modes:
- summarize
- todos
- rewrite

Example:

> 1
mode (summarize/todos/rewrite): summarize
file path: requirements.txt

---

## Option 2 — Router Agent

The router decides what to do based on input.

Examples:

> 2
summarize:requirements.txt

> 2
extract todos from src/main.py

> 2
OpenAI

---

## Option 3 — Validator Agent (Deterministic)

Classifies an entity into one of:

company | person | place | product | organization | concept | other

This agent:
- does NOT call an LLM
- is deterministic and fast
- always returns valid structured output

Examples:

> 3
Entity to classify: McDonalds
{'label': 'company', 'confidence': 0.85}

> 3
Entity to classify: Pikachu
{'label': 'product', 'confidence': 0.6}

---

