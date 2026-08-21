# AdvisorIQ — Wealth-Management Copilot

A Gradio app that answers investment questions with a LangChain ReAct agent, grounds
its answers in a local document library, and scans uploaded transaction CSVs for
suspicious activity. PII is redacted on the way in; over-promising language is stripped
on the way out.

> Educational project. Nothing it produces is financial advice.

## How it works

A question (and optionally a transactions CSV) enters through the Gradio UI. Guardrails
redact PII before anything reaches the model. The agent then reasons about which of its
three tools to call, and the answer is passed back through an output guardrail.

| Tool | Backed by | Purpose |
|---|---|---|
| `MarketData` | yfinance | Live price and 1-month trend for a ticker |
| `Research` | Chroma + all-MiniLM-L6-v2 | RAG over `data/docs/` |
| `CheckTransactions` | RandomForest | Scores uploaded transactions for suspicion |

| File | Role |
|---|---|
| `app.py` | Gradio UI, request flow, guardrail calls |
| `agent.py` | LLM, the three tools, ReAct agent |
| `rag.py` | Builds and loads the Chroma index |
| `checker.py` | Trains and applies the transaction classifier |
| `guardrails.py` | PII redaction, banned-phrase filter, disclaimer |
| `evaluate.py` | ragas faithfulness / answer-relevancy scoring |
| `data/gen_sample_data.py` | Regenerates the sample dataset and docs |

## Setup

Requires Python 3.12 and a Google Gemini API key
([get one here](https://aistudio.google.com/apikey)).

```bash
git clone https://github.com/Kiran-B23/AFCapstone.git
cd AFCapstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Export your key — the app will not start without it:

```bash
export GOOGLE_API_KEY=your-key-here    # or GEMINI_API_KEY
```

Build the two artifacts (both are gitignored, so a fresh clone must create them):

```bash
python checker.py    # trains the classifier -> models/checker.joblib
python rag.py        # indexes data/docs/    -> chroma_store/
```

`checker.py` prints its own evaluation. On the shipped dataset it scores
ROC-AUC ≈ 0.99, with recall ≈ 0.86 on the suspicious class.

## Run

```bash
python app.py
```

Gradio serves on http://127.0.0.1:7860. Ask something like *"Is Apple a good buy right
now?"*, or upload a CSV and ask it to check your transactions. An upload needs the
columns `amount`, `hour`, `is_foreign`, and `merchant_risk` —
`data/sample_transactions.csv` is a working example.

## Sample data

`data/gen_sample_data.py` writes 600 deterministic rows (seed 42) plus the five research
documents. Suspicion is a function of large amounts, foreign countries, risky merchants,
and odd hours, with noise added so the task isn't trivially separable.

The `account_number` column holds fabricated Visa-shaped numbers **on purpose** — they
exist to exercise `redact_pii`. No real account data is in this repo.

## Guardrails

- **Input** — card/account numbers, emails, and phone numbers are replaced with
  placeholders before the question or CSV reaches the model.
- **Output** — phrases like *"guaranteed return"* and *"risk-free"* are rewritten,
  answers with no retrieved sources are marked as weakly supported, and a disclaimer is
  appended to every response.

Note that `redact_pii` is regex-based and catches common formats only. It is a
demonstration of the pattern, not a compliance-grade scrubber.

## Known issue: `evaluate.py` does not run

`import ragas` fails on the pinned stack:

```
ragas/llms/base.py:12
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

ragas 0.4.3 (the latest release) imports a module that `langchain-community` 0.4.2
removed. There is no newer ragas to upgrade to, and downgrading `langchain-community`
to 0.3.x breaks the `langchain` 1.x stack that `agent.py` depends on. Resolving it means
either running ragas 0.2.x against an older LangChain in a separate environment, or
reimplementing the two metrics with direct Gemini calls. The other six modules are
unaffected.

## Dependencies

`requirements.txt` uses exact pins. This is deliberate: LangChain 1.x relocated `Tool`
to `langchain_core.tools` and moved `initialize_agent` / `AgentType` into
`langchain-classic`, so an unpinned install breaks `agent.py` at import time.

`initialize_agent` is itself deprecated even within `langchain-classic`. The modern
equivalent is `create_agent` from `langchain.agents`, which would also mean replacing
`agent.run(question)` in `app.py` with an invoke call.
