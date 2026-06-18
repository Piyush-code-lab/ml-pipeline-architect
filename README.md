# ML Pipeline Architect

An autonomous multi-agent ML assistant built with LangChain and Groq. Given a plain English description of any machine learning problem, the system analyzes it, engineers features, selects models, generates executable code, and self-corrects through an agentic execution loop — all without human intervention.

---

## What It Does

### Tab 1 — Autonomous ML Pipeline

A router-based multi-agent system that takes a problem description and autonomously builds a complete ML pipeline.

The Planner agent analyzes the problem first and decides which downstream agents are actually needed. A clustering problem skips code generation. A simple classification problem might not need feature engineering. Only the relevant agents run.

Each agent returns structured Pydantic objects, not raw text strings. This means outputs are programmatically accessible — the planner's `tools_needed` field directly drives which agents execute next.

The generated code is then passed to a self-correcting execution loop. The code runs in a sandboxed environment. If it throws an error, the error is fed back to the LLM along with the broken code, and the LLM attempts a fix. This repeats up to 3 times. The final status — success, attempts taken, last error — is displayed alongside the code.

**Agent chain:**

```
User Problem
    |
    v
Planner Agent  -->  PlannerOutput (Pydantic)
    |
    |-- tools_needed --> Router
                            |
                            |--> Feature Engineer Agent  -->  FeatureEngineerOutput (Pydantic)
                            |--> Model Selector Agent    -->  ModelSelectorOutput (Pydantic)
                            |--> Code Generator Agent    -->  str (executable Python)
                                        |
                                        v
                                Self-Correcting Execution Loop
                                (max 3 attempts, error feedback)
```

### Tab 2 — Smart Notes Builder

Generates structured notes on any topic. Accepts multiple optional input sources simultaneously:

- A topic string (always required)
- An uploaded PDF
- A blog or article URL
- Web search via DuckDuckGo (checkbox)

All sources are combined and truncated to fit model token limits. The LLM then generates notes grounded in the provided context, or falls back to its own knowledge if no context is given.

Output format is user-selectable:

- **Plain** — rendered Markdown with proper mathematical notation using LaTeX delimiters
- **LaTeX** — compilable `.tex` file available for download, ready to run on Overleaf

### Tab 3 — PDF Q&A with Conversation Memory

Implements a full RAG pipeline over any uploaded PDF. The document is chunked, embedded using a local HuggingFace sentence transformer, and stored in a ChromaDB vector store. On each question, the top 3 most relevant chunks are retrieved and passed to the LLM as context.

Conversation memory is maintained per PDF session. Follow-up questions have access to the full prior exchange, enabling multi-turn dialogue over the document.

---

## Architecture

```
ml-pipeline-architect/
├── agents/
│   ├── schemas.py            # Pydantic output schemas
│   ├── planner.py            # Problem analysis + router decisions
│   ├── feature_engineer.py   # Feature engineering recommendations
│   ├── model_selector.py     # Model recommendations + evaluation metric
│   ├── code_generator.py     # Executable sklearn pipeline generation
│   └── code_executor.py      # Self-correcting execution loop
├── chains/
│   └── pipeline_chain.py     # Router + sequential agent orchestration
├── notes/
│   ├── fetcher.py            # PDF, URL, web search content extraction
│   └── notes_chain.py        # Notes generation (plain + LaTeX)
├── rag/
│   ├── indexer.py            # PDF loading + ChromaDB indexing
│   └── qa_chain.py           # Retrieval chain with conversation memory
├── utils/
│   └── formatters.py         # Pydantic to Markdown formatters
├── app.py                    # Gradio UI (3 tabs)
├── main.py                   # Entry point
└── requirements.txt
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API — Llama 3.3 70B Versatile |
| Agent Framework | LangChain |
| Structured Outputs | Pydantic v2 + LangChain `.with_structured_output()` |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API cost) |
| PDF Parsing | pdfplumber (notes), PyPDFLoader (RAG) |
| Web Search | DuckDuckGo Search (free, no API key) |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

---

## Key Technical Decisions

**Why Pydantic structured outputs instead of string parsing?**

Raw LLM string outputs are unparseable downstream. By using `llm.with_structured_output(PlannerOutput)`, each agent's response is a typed Python object. The planner's `tools_needed: list[str]` field directly controls which agents run next. If the planner returns `["feature_engineering", "model_selection"]` without `"code_generation"`, the code generator and executor never run.

**Why a router instead of a fixed sequential chain?**

A fixed chain runs all four agents on every problem regardless of what is actually needed. A clustering problem has no target variable and does not need the same pipeline as a supervised regression problem. The router makes this decision based on the planner's structured output, not hardcoded logic.

**Why a self-correcting execution loop?**

LLM-generated code frequently references placeholder files, uses incorrect column names, or has import errors. Rather than silently returning broken code, the executor catches the exception, formats the error alongside the code into a correction prompt, and requests a fix. The loop runs up to 3 times and reports the final outcome — success, number of attempts, and last error if failed.

**Why local HuggingFace embeddings for RAG?**

Using OpenAI or Cohere embeddings costs money per token and adds latency. `all-MiniLM-L6-v2` runs locally, is fast enough for document-scale RAG, and produces quality embeddings for semantic search. No API key required, no cost.

---

## Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/Piyush-code-lab/ml-pipeline-architect
cd ml-pipeline-architect
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

**5. Run the app**

```bash
python main.py
```

Open `http://localhost:7860` in your browser.

---

## Live Demo

Deployed on HuggingFace Spaces: [https://huggingface.co/spaces/Piyush-code-lab/ml-pipeline-architect](https://huggingface.co/spaces/Piyush-code-lab/ml-pipeline-architect)

Note: The free Groq tier has rate limits. If you hit a rate limit error, wait 60 seconds and retry.

---

## Limitations

- The self-correcting execution loop cannot successfully run code that requires real data files (e.g. `pd.read_csv('data.csv')`). The generated code is structurally correct and ready to use — it just cannot execute end-to-end without actual data.
- The notes builder truncates combined context to 8000 characters to stay within Groq's token-per-minute limits on the free tier.
- Conversation memory in Tab 3 is session-based and resets when the app restarts.

---

## Related Projects

- [Home Credit Default Risk](https://github.com/Piyush-code-lab/home-credit-default-risk) — XGBoost classifier on 307,511 applicants, ROC-AUC 0.786, deployed on HuggingFace Spaces
- [Aircraft Engine RUL Prediction](https://github.com/Piyush-code-lab/RUL-Prediction) — Bidirectional LSTM with custom attention mechanism on NASA CMAPSS, RMSE 14.39