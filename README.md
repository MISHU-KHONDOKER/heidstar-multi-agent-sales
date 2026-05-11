# Heidstar Multi-Agent Sales System

> A production-ready multi-agent AI system that automates B2B sales lead discovery, qualification, and proposal drafting for precision microscopy hardware. Built end-to-end with **LangGraph**, **Ollama** (Qwen 3), and **Tavily Search**.

Inspired by a real-world business problem at **Heidstar Technology** (Xiamen, China) — a Zeiss Class A ODM partner with 180+ digital pathology scanners installed in hospitals and research labs globally. Their sales team writes proposals one at a time, by hand, in English that is not their first language. This system automates that workflow.

---

## What It Does

You give the system a one-line search brief in plain English. It returns ranked, qualified leads — each with a tailored sales proposal ready for human review.                                                    

INPUT:   "Stem cell research laboratories in Europe doing iPSC work"

┌─────────────────────────────────────┐
     │       LangGraph Orchestrator        │
     └────────┬──────┬──────┬──────────────┘
              ▼      ▼      ▼
          Researcher  Qualifier  Writer
          (web)       (LLM+DB)   (LLM+DB)



OUTPUT:  Ranked qualified leads + tailored Markdown proposals

Three specialized agents coordinated by a Manager:

1. **Researcher Agent** — Generates diverse search queries, calls the live web, extracts structured leads.
2. **Qualifier Agent** — Filters false positives, scores each lead 0–100, matches leads to specific Heidstar products.
3. **Writer Agent** — Drafts tailored proposal emails. Includes a **two-layer hallucination guard** (prompt-level + regex-based post-validation) to prevent inventing fake product names.
4. **Manager Agent** — LangGraph state machine that orchestrates the three agents with conditional flow and shared memory.

---

## Why This Project Matters

This is not a tutorial. It is a working solution to a real revenue problem for a real company.

**The business pain:** Heidstar's sales team writes a single technical proposal in 3 days. Competitors (Leica, Zeiss) deliver automated proposals within 24 hours. Heidstar loses deals not because the product is worse — but because they are slower. Most global prospects never find them.

**What this system does:** Produces tailored proposals at scale — 50 qualified leads with custom proposals overnight, in the time it takes a human to write one.

---

## Architecture

src/
├── agents/
│   ├── researcher.py       Researcher Agent — finds leads from live web
│   ├── qualifier.py        Qualifier Agent — filters + scores + matches products
│   ├── writer.py           Writer Agent — drafts proposals (with hallucination guard)
│   └── manager.py          Manager Agent — LangGraph orchestration
├── tools/
│   ├── web_search.py       Tavily wrapper with exponential-backoff retry
│   └── product_database.py Heidstar product catalog query layer
├── state.py                Shared TypedDict state for the LangGraph pipeline
└── test_setup.py           Environment verification script
data/
└── heidstar_products.json  Heidstar product catalog (6 products, 3 future markets)

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | **LangGraph** | Multi-agent state-machine workflow |
| LLM | **Ollama + Qwen 3 1.7B** | Local inference — runs anywhere, no API keys |
| LLM Interface | **LangChain + langchain-openai** | OpenAI-compatible API surface (model-swappable) |
| Web Search | **Tavily** | AI-native search, structured results |
| Data Validation | **Pydantic + TypedDict** | Strong typing across the pipeline |
| Config | **python-dotenv** | Secrets management |

### Why Local LLM?

The system runs entirely on consumer hardware. **No cloud API keys for inference.** This was deliberate:

- ✅ Works in regions where cloud APIs (Gemini, OpenAI, Anthropic) are blocked
- ✅ Zero ongoing inference cost
- ✅ Privacy-safe — data never leaves the machine
- ✅ Aligns with Heidstar's expansion into regulated industries (healthcare, drug development, space biology)

Web search still requires Tavily (an HTTPS call) — but inference is fully local.

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- A Tavily API key (free tier — [get one here](https://tavily.com/))

### Install

```bash
# Clone the repo
git clone https://github.com/MISHU-KHONDOKER/heidstar-multi-agent-sales.git
cd heidstar-multi-agent-sales

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
.\venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Pull the local LLM
ollama pull qwen3:1.7b

# Create your .env file
echo "TAVILY_API_KEY=your-key-here" > .env
```

### Verify

```bash
python src/test_setup.py
```

Expected output: `🎉 All systems operational. Foundation is ready.`

### Run the Full Pipeline

```bash
python src/agents/manager.py
```

The pipeline takes 5–10 minutes end-to-end on consumer hardware. You will see each agent log its progress in real time.

---

## Example Run

**Input brief:** *"Stem cell research laboratories in Europe doing iPSC work"*

**Researcher** generates 3 diverse queries, executes 12 web searches, extracts 2 structured leads.

**Qualifier** filters false positives via heuristics, then uses the LLM to score each surviving lead 0–100 with reasoning.

**Writer** picks one of two modes per lead:
- **Product Pitch mode** — when a product was matched: writes a specific proposal using real specs.
- **Soft Introduction mode** — when no product matched: writes a relationship-opening email with no product mentions (prevents hallucination).

**Sample output proposal (truncated):**

Subject: Proposal for HDS-MSCAN-60F Fluorescence Imaging Solution
Dear Research Team,
At Heidstar Technology, we understand the demands of high-content
drug screening and FISH-based research workflows. We are pleased
to introduce the HDS-MSCAN-60F, designed to meet your exact needs:

60-slide capacity for high-throughput FISH and immunofluorescence
Submicron resolution (≤ 0.25 µm/pixel)
Multiplex fluorescence channel switching
Scanning a 15mm × 15mm area in under 3 minutes
...

---

## Engineering Highlights

This project deliberately demonstrates several production-grade patterns:

- **Two-layer hallucination defense.** Small local LLMs are prone to inventing facts. The Writer Agent uses both prompt-level structural constraints (Mode 2 has no product mentions allowed) and a post-generation regex validator that strips any product name not in the real catalog.

- **Strategy pattern in the Writer.** The agent picks one of two prompt strategies based on input data, cleanly separating the "we have a match" path from the "no match" path.

- **Exponential backoff in the Web Search tool.** SSL connection drops are common in unstable networks. The tool retries with 2s → 4s → 6s backoff.

- **Hard-filter before LLM call in the Qualifier.** Cheap deterministic keyword filtering catches obvious false positives before any LLM time is spent on them — a real cost optimization pattern.

- **Conditional edges in the LangGraph state machine.** If any stage produces no output, the graph short-circuits to `END` instead of running empty downstream nodes.

- **Strong typing on the shared state.** `PipelineState` is a `TypedDict` so every agent contract is explicit and verifiable.

---

## Known Limitations

Being honest about what is and is not in v1:

- **LLM quality.** Qwen 3 1.7B is small. The pattern works but generation quality is below cloud-grade models. Swap in `qwen3:4b`, `llama3.3`, or any OpenAI-compatible cloud model with a one-line change.
- **No agent memory across runs.** Each invocation is independent. A future version could add long-term memory (e.g., "do not pitch leads we already contacted").
- **English-only proposals.** A multilingual Writer (German, Japanese, Mandarin) is a natural next step.
- **No CRM integration.** Output is JSON + Markdown. Future versions could push to HubSpot, Salesforce, or a custom CRM.

---

## Roadmap

- [ ] v0.2 — Strip self-referential notes from proposals (small Writer polish)
- [ ] v0.3 — Multilingual Writer (en/de/jp/zh)
- [ ] v0.4 — FastAPI web interface for non-technical users
- [ ] v0.5 — Long-term memory (don't re-contact known leads)
- [ ] v1.0 — Dockerization + cloud deployment

---

## License

MIT — see [LICENSE](./LICENSE).

---

## About

Built by **Mohyminul Islam** as an applied AI portfolio project, modeling a real-world sales-automation problem for **Heidstar Technology** (海德星科技), a precision microscopy ODM partner of Zeiss based in Xiamen, China.

- 🎓 M.S. Software Engineering — Northwestern Polytechnical University, Xi'an
- 🏆 IEEE ICCBDAI 2025 — Best Oral Presentation Award (generative AI for cybersecurity)
- 🔗 - 🔗 [LinkedIn](https://www.linkedin.com/in/mohyminul-islam-mishu-9977bb115/) · [GitHub](https://github.com/MISHU-KHONDOKER)

---

*This project is a portfolio demonstration and is not officially affiliated with Heidstar Technology.*