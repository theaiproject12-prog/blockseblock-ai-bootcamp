# Feature 6: Smart Router

**Week 2 · Phase: Knowledge**

---

## What You'll Build

Feature 4 built the ingestion pipeline: documents → chunks → stored.
Feature 5 built the retrieval layer: question → embed → search → ranked chunks.
Feature 6 adds the reasoning layer that decides *what to do with the question before retrieval even begins.*

The core problem: **similarity ≠ relevance.** Feature 4 named it. Feature 5 showed it in action. Feature 6 addresses it — two ways:

1. **LLM routing** — a cheap classification call decides whether the question even needs retrieval, and how confidently
2. **Optional PageIndex path** — when the query type signals a professional document and PageIndex is enabled, bypass vector similarity entirely and use tree-based reasoning instead

You'll implement `classify_query()` and the routing logic that wires classification output to the right retrieval path. The result is a `/api/chat/smart` endpoint that returns not just an answer, but a transparent audit trail: which path was taken, how many chunks were used, and how confident the router was.

---

## Architectural Context: F4 → F5 → F6

The RAG pipeline is complete after this feature.

```
INGESTION (Feature 4)          RETRIEVAL (Feature 5)          ROUTING (Feature 6)
━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━

Document                       User question                  User question
   │                                │                               │
   ▼                                ▼                               ▼
extract_text()               embedding model               classify_query()
   │                                │                          │         │
   ▼                                ▼                   needs_retrieval  confidence
chunk_text()                  query vector               │    │    │
   │                                │                   ▼    ▼    ▼
embedding model ──────► nearest-neighbor search    [RAG] [LLM] [Hybrid]
   │                                │                   │         │
   ▼                                ▼                   └────┬────┘
stored in ChromaDB ──────► top-k chunks returned            ▼
                                    │                  LLM generation
                                    ▼                       │
                             injected into prompt           ▼
                                    │                  SmartChatResponse
                                    ▼
                               LLM answer
```

Features 1–3 gave the assistant a voice and memory. Feature 4 gave it documents. Feature 5 gave it search. Feature 6 gives it judgment.

---

## New API

```
POST /api/chat/smart
Body: { "session_id": str, "message": str }

Response: SmartChatResponse
{
  "answer": str,
  "source": "llm" | "rag" | "hybrid" | "pageindex",
  "chunks_used": int,
  "confidence": float,
  "retrieval_method": str
}
```

The `source` field is the router's decision, logged for every request. `retrieval_method` is a human-readable description of what path was taken and why.

---

## Part A: Smart Routing (Required)

### Concepts

#### The Anti-RAG Principle

Not every question needs retrieval. If a user asks "What is the capital of France?", retrieving from your document store adds latency, cost, and potential noise — and the LLM already knows the answer. Naively routing every question through RAG is a design error.

The flip side is also true: if a user asks "What does Section 4.2 of our uploaded compliance policy say about third-party vendor approvals?", answering from the LLM's training data is a hallucination risk. You need retrieval.

`classify_query()` makes this decision cheaply — a single small-model LLM call before the expensive retrieval pipeline runs.

#### Query Classification

`classify_query(query)` returns:

```python
{
  "needs_retrieval": bool,      # Does this question require document context?
  "confidence": float,          # 0.0–1.0: how sure is the classifier?
  "query_type": str             # "general" | "domain" | "professional_document" | "ambiguous"
}
```

**Query types:**

| Type | Description | Example |
|---|---|---|
| `general` | Common knowledge, no documents needed | "Explain what a neural network is" |
| `domain` | Likely requires uploaded domain docs | "What does our refund policy say?" |
| `professional_document` | Complex professional doc (financial/legal) | "What was net revenue for Q3 per the 10-K?" |
| `ambiguous` | Classifier is uncertain | "Tell me about the policy" |

#### Routing Logic

The router combines `needs_retrieval` and `confidence` into a path decision:

```
classify_query(query)
       │
       ├─ confidence > 0.6 AND needs_retrieval = True
       │        └─→  RAG path (vector search → inject → generate)
       │
       ├─ confidence > 0.6 AND needs_retrieval = False
       │        └─→  LLM direct (no retrieval, answer from training)
       │
       └─ confidence ≤ 0.6 (uncertain either way)
                └─→  Hybrid (retrieve AND answer, let LLM synthesize)
```

#### Confidence Thresholds

| Confidence range | Meaning | Action |
|---|---|---|
| > 0.6 | Classifier is confident | Trust the `needs_retrieval` decision |
| 0.4–0.6 | Borderline | Run hybrid: retrieve + generate + note uncertainty |
| < 0.4 | Low confidence | Default to hybrid; flag for review |

The 0.6 threshold is a starting point. In production you tune it against a labelled query set from your domain.

#### Hybrid Path

The hybrid path runs retrieval *and* passes the retrieved context to the LLM while also noting the uncertainty. The LLM can then use the context if helpful and discard it if not. This is the conservative default: it costs slightly more (retrieval runs even when not strictly needed) but minimises the chance of a confident wrong answer.

#### Optional PageIndex Path

When `query_type == "professional_document"` AND `needs_retrieval == True` AND the environment variable `ENABLE_PAGEINDEX=true` is set, the router bypasses vector similarity entirely and delegates to the PageIndex tree search. This is the direct response to the similarity≠relevance problem for professional documents.

```python
# Config flags (all default to False)
ENABLE_PAGEINDEX       = os.getenv("ENABLE_PAGEINDEX", "false").lower() == "true"
ENABLE_MULTI_TENANT    = os.getenv("ENABLE_MULTI_TENANT", "false").lower() == "true"
ENABLE_LONG_TERM_CONTEXT = os.getenv("ENABLE_LONG_TERM_CONTEXT", "false").lower() == "true"
```

---

> **SIMILARITY ≠ RELEVANCE: RESOLUTION**
>
> Feature 4 named the problem: vector similarity is not the same as answer relevance.
> Feature 5 showed it concretely: a chunk scoring 0.87 can fail to answer the question while a 0.61-scoring chunk answers it precisely.
>
> Feature 6 addresses it two ways:
>
> **Way 1 — LLM routing:** Before retrieval, `classify_query()` asks whether the question even needs document context. If it does, the hybrid path asks the LLM to assess the retrieved chunks, not just inject them blindly.
>
> **Way 2 — PageIndex path (optional):** For `professional_document` queries where vector similarity is structurally likely to fail (financial reports, legal filings, technical manuals), the router can bypass vector search entirely and use PageIndex's tree-reasoning approach, which achieved 98.7% accuracy on FinanceBench vs significantly lower for vector RAG.
>
> The core insight: similarity-based retrieval is a heuristic. The Smart Router adds a reasoning layer that knows when to trust the heuristic and when to bypass it.

---

> **FRAMEWORK BRIDGE**
>
> If you encounter LangChain in production, the Smart Router maps directly to established abstractions:
>
> | This module | LangChain equivalent |
> |---|---|
> | `classify_query()` | `QueryRouter` / LangChain's query routing chains |
> | Routing on `needs_retrieval` + `confidence` | `RouterChain` with conditional branches |
> | Smart Router overall | `RetrievalQAWithSourcesChain` — which also returns source attribution alongside the answer |
> | `source` field in `SmartChatResponse` | LangChain's `"source_documents"` key in the return dict |
> | Hybrid path | `MultiRetrievalQAChain` with fallback |
>
> Building this from scratch means you understand what the abstraction is doing — and can debug it when it does the wrong thing.

---

### Your Task (Part A)

**Step 1:** Implement `classify_query()` in `starter/main.py`:
- [ ] Construct a classification prompt that instructs the LLM to return JSON with `needs_retrieval`, `confidence`, and `query_type`
- [ ] Make a cheap LLM call (use a fast/small model if available — this call should be fast)
- [ ] Parse the JSON response; handle parse errors by returning `{"needs_retrieval": True, "confidence": 0.3, "query_type": "ambiguous"}` as a safe fallback
- [ ] Return the classification dict

**Step 2:** Implement the routing logic in the `/api/chat/smart` endpoint:
- [ ] Call `classify_query(message)` to get the classification
- [ ] Apply threshold logic: confidence > 0.6 → trust `needs_retrieval`; else → hybrid
- [ ] If RAG path: call `search()`, build context, generate with context
- [ ] If LLM direct path: generate without context
- [ ] If hybrid path: call `search()`, generate with context but flag uncertainty in the response
- [ ] If `ENABLE_PAGEINDEX=true` and `query_type == "professional_document"`: route to PageIndex instead of vector search
- [ ] Return `SmartChatResponse` with `answer`, `source`, `chunks_used`, `confidence`, `retrieval_method`

**Step 3:** Verify routing in the UI:
- [ ] Ask a general knowledge question — confirm `source: "llm"` and `chunks_used: 0`
- [ ] Ask a question about an uploaded document — confirm `source: "rag"` and chunks > 0
- [ ] Ask an ambiguous question — confirm `source: "hybrid"`
- [ ] Use Resource 6's worksheet to test all six query types and compare your manual classification to `classify_query()`'s output

**Step 4 (optional):** Test the PageIndex path:
- [ ] Set `ENABLE_PAGEINDEX=true` in your environment
- [ ] Upload a financial or legal PDF
- [ ] Ask a question that requires navigating document structure — confirm `source: "pageindex"`

---

### How to Run

```bash
cd week-2-knowledge/feature-6-smart-router/starter
uvicorn main:app --reload --port 8000
```

Features 1–5 work immediately. The `/api/chat/smart` endpoint returns 501 until you implement `classify_query()` and the routing logic. Open `http://localhost:8000` and use the Smart Chat tab.

To enable optional features:

```bash
# Enable PageIndex routing
ENABLE_PAGEINDEX=true uvicorn main:app --reload --port 8000

# Enable multi-tenant isolation (Part B)
ENABLE_MULTI_TENANT=true uvicorn main:app --reload --port 8000

# Enable long-term context retention (Part C)
ENABLE_LONG_TERM_CONTEXT=true uvicorn main:app --reload --port 8000
```

---

## Key Files

| File | What it does |
|------|-------------|
| `starter/main.py` | Your work: `classify_query()` + routing logic + `/api/chat/smart` |
| `shared/vector_store.py` | Vector search (Feature 5) — called by RAG and hybrid paths |
| `shared/ingestion.py` | Text extraction + chunking (Feature 4) |
| `shared/models.py` | `SmartChatResponse`, `RetrievalLogEntry`, `KnowledgeDigest` |
| `shared/config.py` | `ENABLE_PAGEINDEX`, `ENABLE_MULTI_TENANT`, `ENABLE_LONG_TERM_CONTEXT` flags |
| `solution/main.py` | Complete Feature 6 server |
| `resource/routing-decision-worksheet.md` | Resource 6: routing flowchart + worksheets for Parts A, B, C |

---

## 🏢 For Consultants

**Parts B and C are Enterprise Extensions.** They are optional, clearly labeled, and off by default. Core bootcamp students skip directly to "Week 2 Complete" below.

If you are building production systems — multi-tenant SaaS products, enterprise knowledge platforms, or long-running customer-facing assistants — these sections address the two most common failure modes that emerge after the basic RAG pipeline is working.

---

## Part B: Tenant Isolation (Enterprise Extension — Optional)

### Concepts

#### Multi-Tenancy

A multi-tenant system serves multiple customers (tenants) from a single deployment. Each tenant's data must be completely invisible to every other tenant — not just in the UI, but at the data layer.

The most common mistake in multi-tenant RAG: implementing isolation at the application layer ("only return results where tenant_id matches the user's tenant_id in the API handler"). This is insufficient.

**Why application-level filtering is not enough:**

A bug in the filter logic — a missed `WHERE` clause, an unguarded code path, a test endpoint left open — means Tenant A's documents are returned to Tenant B's query. You don't find out until a customer reports it or a security audit catches it.

**Database-level filtering** adds a second enforcement layer. In ChromaDB, the `where` parameter on `collection.query()` is evaluated by the database engine before results are returned. A bug in your API handler cannot bypass it — the wrong documents are never retrieved.

The goal is: **Tenant B's documents should never appear in Tenant A's query results, even if there is a bug in the application layer.**

#### What Changes

**Request headers:**
```
X-Tenant-ID: acme-corp
```

**Vector store query (with tenant isolation):**
```python
# Feature 5 search (no tenant isolation)
results = collection.query(
    query_texts=[query],
    n_results=top_k
)

# Feature 6 search (with tenant isolation)
results = collection.query(
    query_texts=[query],
    n_results=top_k,
    where={"tenant_id": tenant_id}   # database-level filter
)
```

**Document model:**
```python
class Document(BaseModel):
    id: str
    filename: str
    tenant_id: str          # added
    chunking_strategy: str
    chunk_count: int
    uploaded_at: datetime
```

**Session model:**
```python
class Session(BaseModel):
    id: str
    tenant_id: str          # added
    messages: list
    created_at: datetime
```

#### Tenant Isolation Endpoints

The following endpoints require the `X-Tenant-ID` header and enforce tenant scoping:

| Endpoint | Isolation behaviour |
|---|---|
| `POST /api/documents/upload` | Document stored with `tenant_id` in metadata |
| `GET /api/documents` | Returns only documents where `tenant_id` matches header |
| `POST /api/chat/smart` | Retrieval filtered by `tenant_id` at DB level |
| `GET /api/sessions/{session_id}` | Returns 403 if session `tenant_id` ≠ header `tenant_id` |
| `DELETE /api/documents/{doc_id}` | Deletes only if `tenant_id` matches; 403 otherwise |

#### Isolation Test

To prove isolation holds, run the following scenario:

1. Start the server with `ENABLE_MULTI_TENANT=true`
2. Upload Document A with header `X-Tenant-ID: tenant-alpha`
3. Upload Document B with header `X-Tenant-ID: tenant-beta`
4. Ask a question about Document A's content with header `X-Tenant-ID: tenant-beta`
5. Assert: the response contains zero chunks from Document A
6. Ask the same question with `X-Tenant-ID: tenant-alpha`
7. Assert: the response contains chunks from Document A

See Resource 6, Section 2 for the concrete pytest implementation of this test.

---

### Your Task (Part B — Optional)

- [ ] Add `tenant_id` field to `Document` and `Session` models in `shared/models.py`
- [ ] Modify the document upload endpoint to extract `X-Tenant-ID` header and store it in document metadata when `ENABLE_MULTI_TENANT=true`
- [ ] Modify `vector_store.search()` to accept an optional `tenant_id` parameter and pass it as a `where` filter
- [ ] Pass `tenant_id` from the request header to every `search()` call in the smart router
- [ ] Add a session ownership check: return 403 if `X-Tenant-ID` header doesn't match the session's `tenant_id`
- [ ] Run the isolation test (Resource 6, Section 2) and confirm it passes

---

## Part C: RAG Long-Term Context Retention (Enterprise Extension — Optional)

### Concepts

#### The Problem: Retrieval Amnesia

Feature 3 added conversation memory — the assistant remembers what was said in this session. But it has no memory of what it retrieved in past sessions. If a user asks the same question repeatedly across different sessions, the system retrieves from scratch each time — no learning, no accumulation.

For long-running assistants (customer support bots, internal knowledge bases used daily), this is a waste: the system keeps rediscovering the same high-value chunks and ignoring the same low-value ones. Over time, you could build a "retrieval intuition" — a digest of what the system has found useful.

#### Retrieval Memory: RetrievalLogEntry

Every successful retrieval is logged:

```python
class RetrievalLogEntry(BaseModel):
    id: str
    session_id: str
    query: str
    retrieved_chunks: list[str]     # chunk IDs
    retrieval_scores: list[float]   # similarity scores
    source_used: str                # "rag" | "hybrid" | "pageindex"
    timestamp: datetime
    was_helpful: bool | None        # optional user feedback signal
```

These logs accumulate across sessions. They form a history of what the system retrieved and (if you collect feedback) whether it was helpful.

#### Knowledge Digest: build_knowledge_digest()

`build_knowledge_digest()` reads the retrieval log and asks an LLM to summarise:
- Which chunks have been retrieved most frequently
- Which query patterns tend to succeed (high scores, positive feedback)
- Which query patterns tend to fail (low scores, negative feedback)
- Any patterns in what users are asking that the current documents don't cover well

```python
class KnowledgeDigest(BaseModel):
    generated_at: datetime
    top_chunks: list[str]           # most frequently retrieved chunk IDs
    query_patterns: list[str]       # recurring query themes
    coverage_gaps: list[str]        # questions the system struggled with
    summary: str                    # LLM-generated narrative digest
```

The digest is injected into the system prompt as additional context — a summary of what the system has learned about what users need.

#### How This Differs From Feature 3 Conversation Memory

| | Feature 3 conversation memory | Feature 6 retrieval memory |
|---|---|---|
| **What is remembered** | Messages in this session | Retrieval history across sessions |
| **Scope** | One session | All sessions |
| **Format** | Raw message history | Structured log + LLM digest |
| **Persistence** | Session duration | Long-term (days/weeks) |
| **Benefit** | Coherent within a conversation | System "learns" what users need |
| **Production storage** | In-memory (this course) | Database (production) |

They complement each other. Feature 3 memory makes the assistant coherent. Feature 6 retrieval memory makes it progressively more useful.

---

### Your Task (Part C — Optional)

- [ ] Implement `RetrievalLogEntry` logging in the smart router: after every retrieval, write a log entry with the query, chunk IDs, scores, and source
- [ ] Implement `build_knowledge_digest()`: read retrieval logs, construct a summarisation prompt, call the LLM, return a `KnowledgeDigest`
- [ ] Inject the digest into the system prompt when `ENABLE_LONG_TERM_CONTEXT=true`
- [ ] Add a `GET /api/knowledge-digest` endpoint that returns the current digest (useful for debugging and for showing users "what has the system learned?")
- [ ] Optional: add a feedback endpoint `POST /api/retrieval/{entry_id}/feedback` to mark a retrieval as helpful or not, and use that signal in the digest

---

> **Week 2 Complete**
>
> You've built a complete RAG pipeline:
>
> **ingest** (F4) → **chunk** (F4) → **embed** (F5) → **retrieve** (F5) → **classify** (F6) → **route** (F6) → **generate** (F1–F3)
>
> Every piece connects. Documents flow from upload to embeddings to search to classification to generation. The Smart Router is what turns a collection of retrieval components into a system that can reason about how to answer questions.
>
> Week 3 adds tools and agentic capabilities: the assistant will be able to take actions, call APIs, and coordinate multi-step workflows — not just answer questions.

---

## What's Next

Feature 7 introduces **Tool Use** — giving the assistant the ability to call external functions (APIs, calculators, code interpreters) in response to a question. The same routing intuition applies: the agent classifies what kind of response is needed, then decides whether to answer, retrieve, or act.

Resource 6 (`resource/routing-decision-worksheet.md`) covers the routing flowchart, a six-query manual classification exercise, the tenant isolation test scenario, and the long-term context design guide.
