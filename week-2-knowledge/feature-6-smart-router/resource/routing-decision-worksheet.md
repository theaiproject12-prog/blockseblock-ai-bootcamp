# Resource 6: Routing Decision Worksheet

**Feature 6 · AI Engineering Bootcamp · BlockseBlock**

This worksheet covers three things: the complete routing flowchart for the Smart Router, a manual classification exercise to build intuition before you run `classify_query()`, and (for consultants) design guides for tenant isolation and long-term context retention.

---

## Section 1: The Routing Flowchart

### Complete Decision Path

```
User query
    │
    ▼
classify_query(query)
    │
    ├── needs_retrieval: bool
    ├── confidence: float (0.0–1.0)
    └── query_type: "general" | "domain" | "professional_document" | "ambiguous"
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE CHECK                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ confidence > 0.6 ──────────────────────────────────────────┐
    │       │                                                      │
    │       ├─ needs_retrieval = True                             │
    │       │       │                                             │
    │       │       ▼                                             │
    │       │  ENABLE_PAGEINDEX = true?                           │
    │       │       │                                             │
    │       │       ├─ Yes AND query_type = "professional_document"
    │       │       │       │                                     │
    │       │       │       ▼                                     │
    │       │       │  [PAGEINDEX PATH]                           │
    │       │       │  PageIndex tree search                      │
    │       │       │  source: "pageindex"                        │
    │       │       │  chunks_used: 0 (sections used instead)     │
    │       │       │                                             │
    │       │       └─ No (or other query_type)                  │
    │       │               │                                     │
    │       │               ▼                                     │
    │       │          [RAG PATH]                                 │
    │       │          vector search → top-k chunks               │
    │       │          inject into prompt → LLM generate          │
    │       │          source: "rag"                              │
    │       │                                                     │
    │       └─ needs_retrieval = False                            │
    │               │                                             │
    │               ▼                                             │
    │          [LLM DIRECT PATH]                                  │
    │          no retrieval, answer from training data            │
    │          source: "llm"                                      │
    │          chunks_used: 0                                     │
    │                                                             │
    └─ confidence ≤ 0.6 ──────────────────────────────────────────┘
            │
            ▼
       [HYBRID PATH]
       run vector search AND answer
       LLM uses context if helpful, ignores it if not
       source: "hybrid"
       chunks_used: however many were retrieved
```

### How the Four Architectures Connect Back Here

> This is how the four architectures from Feature 4 connect back to the Smart Router:
>
> - **CAG** bypasses this module entirely — small stable document sets are loaded into the system prompt before the query arrives; no routing decision needed.
> - **PageIndex** replaces vector similarity on the RAG path for professional documents — the router detects `query_type == "professional_document"` and delegates tree search instead.
> - **Vector RAG** handles everything else — the default path when retrieval is needed and PageIndex is not applicable.
> - **LLM direct** is the Anti-RAG path — the router detects that no external knowledge is needed and skips retrieval entirely.
>
> The Smart Router is the integration point for all four architectures. It does not replace them; it decides which one to invoke.

---

### Six-Query Manual Classification Exercise

**Instructions:** For each query below, fill in your manual classification *before* running it through `classify_query()`. Then run the query and record what the function returns. Compare your answer to the model's.

There are deliberately two of each type: 2 general, 2 domain-specific, 1 professional_document, 1 ambiguous.

---

**Query 1:** "What causes inflation?"

Your classification:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

`classify_query()` returned:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

Path taken: ___________

*Hint: This is a general economics question. Any LLM trained on public data knows the answer without needing your uploaded documents.*

---

**Query 2:** "Explain how transformer attention works."

Your classification:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

`classify_query()` returned:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

Path taken: ___________

*Hint: Also general knowledge. Well-documented in public ML literature. Should not trigger retrieval from your uploaded docs.*

---

**Query 3:** "What does our company's refund policy say about digital downloads?"

Your classification:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

`classify_query()` returned:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

Path taken: ___________

*Hint: "Our company's" is a strong domain signal. The answer cannot come from training data — it requires the specific uploaded policy document.*

---

**Query 4:** "According to the uploaded onboarding guide, what is the process for requesting IT equipment?"

Your classification:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

`classify_query()` returned:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

Path taken: ___________

*Hint: Explicit reference to "the uploaded onboarding guide." Strong domain signal, high confidence expected.*

---

**Query 5:** "What was the net revenue attributable to continued operations in Q3, broken down by geographic segment, as reported in the annual filing?"

Your classification:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

`classify_query()` returned:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

Path taken: ___________

*Hint: Financial document, multi-part, requires navigating to the specific segment reporting section. If ENABLE_PAGEINDEX=true, this should route to PageIndex. If not, RAG with a note that similarity-based retrieval may struggle.*

---

**Query 6:** "Tell me about the policy."

Your classification:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

`classify_query()` returned:
- `needs_retrieval`: ___________
- `confidence`: ___________
- `query_type`: ___________

Path taken: ___________

*Hint: Which policy? "Policy" appears in countless contexts. "Tell me about" is vague. Classifier should return low confidence → hybrid path. This is the ambiguous case by design.*

---

### What to Look For

After running all six queries:

1. **Where did the classifier agree with you?** These are the easy, unambiguous cases. Notice what linguistic features made them clear.

2. **Where did it disagree?** Review the prompt you wrote for `classify_query()`. Did it include enough guidance about what counts as domain-specific vs general?

3. **Where was confidence low?** Low confidence on queries 3 and 4 (which have strong domain signals) suggests your classification prompt may need more examples or clearer instructions.

4. **Query 6 test:** If the classifier returned `confidence > 0.6` on "Tell me about the policy", that's a false positive. The query is genuinely ambiguous and the classifier should reflect that. Adjust your prompt to flag vague queries appropriately.

---

## Section 2: For Consultants — Tenant Isolation Checklist

### Security Properties That Must Hold

For a multi-tenant RAG system, the following properties are non-negotiable:

| Property | Description | How to verify |
|---|---|---|
| **Retrieval isolation** | Tenant B's queries never return Tenant A's chunks | Isolation test (see below) |
| **Document visibility** | `GET /api/documents` for Tenant B lists only Tenant B's documents | API test with mismatched headers |
| **Session ownership** | Tenant B cannot read or continue Tenant A's sessions | 403 response test |
| **Delete scope** | Tenant B cannot delete Tenant A's documents | 403 response test |
| **No cross-contamination in logs** | Retrieval logs are scoped per tenant | Log query test |

### Database-Level vs Application-Level Filtering

**Application-level filtering (insufficient on its own):**

```python
# ❌ Application-level only — a bug here means a data breach
@app.post("/api/chat/smart")
async def smart_chat(request: Request, body: SmartChatBody):
    tenant_id = request.headers.get("X-Tenant-ID")
    chunks = vector_store.search(body.message, top_k=5)
    # Bug: forgot to filter by tenant_id
    # Result: chunks from all tenants returned
    ...
```

The problem: if this function has a bug — a missed parameter, a refactor that dropped the filter, a test endpoint that bypasses it — chunks from other tenants are returned. The database has no knowledge of the bug; it returns whatever is asked.

**Database-level filtering (required):**

```python
# ✅ Database-level filter — cannot be bypassed by application bugs
def search(query: str, top_k: int, tenant_id: str | None = None) -> list[dict]:
    where_filter = {}
    if tenant_id:
        where_filter["tenant_id"] = tenant_id   # enforced by ChromaDB engine

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter if where_filter else None
    )
    ...
```

The ChromaDB `where` clause is evaluated at the storage engine level. Even if the application layer has a bug that passes the wrong `tenant_id` or passes none at all, the worst that can happen with a DB-level filter is:
- Wrong `tenant_id` → wrong tenant's data (still isolated from other tenants)
- No `tenant_id` → depends on how you handle None in the search function

The correct approach: make `tenant_id` required when `ENABLE_MULTI_TENANT=true`. If it's absent, reject the request at the handler level before reaching the database.

**Why it matters in practice:**

> A bug in application-level filtering means a data breach.
> A bug that reaches database-level filtering means an error or wrong-tenant response.
>
> The difference between "we had a bug" and "we had a data breach" is whether you enforced isolation at the database layer.

### Production Patterns for Tenant Resolution

**Pattern 1: JWT claims**
```python
# Tenant extracted from JWT — no trust on the client to pass correct ID
def get_tenant_id(authorization: str = Header(...)) -> str:
    token = authorization.replace("Bearer ", "")
    claims = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return claims["tenant_id"]   # server-verified, cannot be spoofed
```

**Pattern 2: API key → tenant mapping**
```python
# API key resolves to a tenant in your database
def get_tenant_id(x_api_key: str = Header(...)) -> str:
    tenant = db.query(ApiKey).filter(ApiKey.key == x_api_key).first()
    if not tenant:
        raise HTTPException(403, "Invalid API key")
    return tenant.tenant_id
```

**Pattern 3: Subdomain routing**
```
acme.yourtool.com     → tenant_id: "acme"
betacorp.yourtool.com → tenant_id: "betacorp"
```
Tenant ID extracted from the `Host` header at the ingress/gateway level before the request reaches application code.

**Which to use:**
- Bootcamp/prototype: `X-Tenant-ID` header (what we implement — simple, explicit)
- Production API: JWT claims or API key → tenant mapping (server-verified, not client-trusted)
- Enterprise SaaS: Subdomain routing + JWT (belt and suspenders)

### The Isolation Test: Proving It Works

Run this test after implementing Part B. It proves isolation holds at the database level, not just in the happy path.

```python
# tests/test_tenant_isolation.py
import pytest
import httpx

BASE_URL = "http://localhost:8000"

@pytest.fixture
def alpha_client():
    return httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"X-Tenant-ID": "tenant-alpha"}
    )

@pytest.fixture
def beta_client():
    return httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"X-Tenant-ID": "tenant-beta"}
    )

async def test_tenant_isolation(alpha_client, beta_client):
    # Step 1: Upload a document as tenant-alpha
    alpha_doc_content = b"The Acme refund policy: all sales are final after 30 days."
    await alpha_client.post(
        "/api/documents/upload",
        files={"file": ("acme_policy.txt", alpha_doc_content, "text/plain")}
    )

    # Step 2: Upload a different document as tenant-beta
    beta_doc_content = b"BetaCorp warranty: 2-year full replacement guarantee."
    await beta_client.post(
        "/api/documents/upload",
        files={"file": ("betacorp_warranty.txt", beta_doc_content, "text/plain")}
    )

    # Step 3: tenant-beta asks a question about tenant-alpha's content
    response = await beta_client.post(
        "/api/chat/smart",
        json={"session_id": "beta-session-1", "message": "What is the refund policy after 30 days?"}
    )
    result = response.json()

    # Step 4: Assert — tenant-alpha's document must not appear
    assert result["chunks_used"] == 0 or "Acme" not in result["answer"]
    # Stronger assertion: check that no retrieved chunk contains alpha's content
    # (requires the response to include chunk text, or check retrieval logs)

    # Step 5: tenant-alpha asks the same question — should get a real answer
    response = await alpha_client.post(
        "/api/chat/smart",
        json={"session_id": "alpha-session-1", "message": "What is the refund policy after 30 days?"}
    )
    result = response.json()
    assert result["chunks_used"] > 0   # should retrieve from alpha's document
    assert "30 days" in result["answer"] or "final" in result["answer"]

async def test_session_ownership(alpha_client, beta_client):
    # tenant-alpha creates a session
    response = await alpha_client.post(
        "/api/chat/smart",
        json={"session_id": "alpha-private-session", "message": "Hello"}
    )
    assert response.status_code == 200

    # tenant-beta tries to access tenant-alpha's session
    response = await beta_client.get("/api/sessions/alpha-private-session")
    assert response.status_code == 403   # must be forbidden, not 200 or 404
```

**Run it:**
```bash
ENABLE_MULTI_TENANT=true uvicorn main:app --reload --port 8000 &
pytest tests/test_tenant_isolation.py -v
```

All tests passing means your isolation is working at the database level. If `test_tenant_isolation` passes but you know you have no `where` clause in your `search()` call, something is wrong — the test may not be testing what you think. Add a log statement to `search()` to confirm the filter is actually being applied.

---

## Section 3: For Consultants — Long-Term Context Design Guide

### When Retrieval Memory Helps vs Hurts

**Retrieval memory helps when:**

| Scenario | Why it helps |
|---|---|
| Users ask similar questions repeatedly | Digest identifies the high-value chunks upfront; retrieval is faster conceptually and the system knows what to prioritise |
| The document set is large and sparse | Users tend to need the same 20% of chunks; the digest surfaces them |
| You collect feedback signals | Marking retrievals as helpful/unhelpful creates a training signal for improving chunk quality or document coverage |
| Coverage gap identification | The digest shows what users asked that the documents couldn't answer — tells you what to add to the knowledge base |

**Retrieval memory hurts when:**

| Scenario | Why it hurts |
|---|---|
| Users ask highly varied questions | No recurring patterns to digest; memory adds overhead without benefit |
| Documents change frequently | Digests built on old retrieval patterns may reinforce chunks that are now outdated |
| The knowledge base is small (<100 chunks) | You don't need a digest to know which chunks exist |
| Injecting the digest makes prompts too long | A 2,000-token digest injected on every call adds cost and may crowd out the actual retrieved context |

The signal for "retrieval memory is helping": user questions that previously triggered hybrid or low-confidence paths start triggering high-confidence RAG paths, because the digest pre-primes the system with what the user population actually needs.

### Digest Rebuild Strategy

**How often to rebuild:**

| Trigger | Recommendation |
|---|---|
| Time-based | Rebuild daily (overnight batch job) or weekly for stable knowledge bases |
| Volume-based | Rebuild after every N new retrieval log entries (e.g., every 500 queries) |
| Event-based | Rebuild whenever the document set changes significantly (new upload, large delete) |
| On-demand | `POST /api/knowledge-digest/rebuild` endpoint for manual triggers |

**What to summarise:**

The digest LLM prompt should ask for:
1. The 10–20 most frequently retrieved chunk IDs and their topics
2. The query patterns that produced the highest average similarity scores
3. The query patterns that consistently returned low-scoring chunks (retrieval failure signal)
4. Questions asked for which no chunk scored above a threshold (coverage gaps)
5. A 2–3 paragraph narrative: "Users of this knowledge base tend to ask about X. They frequently need information about Y. The system struggles with Z."

**Avoiding stale digests:**

If you rebuild daily but documents change hourly, the digest references chunks that no longer exist. Solutions:
- Track chunk creation/deletion timestamps in the log
- Validate chunk IDs in the digest against the current collection before injecting
- Mark digests with a `valid_until` timestamp and rebuild proactively on document changes

### How This Differs From Feature 3 Conversation Memory

A common confusion: "isn't this the same as what we built in Feature 3?"

**Feature 3 conversation memory** stores the message history for the current session:
```
[User]: What is the refund policy?
[Assistant]: Our refund policy states...
[User]: What about digital downloads?
[Assistant]: For digital downloads specifically...
```

This keeps the assistant coherent within a single conversation. It is session-scoped and ephemeral.

**Feature 6 retrieval memory** stores what the system *retrieved* across all sessions:
```
Log entry 1: query="refund policy", chunks=[chunk_42, chunk_17], scores=[0.87, 0.74], session="abc123"
Log entry 2: query="returns after 30 days", chunks=[chunk_42, chunk_19], scores=[0.91, 0.68], session="def456"
Log entry 3: query="digital download refunds", chunks=[chunk_42, chunk_88], scores=[0.83, 0.71], session="ghi789"
```

After N sessions, chunk_42 appears repeatedly. The digest notes: "Users frequently need chunk_42 (refund policy — full text). The question pattern 'refund after X days' reliably retrieves it with high scores." This is cross-session, persistent, and aggregate.

| | Feature 3 | Feature 6 |
|---|---|---|
| **Scope** | One session | All sessions |
| **What's stored** | Messages | Retrieval events |
| **Format** | List of `{role, content}` | Structured `RetrievalLogEntry` + digest |
| **Benefit** | Conversational coherence | Progressive system improvement |
| **Updated** | Every message | Every retrieval |
| **Expires** | With session | Never (or explicit TTL) |
| **Injected as** | Recent message history | System-level digest in system prompt |

They work together, not instead of each other. A query comes in → conversation memory provides the recent context of this session → retrieval memory (digest) provides the system-level priming of what's most useful → retrieval runs → answer generated with both.

### Production Architecture Sketch

For a production system serving multiple tenants with long-term context:

```
                          ┌─────────────────────────────┐
                          │       Digest Store           │
                          │  (PostgreSQL or DynamoDB)    │
                          │                              │
                          │  tenant_id | digest_json     │
                          │  tenant_id | generated_at    │
                          │  tenant_id | valid_until     │
                          └──────────────┬──────────────┘
                                         │ read on request
                                         │
User query ──► Smart Router ──► [inject digest as system context]
                   │                     ↑
                   │ after retrieval      │ rebuild trigger
                   ▼                     │
          ┌─────────────────┐    ┌───────────────────┐
          │  Retrieval Log  │    │   Digest Builder  │
          │  (append-only)  │───►│  (batch job)      │
          │                 │    │  LLM summarisation│
          │  PostgreSQL or  │    │  runs nightly or  │
          │  ClickHouse for │    │  on volume trigger│
          │  analytics      │    └───────────────────┘
          └─────────────────┘
```

**Storage recommendations:**

| What | Where | Why |
|---|---|---|
| Retrieval logs | Append-only table (PostgreSQL, ClickHouse) | High write volume, analytical queries over time |
| Knowledge digests | Key-value by tenant (PostgreSQL, DynamoDB, Redis) | Fast read on every request |
| Feedback signals | Same table as retrieval log (add `feedback_score` column) | Keeps the signal next to the event |

**At scale:**
- Use a message queue (SQS, Pub/Sub) between the retrieval path and the log writer — don't block the response on log writes
- Run digest builds as async background jobs, not inline with requests
- Cache digests in Redis (TTL = rebuild interval) to avoid database reads on every query
- For multi-tenant systems: build one digest per tenant, scoped to that tenant's retrieval history

**The production signal that long-term context is working:**

Track the percentage of queries that route to `source: "rag"` vs `source: "hybrid"` over time. If the digest is working, confidence should increase as the system learns what users need — more queries should land in the high-confidence RAG path and fewer in the uncertain hybrid path.
