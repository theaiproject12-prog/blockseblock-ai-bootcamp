"""
Feature 6: Smart Router — solution

Adds an intelligence layer on top of Feature 5's semantic search. Instead of
retrieving for every query, the Smart Router first CLASSIFIES the query, then
decides whether to retrieve, use PageIndex, answer directly, or take a hybrid path.

New endpoints vs Feature 5:
  POST /api/chat/smart                  — the Smart Router chat endpoint
  GET  /api/tenant/info                 — tenant identity (Part B)
  GET  /api/retrieval-memory/recent     — recent retrieval log entries (Part C)
  POST /api/retrieval-memory/rebuild    — rebuild knowledge digest (Part C)
  GET  /api/retrieval-memory/digest     — get current knowledge digest (Part C)

Updated endpoints vs Feature 5:
  POST /api/sessions                    — now creates session with tenant_id (Part B)
  POST /api/sessions/{id}/chat          — now stores session with tenant isolation (Part B)
  POST /api/documents/upload            — now tags document with tenant_id (Part B)
  GET  /api/documents                   — now filters by tenant (Part B)
  DELETE /api/documents/{id}            — now enforces tenant ownership (Part B)

All Feature 1-5 endpoints remain unchanged.

ROUTING LOGIC:
  confidence > 0.6 + needs_retrieval=True  → RAG (or PageIndex for professional docs)
  confidence > 0.6 + needs_retrieval=False → LLM direct (no retrieval cost)
  confidence 0.4–0.6                        → hybrid (retrieve but flag uncertainty)

Run with:
    uvicorn main:app --reload --port 8000

Feature flags (in .env):
    ENABLE_PAGEINDEX=true        — activate PageIndex routing for professional docs
    ENABLE_MULTI_TENANT=true     — activate X-Tenant-ID header isolation (Part B)
    ENABLE_LONG_TERM_CONTEXT=true — inject KnowledgeDigest into system prompt (Part C)
"""
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.document_store import (
    delete_document,
    get_chunks,
    get_document,
    list_documents,
    save_chunk,
    save_document,
    update_document,
)
from shared.ingestion import CHUNKING_STRATEGIES, extract_pages, extract_text
from shared.llm_client import call_llm
from shared.models import (
    Chunk,
    Document,
    Message,
    SmartChatResponse,
    StructuredResponse,
)
from shared.provider_check import check_provider_config
from shared.retrieval_memory import (
    build_knowledge_digest,
    get_current_digest,
    get_recent_retrievals,
    log_retrieval,
)
from shared.router import classify_query
from shared.session_store import add_message, create_session, get_session, list_sessions
from shared.tenant_context import get_tenant_id
from shared.vector_store import (
    add_chunks,
    delete_document_chunks,
    get_stats as vector_get_stats,
    search as vector_search,
)

CONTEXT_WINDOW_SIZE = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before the server begins accepting requests."""
    await check_provider_config()
    yield


app = FastAPI(
    title="My AI BlockSeBlock Assistant",
    description="Domain-Specific AI Assistant — AI Engineering Bootcamp, BlockseBlock",
    version="6.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    """The body expected by plain/structured chat endpoints."""
    message: str


class ChatResponse(BaseModel):
    """The body returned by POST /api/chat (plain text mode)."""
    response: str


class SessionSummary(BaseModel):
    """A lightweight session descriptor for the sidebar list."""
    id: str
    created_at: str
    message_count: int
    title: str


class SearchRequest(BaseModel):
    """Body for POST /api/search."""
    query: str
    top_k: int = 5
    document_id: str | None = None


class SmartChatRequest(BaseModel):
    """Body for POST /api/chat/smart."""
    message: str


# ---------------------------------------------------------------------------
# Feature 1: Plain chat  (kept verbatim — convention #3)
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message and get a plain-text reply (stateless — no history)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant for [YOUR_DOMAIN]. "
                "Answer clearly and concisely. "
                "If you don't know something, say so honestly rather than guessing."
            ),
        },
        {"role": "user", "content": request.message},
    ]
    result = await call_llm(messages)
    return ChatResponse(response=result.content or "")


# ---------------------------------------------------------------------------
# Feature 2: Structured chat  (kept verbatim — convention #3)
# ---------------------------------------------------------------------------

_STRUCTURED_SYSTEM_PROMPT = """You are a helpful AI assistant for [YOUR_DOMAIN].

For every user message, respond ONLY with a JSON object (no markdown, no extra text)
with exactly these four fields:

{
  "intent": "<one of: general_question | domain_question | action_request | unclear>",
  "answer": "<your response to the user, written in plain English>",
  "confidence": <a number between 0.0 and 1.0 representing how sure you are>,
  "sources_needed": <true if domain documents would improve this answer, false otherwise>
}

Intent definitions:
- "general_question": factual/knowledge query not specific to [YOUR_DOMAIN]
- "domain_question": a question specifically about [YOUR_DOMAIN] and its offerings
- "action_request": the user wants something DONE (book, schedule, find, order, send…)
- "unclear": ambiguous, nonsensical, or doesn't fit the other categories

Confidence guidelines:
- 0.9–1.0: you are certain (common knowledge, clear domain fact)
- 0.6–0.8: you are reasonably sure but the user should verify
- 0.3–0.5: you are uncertain; the answer may be incomplete or partly guessed
- 0.0–0.2: you don't know and are mostly guessing

Respond ONLY with the JSON object. No preamble, no explanation, no markdown fences."""


def _parse_structured(raw_text: str) -> StructuredResponse:
    """Parse raw LLM text into a StructuredResponse, with a safe fallback."""
    try:
        return StructuredResponse(**json.loads(raw_text))
    except Exception:
        return StructuredResponse(
            intent="unclear",
            answer=raw_text or "The assistant returned an unexpected response.",
            confidence=0.0,
            sources_needed=False,
        )


@app.post("/api/chat/structured", response_model=StructuredResponse)
async def chat_structured(request: ChatRequest) -> StructuredResponse:
    """Send a message and receive a structured response (stateless — no history)."""
    messages = [
        {"role": "system", "content": _STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": request.message},
    ]
    result = await call_llm(messages, temperature=0.3, response_format={"type": "json_object"})
    return _parse_structured(result.content or "")


# ---------------------------------------------------------------------------
# Feature 3: Session management  (updated — tenant-aware in Part B)
# ---------------------------------------------------------------------------

@app.post("/api/sessions")
async def new_session(tenant_id: str = Depends(get_tenant_id)) -> dict:
    """Start a new conversation session, tagged with the tenant (Part B)."""
    session_id = create_session(tenant_id=tenant_id)
    return {"session_id": session_id}


@app.get("/api/sessions", response_model=list[SessionSummary])
async def sessions_list() -> list[SessionSummary]:
    """List all sessions, most recent first."""
    summaries = []
    for s in list_sessions():
        first_user_msg = next((m.content for m in s.messages if m.role == "user"), "")
        title = (first_user_msg[:60] + "…") if len(first_user_msg) > 60 else (first_user_msg or "New conversation")
        summaries.append(SessionSummary(
            id=s.id,
            created_at=s.created_at.isoformat(),
            message_count=len(s.messages),
            title=title,
        ))
    return summaries


@app.post("/api/sessions/{session_id}/chat", response_model=StructuredResponse)
async def session_chat(
    session_id: str,
    request: ChatRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> StructuredResponse:
    """Send a message within a session and get a structured reply."""
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    messages: list[dict] = [{"role": "system", "content": _STRUCTURED_SYSTEM_PROMPT}]
    history = session.messages
    if len(history) > CONTEXT_WINDOW_SIZE:
        history = history[-CONTEXT_WINDOW_SIZE:]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})

    add_message(session_id, "user", request.message)
    result = await call_llm(messages, temperature=0.3, response_format={"type": "json_object"})
    structured = _parse_structured(result.content or "")
    add_message(session_id, "assistant", structured.answer)
    return structured


@app.get("/api/sessions/{session_id}/history", response_model=list[Message])
async def session_history(
    session_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> list[Message]:
    """Return the full message history for a session."""
    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.messages


# ---------------------------------------------------------------------------
# Feature 4: Document ingestion  (updated — tenant-aware in Part B)
# ---------------------------------------------------------------------------

@app.post("/api/documents/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("sentence"),
    tenant_id: str = Depends(get_tenant_id),
) -> Document:
    """
    Upload a document, extract text, chunk it, store chunks, and index vectors.

    Tags the document with the tenant_id (Part B) so only this tenant can
    retrieve it when ENABLE_MULTI_TENANT=true.

    Form fields:
      file     — the uploaded file (.txt, .pdf, .docx)
      strategy — chunking strategy: "sentence" (default) or "paragraph"
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{strategy}'. Choose: {', '.join(CHUNKING_STRATEGIES)}.",
        )

    filename = file.filename or "unknown"
    doc = save_document(filename, tenant_id=tenant_id)

    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, filename)
        pages = extract_pages(file_bytes, filename)

        chunk_dicts = CHUNKING_STRATEGIES[strategy](text, pages)

        for cd in chunk_dicts:
            save_chunk(Chunk(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                text=cd["text"],
                chunk_index=cd["chunk_index"],
                metadata={
                    "filename": filename,
                    "chunk_index": cd["chunk_index"],
                    "page_number": cd.get("page_number"),
                    "strategy": strategy,
                },
            ))

        chunk_texts = [cd["text"] for cd in chunk_dicts]
        chunk_metas = [
            {"filename": filename, "chunk_index": cd["chunk_index"], "strategy": strategy}
            for cd in chunk_dicts
        ]
        add_chunks(doc.id, chunk_texts, chunk_metas, tenant_id=tenant_id)

        update_document(
            doc.id,
            status="ready",
            chunk_count=len(chunk_dicts),
            chunking_strategy=strategy,
        )
    except Exception as exc:
        update_document(doc.id, status="error", chunk_count=0, chunking_strategy=strategy)
        raise HTTPException(status_code=422, detail=str(exc))

    return get_document(doc.id, tenant_id=tenant_id)  # type: ignore[return-value]


@app.get("/api/documents", response_model=list[Document])
async def documents_list(tenant_id: str = Depends(get_tenant_id)) -> list[Document]:
    """List all uploaded documents for this tenant, most recently uploaded first."""
    return list_documents(tenant_id=tenant_id)


@app.delete("/api/documents/{doc_id}")
async def remove_document(
    doc_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """Delete a document, its chunks, and its vectors. Enforces tenant ownership."""
    if get_document(doc_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    delete_document_chunks(doc_id)
    delete_document(doc_id, tenant_id=tenant_id)
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/chunks", response_model=list[Chunk])
async def document_chunks(
    doc_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> list[Chunk]:
    """Return all chunks for a document — useful for inspecting chunk quality."""
    if get_document(doc_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return get_chunks(doc_id)


# ---------------------------------------------------------------------------
# Feature 5: Semantic search  (updated — tenant-isolated search in Part B)
# ---------------------------------------------------------------------------

@app.post("/api/search")
async def search_documents(
    req: SearchRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> list[dict]:
    """
    Semantic search across indexed document chunks for this tenant.

    When ENABLE_MULTI_TENANT=true, only chunks belonging to this tenant
    are returned — cross-tenant isolation is enforced at the vector DB level.
    """
    filters = {"document_id": req.document_id} if req.document_id else None
    return vector_search(req.query, top_k=req.top_k, filters=filters, tenant_id=tenant_id)


@app.get("/api/search/stats")
async def search_stats() -> dict:
    """Vector store statistics: total vectors, documents indexed, embedding model."""
    return vector_get_stats()


# ---------------------------------------------------------------------------
# Feature 6: Smart Router  (Part A — required)
# ---------------------------------------------------------------------------

_SMART_SYSTEM_PROMPT = """You are a helpful AI assistant for [YOUR_DOMAIN].
Answer clearly and concisely in plain English.
If you don't know something, say so honestly rather than guessing."""

_SMART_RAG_SYSTEM_PROMPT = """You are a helpful AI assistant for [YOUR_DOMAIN].
You have been given relevant excerpts from documents to help answer the user's question.
Use the provided context to give an accurate, grounded answer.
If the context doesn't contain enough information, say so and answer from general knowledge where appropriate.
Cite the source document when quoting directly."""

_SMART_HYBRID_SYSTEM_PROMPT = """You are a helpful AI assistant for [YOUR_DOMAIN].
Some potentially relevant document excerpts have been retrieved for context, but their
relevance to this specific question is uncertain. Use them if helpful; ignore them if not.
Answer honestly and flag if you are uncertain."""


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    lines = ["--- RETRIEVED CONTEXT ---"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"\n[{i}] From: {chunk.get('filename', 'unknown')} (chunk {chunk.get('chunk_index', 0)})")
        lines.append(chunk.get("text", ""))
    lines.append("--- END CONTEXT ---")
    return "\n".join(lines)


@app.post("/api/sessions/{session_id}/chat/smart", response_model=SmartChatResponse)
async def smart_chat(
    session_id: str,
    request: SmartChatRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> SmartChatResponse:
    """
    Smart Router chat — classifies the query before deciding how to answer.

    Routing paths:
      confidence > 0.6 + needs_retrieval=True  → RAG (or PageIndex if enabled)
      confidence > 0.6 + needs_retrieval=False → LLM direct (no retrieval)
      confidence 0.4–0.6 (ambiguous)           → hybrid (retrieve + flag uncertainty)

    Part B: requests are scoped to tenant_id from X-Tenant-ID header.
    Part C: injects KnowledgeDigest into system prompt when ENABLE_LONG_TERM_CONTEXT=true.
    """
    from shared.config import settings

    session = get_session(session_id, tenant_id=tenant_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    # --- Step 1: Classify the query (Anti-RAG pre-retrieval step) ---
    classification = await classify_query(request.message)
    needs_retrieval = classification["needs_retrieval"]
    confidence = classification["confidence"]
    query_type = classification["query_type"]

    # --- Step 2: Route based on classification ---
    chunks_used: list[dict] = []
    source: str
    retrieval_method: str = "none"
    system_prompt: str

    high_confidence = confidence > 0.6
    low_confidence = confidence <= 0.6  # 0.4–0.6 range → hybrid

    if high_confidence and needs_retrieval:
        # RAG path — or PageIndex for professional documents
        if query_type == "professional_document" and settings.enable_pageindex:
            # PageIndex routing (ENABLE_PAGEINDEX=true required)
            # PageIndex provides reasoning-based tree navigation instead of
            # vector similarity — critical for financial/legal documents.
            # Integration: pip install pageindex (github.com/VectifyAI/PageIndex)
            #
            # from pageindex import PageIndex
            # pi = PageIndex.load(tree_json_path)
            # result = pi.retrieve(request.message)
            # chunks_used = [{"text": result.text, "filename": result.source,
            #                  "chunk_index": 0, "score": 1.0, "document_id": ""}]
            source = "pageindex"
            retrieval_method = "pageindex"
            # Fallback: use vector search if PageIndex is not fully configured
            chunks_used = vector_search(request.message, top_k=5, tenant_id=tenant_id)
        else:
            chunks_used = vector_search(request.message, top_k=5, tenant_id=tenant_id)
            source = "rag"
            retrieval_method = "vector"
        system_prompt = _SMART_RAG_SYSTEM_PROMPT

    elif high_confidence and not needs_retrieval:
        # LLM direct path — answer without retrieval
        source = "llm"
        retrieval_method = "none"
        system_prompt = _SMART_SYSTEM_PROMPT

    else:
        # Hybrid path — confidence 0.4–0.6: retrieve but flag uncertainty
        chunks_used = vector_search(request.message, top_k=3, tenant_id=tenant_id)
        source = "hybrid"
        retrieval_method = "vector"
        system_prompt = _SMART_HYBRID_SYSTEM_PROMPT

    # --- Step 3: Optionally inject KnowledgeDigest (Part C) ---
    if settings.enable_long_term_context:
        digest = get_current_digest(tenant_id)
        if digest and digest.summary:
            system_prompt += f"\n\nContext about this user's history: {digest.summary}"

    # --- Step 4: Build message list with session history ---
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if chunks_used:
        context_block = _build_context_block(chunks_used)
        messages.append({"role": "system", "content": context_block})

    history = session.messages[-CONTEXT_WINDOW_SIZE:]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})

    # --- Step 5: Generate the response ---
    result = await call_llm(messages)
    answer = result.content or ""

    # --- Step 6: Persist the exchange to session history ---
    add_message(session_id, "user", request.message)
    add_message(session_id, "assistant", answer)

    # --- Step 7: Log retrieval event for Part C long-term memory ---
    if chunks_used and settings.enable_long_term_context:
        chunk_ids = [
            f"{c.get('document_id', '')}_{c.get('chunk_index', 0)}"
            for c in chunks_used
        ]
        log_retrieval(
            session_id=session_id,
            tenant_id=tenant_id,
            query=request.message,
            chunk_ids=chunk_ids,
            retrieval_method=retrieval_method,
        )

    return SmartChatResponse(
        answer=answer,
        source=source,  # type: ignore[arg-type]
        chunks_used=chunks_used,
        confidence=confidence,
        retrieval_method=retrieval_method,
    )


# ---------------------------------------------------------------------------
# Feature 6: Tenant info  (Part B — optional)
# ---------------------------------------------------------------------------

@app.get("/api/tenant/info")
async def tenant_info(tenant_id: str = Depends(get_tenant_id)) -> dict:
    """
    Return the active tenant identity and multi-tenant mode status.

    Useful for the UI Admin panel to show which tenant's data is visible
    and to confirm that the X-Tenant-ID header is being read correctly.
    """
    from shared.config import settings

    docs = list_documents(tenant_id=tenant_id)
    return {
        "tenant_id": tenant_id,
        "multi_tenant_enabled": settings.enable_multi_tenant,
        "document_count": len(docs),
    }


# ---------------------------------------------------------------------------
# Feature 6: Retrieval memory  (Part C — optional)
# ---------------------------------------------------------------------------

@app.get("/api/retrieval-memory/recent")
async def retrieval_memory_recent(
    limit: int = 20,
    tenant_id: str = Depends(get_tenant_id),
) -> list[dict]:
    """
    Return the most recent retrieval log entries for this tenant.

    Each entry records: session_id, query, chunk_ids retrieved, timestamp,
    and retrieval_method. Together these entries form the raw material for
    build_knowledge_digest().
    """
    entries = get_recent_retrievals(tenant_id=tenant_id, limit=limit)
    return [
        {
            "session_id": e.session_id,
            "query": e.query,
            "chunks_retrieved": e.chunks_retrieved,
            "timestamp": e.timestamp.isoformat(),
            "retrieval_method": e.retrieval_method,
        }
        for e in entries
    ]


@app.post("/api/retrieval-memory/rebuild")
async def retrieval_memory_rebuild(
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Rebuild the KnowledgeDigest from this tenant's retrieval history.

    Makes one LLM call to summarize the recent retrieval patterns into a
    2-3 sentence digest. The result is cached in memory and injected into
    the Smart Router's system prompt on subsequent requests.

    Returns the digest summary, topics, and how many sessions it covers.
    """
    digest = await build_knowledge_digest(tenant_id=tenant_id)
    if digest is None:
        return {"message": "No retrieval history found for this tenant — upload documents and ask questions first."}
    return {
        "summary": digest.summary,
        "topics_covered": digest.topics_covered,
        "source_session_count": digest.source_session_count,
        "last_updated": digest.last_updated.isoformat(),
    }


@app.get("/api/retrieval-memory/digest")
async def retrieval_memory_digest(
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Return the current cached KnowledgeDigest for this tenant.

    Returns a 404-style response (not an HTTP error) if no digest has been
    built yet — call POST /api/retrieval-memory/rebuild first.
    """
    digest = get_current_digest(tenant_id=tenant_id)
    if digest is None:
        return {"message": "No digest built yet. Call POST /api/retrieval-memory/rebuild first."}
    return {
        "summary": digest.summary,
        "topics_covered": digest.topics_covered,
        "source_session_count": digest.source_session_count,
        "last_updated": digest.last_updated.isoformat(),
    }


# ---------------------------------------------------------------------------
# Health + provider info
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Quick liveness check — returns 200 OK if the server is running."""
    return {"status": "ok"}


@app.get("/api/provider-info")
async def provider_info():
    """Return which LLM and voice provider are currently active (no API keys)."""
    from shared.config import settings

    voice_name = settings.effective_voice_provider().lower().strip()
    llm_name = settings.llm_provider.lower().strip()
    model_map = {
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
        "cohere": settings.cohere_model,
        "ollama": settings.ollama_model,
        "groq": settings.groq_model,
        "azure": settings.azure_openai_deployment_name,
        "bedrock": settings.bedrock_model_id,
        "vertex": settings.vertex_model,
        "custom": settings.custom_model,
    }
    return {
        "llm_provider": llm_name,
        "llm_model": model_map.get(llm_name, "unknown"),
        "voice_provider": voice_name if voice_name != llm_name else None,
        "voice_model": model_map.get(voice_name) if voice_name != llm_name else None,
    }


_ui_path = Path(__file__).resolve().parents[3] / "ui"
if _ui_path.exists():
    app.mount("/", StaticFiles(directory=str(_ui_path), html=True), name="ui")
