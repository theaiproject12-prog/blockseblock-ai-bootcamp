"""
Tests for Feature 6, Part B: multi-tenant isolation.

These tests verify that tenant data is completely invisible across tenant
boundaries — no API calls, no vector DB, pure in-memory store logic.

The critical property being tested: when ENABLE_MULTI_TENANT=true, a
document, session, or search belonging to Tenant A must NEVER be accessible
by Tenant B, even if Tenant B knows the exact document ID or session ID.

Run with:
    pytest tests/test_tenant_isolation.py -v

All tests pass with zero network calls and zero LLM API keys required.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def enable_multi_tenant(monkeypatch):
    """Activate multi-tenant mode for every test in this file."""
    import shared.config as cfg
    monkeypatch.setattr(cfg.settings, "enable_multi_tenant", True)


@pytest.fixture(autouse=True)
def clean_stores():
    """Reset all in-memory stores before each test."""
    import shared.document_store as ds
    import shared.session_store as ss

    ds._documents.clear()
    ds._chunks.clear()
    ss._store.clear()
    yield
    ds._documents.clear()
    ds._chunks.clear()
    ss._store.clear()


# =============================================================================
# Document isolation
# =============================================================================

class TestDocumentIsolation:
    def test_tenant_a_cannot_read_tenant_b_document(self):
        """get_document() returns None when accessed by a different tenant."""
        from shared.document_store import get_document, save_document

        doc = save_document("secret.pdf", tenant_id="tenant-a")

        # Tenant A can read its own document
        assert get_document(doc.id, tenant_id="tenant-a") is not None

        # Tenant B cannot read Tenant A's document, even with the exact ID
        assert get_document(doc.id, tenant_id="tenant-b") is None

    def test_list_documents_filters_by_tenant(self):
        """list_documents() only returns documents owned by the given tenant."""
        from shared.document_store import list_documents, save_document

        save_document("doc-a.pdf", tenant_id="tenant-a")
        save_document("doc-a-2.pdf", tenant_id="tenant-a")
        save_document("doc-b.pdf", tenant_id="tenant-b")

        docs_a = list_documents(tenant_id="tenant-a")
        docs_b = list_documents(tenant_id="tenant-b")

        assert len(docs_a) == 2
        assert len(docs_b) == 1
        assert all(d.tenant_id == "tenant-a" for d in docs_a)
        assert all(d.tenant_id == "tenant-b" for d in docs_b)

    def test_tenant_b_cannot_delete_tenant_a_document(self):
        """delete_document() returns False and leaves data intact for cross-tenant delete."""
        from shared.document_store import delete_document, get_document, save_document

        doc = save_document("confidential.txt", tenant_id="tenant-a")

        # Tenant B tries to delete Tenant A's document — should silently fail
        deleted = delete_document(doc.id, tenant_id="tenant-b")
        assert deleted is False

        # Tenant A's document still exists
        assert get_document(doc.id, tenant_id="tenant-a") is not None

    def test_tenant_can_delete_own_document(self):
        """delete_document() returns True and removes the document for the owning tenant."""
        from shared.document_store import delete_document, get_document, save_document

        doc = save_document("mine.txt", tenant_id="tenant-a")

        deleted = delete_document(doc.id, tenant_id="tenant-a")
        assert deleted is True
        assert get_document(doc.id, tenant_id="tenant-a") is None

    def test_document_tenant_id_is_stored_on_record(self):
        """Documents carry their tenant_id so it survives through the store."""
        from shared.document_store import save_document

        doc = save_document("tagged.pdf", tenant_id="acme-corp")
        assert doc.tenant_id == "acme-corp"


# =============================================================================
# Session isolation
# =============================================================================

class TestSessionIsolation:
    def test_tenant_a_cannot_read_tenant_b_session(self):
        """get_session() returns None when accessed by a different tenant."""
        from shared.session_store import create_session, get_session

        session_id = create_session(tenant_id="tenant-a")

        # Tenant A can access its own session
        assert get_session(session_id, tenant_id="tenant-a") is not None

        # Tenant B cannot access Tenant A's session
        assert get_session(session_id, tenant_id="tenant-b") is None

    def test_session_carries_tenant_id(self):
        """Sessions store their tenant_id and it is retrievable."""
        from shared.session_store import create_session, get_session

        session_id = create_session(tenant_id="globex")
        session = get_session(session_id, tenant_id="globex")
        assert session is not None
        assert session.tenant_id == "globex"

    def test_multiple_tenants_have_separate_sessions(self):
        """Sessions from different tenants are completely separate in-memory."""
        from shared.session_store import create_session, get_session

        sid_a = create_session(tenant_id="tenant-a")
        sid_b = create_session(tenant_id="tenant-b")

        # Each tenant can only access their own session
        assert get_session(sid_a, tenant_id="tenant-a") is not None
        assert get_session(sid_b, tenant_id="tenant-b") is not None
        assert get_session(sid_a, tenant_id="tenant-b") is None
        assert get_session(sid_b, tenant_id="tenant-a") is None


# =============================================================================
# Default tenant passthrough (backward-compatibility)
# =============================================================================

class TestDefaultTenantBackcompat:
    def test_default_tenant_creates_and_retrieves_document(self):
        """When tenant_id='default', the system works exactly as in single-tenant mode."""
        from shared.document_store import get_document, save_document

        doc = save_document("readme.txt", tenant_id="default")
        retrieved = get_document(doc.id, tenant_id="default")
        assert retrieved is not None
        assert retrieved.id == doc.id

    def test_default_tenant_session_roundtrip(self):
        """Sessions created with the default tenant are accessible with it."""
        from shared.session_store import create_session, get_session

        session_id = create_session(tenant_id="default")
        session = get_session(session_id, tenant_id="default")
        assert session is not None

    def test_default_tenant_isolated_from_named_tenant(self):
        """The 'default' tenant is isolated from named tenants like any other."""
        from shared.document_store import get_document, save_document

        doc_default = save_document("shared.pdf", tenant_id="default")
        doc_acme = save_document("private.pdf", tenant_id="acme")

        # 'default' cannot see 'acme' docs and vice versa
        assert get_document(doc_default.id, tenant_id="acme") is None
        assert get_document(doc_acme.id, tenant_id="default") is None


# =============================================================================
# Multi-tenant flag disabled (all-access passthrough)
# =============================================================================

class TestMultiTenantDisabled:
    @pytest.fixture(autouse=True)
    def disable_multi_tenant(self, monkeypatch):
        """Override the class-level enable_multi_tenant fixture — disable it."""
        import shared.config as cfg
        monkeypatch.setattr(cfg.settings, "enable_multi_tenant", False)

    def test_disabled_mode_ignores_tenant_id(self):
        """When multi-tenant is off, any tenant_id can access any document."""
        from shared.document_store import get_document, save_document

        # Document saved with tenant-a
        doc = save_document("public.pdf", tenant_id="tenant-a")

        # Accessible with any tenant_id — isolation is off
        assert get_document(doc.id, tenant_id="tenant-b") is not None
        assert get_document(doc.id, tenant_id="default") is not None
        assert get_document(doc.id, tenant_id="tenant-a") is not None
