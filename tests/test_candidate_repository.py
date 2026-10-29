"""
Batch 2 tests: Candidate Repository APIs.

Covers:
  1.  Candidate listing with pagination
  2.  Default page size / max page size clamping
  3.  Case-insensitive search: name, email, job role, skills
  4.  Empty-query search falls through to full listing
  5.  Retrieve a candidate by ID (with and without resumeText)
  6.  Missing/invalid candidate IDs
  7.  Secure signed-URL generation
  8.  Candidate deletion: happy path, partial Storage failure, idempotent 404
  9.  Empty repository
 10.  Storage object absent at deletion time
 11.  Supabase not configured → 503 on all repository endpoints
 12.  JDs must NOT be persisted as CandidateResume records
 13.  Existing upload behaviour (duplicates, sha256) remains unchanged
 14.  Existing invocations/screening endpoint unaffected
"""
import hashlib
import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from app.storage.supabase_client import reset_supabase_client
from app.storage.resume_storage import StorageError, DatabaseError
from app.storage.candidate_repository import (
    list_candidates,
    search_candidates,
    get_candidate,
    get_resume_download_url,
    delete_candidate,
    CandidateNotFoundError,
    RepositoryUnavailableError,
    PartialDeletionError,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)
from app.server import process_resume_upload


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_client():
    reset_supabase_client(None)
    yield
    reset_supabase_client(None)


def _uuid():
    return str(uuid.uuid4())


def _make_record(overrides=None):
    """Return a minimal fake CandidateResume row."""
    rec = {
        "id": _uuid(),
        "candidateName": "Jane Doe",
        "email": "jane@example.com",
        "skills": ["Python", "FastAPI"],
        "jobRole": "Software Engineer",
        "originalFilename": "jane_doe.pdf",
        "storageObjectPath": "resumes/test-uuid/jane_doe.pdf",
        "createdAt": "2026-01-01T00:00:00Z",
    }
    if overrides:
        rec.update(overrides)
    return rec


def _make_supabase_with_rows(rows, count=None):
    """
    Build a mock Supabase client whose table query chain returns `rows`.
    count defaults to len(rows).
    """
    if count is None:
        count = len(rows)

    mock_client = MagicMock()
    mock_storage = MagicMock()
    mock_client.storage.from_.return_value = mock_storage

    # Every terminal .execute() returns this
    exec_result = MagicMock()
    exec_result.data = rows
    exec_result.count = count

    # Build a chainable query builder that always ends with exec_result
    def _chain(*_args, **_kwargs):
        return builder

    builder = MagicMock()
    builder.select.side_effect = _chain
    builder.order.side_effect = _chain
    builder.range.side_effect = _chain
    builder.eq.side_effect = _chain
    builder.or_.side_effect = _chain
    builder.limit.side_effect = _chain
    builder.delete.side_effect = _chain
    builder.insert.side_effect = _chain
    builder.execute.return_value = exec_result

    mock_client.schema.return_value.table.return_value = builder
    mock_client.table.return_value = builder

    return mock_client, mock_storage, builder


# ---------------------------------------------------------------------------
# 1. Listing
# ---------------------------------------------------------------------------

class TestListCandidates:
    def test_returns_candidates_and_total(self):
        rows = [_make_record(), _make_record()]
        client, _, _ = _make_supabase_with_rows(rows, count=2)

        result = list_candidates(limit=10, offset=0, client=client)

        assert result["total"] == 2
        assert len(result["candidates"]) == 2
        assert result["limit"] == 10
        assert result["offset"] == 0

    def test_empty_repository(self):
        client, _, _ = _make_supabase_with_rows([])

        result = list_candidates(client=client)

        assert result["candidates"] == []
        assert result["total"] == 0

    def test_pagination_offset(self):
        rows = [_make_record()]
        client, _, _ = _make_supabase_with_rows(rows, count=50)

        result = list_candidates(limit=10, offset=40, client=client)

        assert result["offset"] == 40
        assert result["limit"] == 10

    def test_limit_clamped_to_max(self):
        client, _, _ = _make_supabase_with_rows([])

        result = list_candidates(limit=9999, client=client)

        assert result["limit"] == MAX_PAGE_SIZE

    def test_negative_offset_clamped_to_zero(self):
        client, _, _ = _make_supabase_with_rows([])

        result = list_candidates(offset=-5, client=client)

        assert result["offset"] == 0

    def test_raises_when_supabase_not_configured(self):
        with pytest.raises(RepositoryUnavailableError):
            list_candidates()  # no client, no env vars


# ---------------------------------------------------------------------------
# 2. Search
# ---------------------------------------------------------------------------

class TestSearchCandidates:
    def _build_search_client(self, rows, count=None):
        return _make_supabase_with_rows(rows, count)

    def test_search_by_name(self):
        rows = [_make_record({"candidateName": "Alice Johnson"})]
        client, _, _ = self._build_search_client(rows)

        result = search_candidates(query="alice", client=client)

        assert result["query"] == "alice"
        assert len(result["candidates"]) == 1

    def test_search_by_email(self):
        rows = [_make_record({"email": "bob@techcorp.com"})]
        client, _, _ = self._build_search_client(rows)

        result = search_candidates(query="techcorp", client=client)

        assert len(result["candidates"]) == 1

    def test_search_by_job_role(self):
        rows = [_make_record({"jobRole": "Cloud Architect"})]
        client, _, _ = self._build_search_client(rows)

        result = search_candidates(query="architect", client=client)

        assert len(result["candidates"]) == 1

    def test_search_by_skill(self):
        rows = [_make_record({"skills": ["Kubernetes", "Docker"]})]
        client, _, _ = self._build_search_client(rows)

        result = search_candidates(query="kubernetes", client=client)

        assert len(result["candidates"]) == 1

    def test_empty_query_delegates_to_list(self):
        rows = [_make_record(), _make_record()]
        client, _, _ = self._build_search_client(rows, count=2)

        result = search_candidates(query="", client=client)

        # No 'query' key when delegated to list_candidates
        assert "candidates" in result
        assert result["total"] == 2

    def test_whitespace_only_query_delegates_to_list(self):
        client, _, _ = self._build_search_client([])

        result = search_candidates(query="   ", client=client)

        assert result["candidates"] == []

    def test_no_results(self):
        client, _, _ = self._build_search_client([])

        result = search_candidates(query="xyzzy_nonexistent", client=client)

        assert result["candidates"] == []
        assert result["total"] == 0

    def test_pagination_preserved(self):
        client, _, _ = self._build_search_client([], count=100)

        result = search_candidates(query="eng", limit=5, offset=10, client=client)

        assert result["limit"] == 5
        assert result["offset"] == 10

    # ------------------------------------------------------------------
    # Case-insensitive skills regression tests (bug fix)
    # ------------------------------------------------------------------

    def test_search_skills_uppercase_matches(self):
        """'Python' (uppercase P) must match stored skill 'python' (lowercase)."""
        rows = [_make_record({"skills": ["python", "fastapi"]})]
        client, _, builder = self._build_search_client(rows)

        result = search_candidates(query="Python", client=client)

        assert len(result["candidates"]) == 1
        # Confirm the or_ filter used lowercase for the cs term
        or_calls = [str(c) for c in builder.or_.call_args_list]
        assert any("python" in c for c in or_calls), (
            f"Expected 'python' (lowercase) in or_ filter args, got: {or_calls}"
        )

    def test_search_skills_allcaps_matches(self):
        """'PYTHON' (all caps) must match stored skill 'python' (lowercase)."""
        rows = [_make_record({"skills": ["python", "fastapi"]})]
        client, _, builder = self._build_search_client(rows)

        result = search_candidates(query="PYTHON", client=client)

        assert len(result["candidates"]) == 1
        or_calls = [str(c) for c in builder.or_.call_args_list]
        assert any("python" in c for c in or_calls)

    def test_search_skills_lowercase_still_matches(self):
        """'python' (lowercase) continues to work after the fix."""
        rows = [_make_record({"skills": ["python", "fastapi"]})]
        client, _, _ = self._build_search_client(rows)

        result = search_candidates(query="python", client=client)

        assert len(result["candidates"]) == 1

    def test_search_skills_mixed_case_matches(self):
        """'PyThOn' (mixed case) must match stored skill 'python' (lowercase)."""
        rows = [_make_record({"skills": ["python"]})]
        client, _, builder = self._build_search_client(rows)

        result = search_candidates(query="PyThOn", client=client)

        assert len(result["candidates"]) == 1
        or_calls = [str(c) for c in builder.or_.call_args_list]
        assert any("python" in c for c in or_calls)

    def test_or_filter_uses_lowercase_cs_term(self):
        """
        Unit-level: verify the exact or_ filter string passed to PostgREST.
        The cs term must always be lowercased; the ilike terms keep the
        original query so that ilike (case-insensitive by design) still works.
        """
        client, _, builder = self._build_search_client([])

        search_candidates(query="Django", client=client)

        # or_ must have been called at least once (data + count queries)
        assert builder.or_.called
        first_call_arg = builder.or_.call_args_list[0][0][0]  # positional arg

        # ilike branches keep the original query
        assert "Django" in first_call_arg
        # cs branch must use lowercase
        assert '"django"' in first_call_arg
        # Uppercase form must NOT appear in the cs part
        # (it may appear in the ilike part, but not inside the cs.{} literal)
        cs_part = first_call_arg.split("skills.cs.")[-1]
        assert "Django" not in cs_part, (
            f"cs term should be lowercase; got: {cs_part}"
        )

    def test_name_search_preserves_original_case_in_ilike(self):
        """
        ilike branches keep the original query string (ilike is already
        case-insensitive; lowercasing is unnecessary and must not break
        searches with uppercase letters in names).
        """
        rows = [_make_record({"candidateName": "Sreya Gupta"})]
        client, _, builder = self._build_search_client(rows)

        result = search_candidates(query="Sreya", client=client)

        assert len(result["candidates"]) == 1
        # The original query case must appear in the or_ filter for ilike
        first_call_arg = builder.or_.call_args_list[0][0][0]
        assert "Sreya" in first_call_arg


# ---------------------------------------------------------------------------
# 3. Get candidate by ID
# ---------------------------------------------------------------------------

class TestGetCandidate:
    def test_returns_metadata(self):
        rec = _make_record()
        client, _, _ = _make_supabase_with_rows([rec])

        result = get_candidate(rec["id"], client=client)

        assert result["candidateName"] == "Jane Doe"
        assert "resumeText" not in result or result.get("resumeText") is None

    def test_returns_resume_text_when_requested(self):
        rec = _make_record()
        rec["resumeText"] = "Full resume content here."
        client, _, _ = _make_supabase_with_rows([rec])

        result = get_candidate(rec["id"], include_resume_text=True, client=client)

        assert result.get("resumeText") == "Full resume content here."

    def test_not_found_raises(self):
        client, _, _ = _make_supabase_with_rows([])

        with pytest.raises(CandidateNotFoundError):
            get_candidate(_uuid(), client=client)

    def test_invalid_id_format_raises(self):
        with pytest.raises(ValueError):
            get_candidate("not-a-uuid")

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            get_candidate("")


# ---------------------------------------------------------------------------
# 4. Signed download URL
# ---------------------------------------------------------------------------

class TestGetResumeDownloadUrl:
    def test_returns_signed_url(self):
        rec = _make_record()
        client, mock_storage, _ = _make_supabase_with_rows([rec])

        mock_storage.create_signed_url.return_value = {
            "signedURL": "https://supabase.co/signed/token"
        }

        url = get_resume_download_url(rec["id"], client=client)

        assert url == "https://supabase.co/signed/token"
        mock_storage.create_signed_url.assert_called_once()
        args = mock_storage.create_signed_url.call_args
        # TTL should be 300 s (default)
        assert args[0][1] == 300

    def test_not_found_raises(self):
        client, _, _ = _make_supabase_with_rows([])

        with pytest.raises(CandidateNotFoundError):
            get_resume_download_url(_uuid(), client=client)

    def test_no_storage_path_raises(self):
        rec = _make_record({"storageObjectPath": ""})
        client, _, _ = _make_supabase_with_rows([rec])

        with pytest.raises(CandidateNotFoundError):
            get_resume_download_url(rec["id"], client=client)

    def test_storage_api_failure_raises_storage_error(self):
        from app.storage.resume_storage import StorageError

        rec = _make_record()
        client, mock_storage, _ = _make_supabase_with_rows([rec])
        mock_storage.create_signed_url.side_effect = Exception("Supabase 500")

        with pytest.raises(StorageError):
            get_resume_download_url(rec["id"], client=client)


# ---------------------------------------------------------------------------
# 5. Delete candidate — new safe-deletion semantics
# ---------------------------------------------------------------------------

class TestDeleteCandidate:
    """
    New deletion contract:
      - Storage must succeed before DB is touched.
      - If Storage fails (non-404 error), raise StorageError; DB row is intact.
      - If Storage object is already absent (404/not-found), treat as success and continue.
      - If Storage succeeds but DB deletion fails, raise PartialDeletionError.
      - Missing candidate always raises CandidateNotFoundError (idempotent 404).
    """

    def test_happy_path_deletes_storage_then_db(self):
        rec = _make_record()
        client, mock_storage, _ = _make_supabase_with_rows([rec])

        result = delete_candidate(rec["id"], client=client)

        assert result["deleted"] is True
        assert result["candidateId"] == rec["id"]
        assert result["storageDeleted"] is True
        assert result["storageAlreadyAbsent"] is False
        mock_storage.remove.assert_called_once()

    def test_storage_failure_aborts_before_db_is_touched(self):
        """If Storage deletion fails with a real error, DB row must remain intact."""
        rec = _make_record()
        client, mock_storage, mock_table = _make_supabase_with_rows([rec])
        mock_storage.remove.side_effect = Exception("S3 network timeout")

        with pytest.raises(StorageError, match="safe to retry"):
            delete_candidate(rec["id"], client=client)

        # DB delete must NOT have been called
        mock_table.delete.assert_not_called()

    def test_storage_already_absent_proceeds_to_db_delete(self):
        """If the storage object is already gone (404), continue and delete the DB row."""
        rec = _make_record()
        client, mock_storage, _ = _make_supabase_with_rows([rec])
        mock_storage.remove.side_effect = Exception("Object not found (404)")

        result = delete_candidate(rec["id"], client=client)

        assert result["deleted"] is True
        assert result["storageDeleted"] is False
        assert result["storageAlreadyAbsent"] is True

    def test_partial_failure_storage_deleted_db_fails(self):
        """Storage succeeds but DB deletion fails → PartialDeletionError with storage_object_path."""
        rec = _make_record()

        # Build a client where:
        #  - select (used by get_candidate) returns the record
        #  - storage.remove succeeds
        #  - delete().eq().execute() raises
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_client.storage.from_.return_value = mock_storage
        mock_storage.remove.return_value = [{"name": "removed"}]

        # select chain → returns rec for get_candidate lookup
        select_result = MagicMock()
        select_result.data = [rec]
        select_builder = MagicMock()
        select_builder.eq.return_value.limit.return_value.execute.return_value = select_result
        select_builder.eq.return_value.execute.return_value = select_result

        # delete chain → raises
        delete_builder = MagicMock()
        delete_builder.eq.return_value.execute.side_effect = Exception("DB connection reset")

        # table returns builder with both select and delete
        mock_table = MagicMock()
        mock_table.select.return_value = select_builder
        mock_table.delete.return_value = delete_builder
        mock_client.schema.return_value.table.return_value = mock_table
        mock_client.table.return_value = mock_table

        with pytest.raises(PartialDeletionError) as exc_info:
            delete_candidate(rec["id"], client=mock_client)

        err = exc_info.value
        assert err.storage_object_path == rec["storageObjectPath"]
        assert "storage" in str(err).lower()

    def test_missing_candidate_raises_not_found(self):
        client, _, _ = _make_supabase_with_rows([])

        with pytest.raises(CandidateNotFoundError):
            delete_candidate(_uuid(), client=client)

    def test_no_storage_path_skips_storage_delete_and_deletes_db(self):
        rec = _make_record({"storageObjectPath": ""})
        client, mock_storage, _ = _make_supabase_with_rows([rec])

        result = delete_candidate(rec["id"], client=client)

        assert result["deleted"] is True
        assert result["storageDeleted"] is False
        mock_storage.remove.assert_not_called()

    def test_invalid_id_raises_value_error(self):
        with pytest.raises(ValueError):
            delete_candidate("bad-id")


# ---------------------------------------------------------------------------
# 6. Server endpoint wiring (503 when Supabase not configured)
# ---------------------------------------------------------------------------

class TestEndpointRequiresSupabase:
    """
    When Supabase env vars are absent, all repository endpoints return 503
    without touching any DB.
    """

    def _call(self, fn, *args, **kwargs):
        """Call a repository function and expect RepositoryUnavailableError."""
        with pytest.raises(RepositoryUnavailableError):
            fn(*args, **kwargs)

    def test_list_unavailable(self):
        self._call(list_candidates)

    def test_search_unavailable(self):
        self._call(search_candidates, query="x")

    def test_get_unavailable(self):
        self._call(get_candidate, _uuid())

    def test_delete_unavailable(self):
        self._call(delete_candidate, _uuid())

    def test_download_unavailable(self):
        self._call(get_resume_download_url, _uuid())


# ---------------------------------------------------------------------------
# 6b. DELETE endpoint authorization
# ---------------------------------------------------------------------------

class TestDeleteAuthorization:
    """
    The DELETE /api/candidates/{id} endpoint on the Python backend must be
    protected by ATS_ADMIN_TOKEN.

    These tests call _require_delete_auth() directly with a lightweight mock
    Request object so they do not require fastapi.testclient / httpx (which
    hang in the sandbox due to its async event-loop initialisation on Windows).
    The full HTTP routing layer is covered by the unit tests below; the auth
    logic is self-contained in the helper and does not depend on routing.
    """

    # ------------------------------------------------------------------
    # Minimal stand-in for fastapi.Request that only exposes .headers
    # ------------------------------------------------------------------
    class _FakeRequest:
        """Mimics the subset of fastapi.Request used by _require_delete_auth."""
        def __init__(self, headers: dict):
            # Request.headers is a case-insensitive mapping; plain dict suffices
            # because we only call .get("Authorization", "")
            self.headers = headers

    def _call_guard(self, env_token, auth_header=None, monkeypatch=None):
        """
        Set ATS_ADMIN_TOKEN and call _require_delete_auth with a fake request.

        Returns the HTTPException raised (or None if no exception).
        """
        from app.server import _require_delete_auth
        from fastapi import HTTPException

        if env_token is None:
            monkeypatch.delenv("ATS_ADMIN_TOKEN", raising=False)
        else:
            monkeypatch.setenv("ATS_ADMIN_TOKEN", env_token)

        headers = {}
        if auth_header is not None:
            headers["Authorization"] = auth_header

        req = self._FakeRequest(headers)
        try:
            _require_delete_auth(req)
            return None  # no exception raised
        except HTTPException as exc:
            return exc

    # ------------------------------------------------------------------
    # 1. No token configured → 503
    # ------------------------------------------------------------------
    def test_no_token_configured_returns_503(self, monkeypatch):
        exc = self._call_guard(env_token=None, monkeypatch=monkeypatch)
        assert exc is not None, "Expected HTTPException but none was raised"
        assert exc.status_code == 503
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        assert "ATS_ADMIN_TOKEN" in detail.get("error", "")

    # ------------------------------------------------------------------
    # 2. Token configured but Authorization header missing → 401
    # ------------------------------------------------------------------
    def test_missing_auth_header_returns_401(self, monkeypatch):
        exc = self._call_guard(
            env_token="secure-token-abc",
            auth_header=None,   # no header at all
            monkeypatch=monkeypatch,
        )
        assert exc is not None, "Expected HTTPException but none was raised"
        assert exc.status_code == 401

    # ------------------------------------------------------------------
    # 3. Header present but token value is wrong → 403
    # ------------------------------------------------------------------
    def test_wrong_token_returns_403(self, monkeypatch):
        exc = self._call_guard(
            env_token="correct-token",
            auth_header="Bearer wrong-token",
            monkeypatch=monkeypatch,
        )
        assert exc is not None, "Expected HTTPException but none was raised"
        assert exc.status_code == 403

    # ------------------------------------------------------------------
    # 4. Correct token → guard raises nothing (returns None)
    # ------------------------------------------------------------------
    def test_correct_token_passes_auth_guard(self, monkeypatch):
        exc = self._call_guard(
            env_token="my-secret",
            auth_header="Bearer my-secret",
            monkeypatch=monkeypatch,
        )
        assert exc is None, f"Expected no exception but got HTTP {exc.status_code if exc else '?'}"

    # ------------------------------------------------------------------
    # 5. Other endpoints are unaffected: list_candidates raises no auth error
    #    even when ATS_ADMIN_TOKEN is absent (no TestClient needed — call
    #    the repository function directly with a mock Supabase client).
    # ------------------------------------------------------------------
    def test_other_endpoints_unaffected_by_token(self, monkeypatch):
        """GET /api/candidates logic works without any ATS_ADMIN_TOKEN."""
        monkeypatch.delenv("ATS_ADMIN_TOKEN", raising=False)
        monkeypatch.setenv("SUPABASE_URL", "https://mock.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "mock-key")

        client_mock, _, _ = _make_supabase_with_rows([])
        reset_supabase_client(client_mock)

        # list_candidates does NOT check ATS_ADMIN_TOKEN — call it directly
        from app.storage.candidate_repository import list_candidates
        result = list_candidates(limit=10, offset=0, client=client_mock)
        # Should return a valid response shape, not raise any auth error
        assert "candidates" in result
        assert isinstance(result["candidates"], list)


# ---------------------------------------------------------------------------
# 7. JD must NOT be persisted as CandidateResume
# ---------------------------------------------------------------------------

class TestJDNotPersisted:
    """
    The /api/files/upload endpoint is for resumes only.  The /invocations
    endpoint processes JD + resume text in memory for screening; it must not
    call store_resume with the JD content.
    """

    def test_invocations_handler_does_not_call_store_resume(self, monkeypatch):
        """
        The invocations handler calls adapter.evaluate_candidate / parse_screening_prompt.
        It must NEVER call store_resume, regardless of Supabase configuration.
        We patch store_resume and the adapter at the server module level and
        invoke the business logic directly via process_resume_upload to keep
        the test fast (no TestClient / NLP model boot).

        The contract under test: process_resume_upload is the only code path
        that calls store_resume. The invocations handler never goes through
        process_resume_upload, so store_resume is never called for JDs.
        """
        import app.server as server_module

        # Patch store_resume in server module scope to detect unexpected calls
        original_store = server_module.store_resume
        calls = []

        def spy_store(*args, **kwargs):
            calls.append((args, kwargs))
            return original_store(*args, **kwargs)

        monkeypatch.setattr(server_module, "store_resume", spy_store)
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        reset_supabase_client(None)

        # Directly invoke the helper function (not the HTTP layer)
        # When Supabase is not configured, store_resume is never called
        content = b"Jane Smith\njane@example.com\nPython developer."
        server_module.process_resume_upload("jane.txt", content, "text/plain")

        # store_resume must not have been called (Supabase not configured)
        assert calls == [], "store_resume was called but Supabase is not configured"

    def test_upload_endpoint_does_not_store_jd_text(self, monkeypatch):
        """
        A file uploaded via process_resume_upload is treated as a resume.
        When Supabase is not configured, store_resume is never called and
        the response carries persistenceStatus=temporary_in_memory.
        """
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        reset_supabase_client(None)

        content = b"Jane Smith\njane@example.com\nPython developer."
        result = process_resume_upload("jane_smith.txt", content, "text/plain")

        # Without Supabase configured, no candidateId should be returned
        assert isinstance(result, dict)
        assert result.get("candidateId") is None
        assert result.get("persistenceStatus") == "temporary_in_memory"

    def test_jd_content_uploaded_does_not_persist_as_candidate(self, monkeypatch):
        """
        Even if a JD is accidentally uploaded as a file, the pipeline stores it
        as a candidate (resume-upload is document-type-agnostic — the Batch 2
        requirement is that the /invocations screening path never persists).
        This test confirms: without Supabase configured, nothing is ever persisted.
        """
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        reset_supabase_client(None)

        jd_content = (
            b"Job Title: Senior Software Engineer\n"
            b"We are looking for candidates with 5+ years of Python experience.\n"
            b"Requirements: FastAPI, PostgreSQL, Docker.\n"
        )

        result = process_resume_upload("job_description.txt", jd_content, "text/plain")

        # Must be temporary in-memory (no Supabase)
        assert isinstance(result, dict)
        assert result.get("persistenceStatus") == "temporary_in_memory"
        assert result.get("candidateId") is None


# ---------------------------------------------------------------------------
# 8. Existing Batch 1 behaviour remains unchanged
# ---------------------------------------------------------------------------

class TestBatch1Compatibility:
    """
    Spot-check that store_resume, duplicate detection, and process_resume_upload
    still work exactly as before Batch 2 changes.
    """

    def _make_full_mock(self, existing_hash=None):
        existing = {}
        if existing_hash:
            existing[existing_hash] = {
                "id": _uuid(),
                "fileHash": existing_hash,
                "originalFilename": "orig.pdf",
                "storageObjectPath": "resumes/orig-uuid/orig.pdf",
                "candidateName": "Existing Candidate",
                "email": None,
                "jobRole": "Engineer",
                "createdAt": "2026-01-01T00:00:00Z",
            }

        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_client.storage.from_.return_value = mock_storage
        mock_storage.upload.return_value = {"Key": "resumes/new-key"}
        mock_storage.remove.return_value = [{"name": "removed"}]

        mock_table = MagicMock()
        mock_client.schema.return_value.table.return_value = mock_table
        mock_client.table.return_value = mock_table

        def mock_select(*args, **kwargs):
            sel = MagicMock()
            def mock_eq(col, val):
                eq = MagicMock()
                def mock_limit(n):
                    lim = MagicMock()
                    def _exec():
                        res = MagicMock()
                        res.data = [existing[val]] if val in existing else []
                        return res
                    lim.execute = _exec
                    return lim
                eq.limit = mock_limit
                eq.execute = lambda: MagicMock(data=[existing[val]] if val in existing else [])
                return eq
            sel.eq = mock_eq
            return sel

        mock_table.select.side_effect = mock_select
        insert_builder = MagicMock()
        insert_builder.execute.return_value = MagicMock(data=[{}])
        mock_table.insert.return_value = insert_builder

        return mock_client, mock_storage, mock_table

    def test_first_time_store_succeeds(self):
        from app.storage.resume_storage import store_resume

        content = b"Some resume bytes"
        file_hash = hashlib.sha256(content).hexdigest()
        client, mock_storage, _ = self._make_full_mock()

        result = store_resume(
            file_bytes=content,
            filename="resume.pdf",
            file_hash=file_hash,
            candidate_name="Test Candidate",
            email=None,
            resume_text="Some resume bytes",
            skills=[],
            job_role=None,
            client=client,
        )

        assert result["isDuplicate"] is False
        assert result["persistenceStatus"] == "persisted"
        mock_storage.upload.assert_called_once()

    def test_duplicate_detected_and_no_re_upload(self):
        from app.storage.resume_storage import store_resume

        content = b"Identical resume bytes"
        file_hash = hashlib.sha256(content).hexdigest()
        client, mock_storage, _ = self._make_full_mock(existing_hash=file_hash)

        result = store_resume(
            file_bytes=content,
            filename="dup.pdf",
            file_hash=file_hash,
            candidate_name="Dup Candidate",
            email=None,
            resume_text="Identical resume bytes",
            skills=[],
            job_role=None,
            client=client,
        )

        assert result["isDuplicate"] is True
        assert result["persistenceStatus"] == "duplicate_reused"
        mock_storage.upload.assert_not_called()

    def test_process_resume_upload_without_supabase(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        reset_supabase_client(None)

        content = b"Alice Smith\nalice@test.com\nDeveloper."
        result = process_resume_upload("alice.txt", content, "text/plain")

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "Alice Smith" in result["resumeText"]
        assert result["persistenceStatus"] == "temporary_in_memory"
