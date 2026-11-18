-- Migration: 002_candidate_repository_indexes.sql
-- Batch 2: Add additional indexes to support Repository API list/search/sort performance.
-- All changes are additive and idempotent (IF NOT EXISTS / ON CONFLICT DO NOTHING).

-- Full-text / trigram search support (if pg_trgm is available)
-- These are optional: the ilike-based search in PostgREST works without them,
-- but they speed up ILIKE '%query%' scans on larger candidate tables.
-- Uncomment if the Supabase project has pg_trgm enabled:
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- CREATE INDEX IF NOT EXISTS idx_candidate_resumes_name_trgm
--     ON ai_chatbot."CandidateResume" USING gin ("candidateName" gin_trgm_ops);

-- CREATE INDEX IF NOT EXISTS idx_candidate_resumes_jobrole_trgm
--     ON ai_chatbot."CandidateResume" USING gin ("jobRole" gin_trgm_ops);

-- Composite index for the default sort + pagination (createdAt DESC, id)
-- already covered by idx_candidate_resumes_created_at from migration 001,
-- but an explicit covering index can improve LIMIT/OFFSET plans:
CREATE INDEX IF NOT EXISTS idx_candidate_resumes_created_at_id
    ON ai_chatbot."CandidateResume"("createdAt" DESC, id);

-- Index on jobRole for role-based filtering
CREATE INDEX IF NOT EXISTS idx_candidate_resumes_job_role
    ON ai_chatbot."CandidateResume"("jobRole")
    WHERE "jobRole" IS NOT NULL;

-- GIN index on skills TEXT[] for contains (@>) queries
CREATE INDEX IF NOT EXISTS idx_candidate_resumes_skills_gin
    ON ai_chatbot."CandidateResume" USING gin (skills);
