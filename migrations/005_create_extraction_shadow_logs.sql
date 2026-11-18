-- migrations/005_create_extraction_shadow_logs.sql
-- =============================================================================
-- Migration 005: Create extraction_shadow_logs table for Phase 3 shadow evaluation
-- =============================================================================
-- Stores side-by-side extraction outputs (legacy vs. fine-tuned SLM), agreement
-- metrics, validity flags, grounding checks, and latency without impacting production traffic.

CREATE TABLE IF NOT EXISTS "extraction_shadow_logs" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "fileHash" TEXT NOT NULL,
    "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "legacyVersion" TEXT NOT NULL,
    "slmVersion" TEXT NOT NULL,
    "legacyOutput" JSONB NOT NULL,
    "slmOutput" JSONB NOT NULL,
    "fieldAgreements" JSONB NOT NULL,
    "validityFlags" JSONB NOT NULL,
    "groundingFlags" JSONB NOT NULL,
    "slmLatencyMs" NUMERIC(10, 2),
    "tokensOut" INT,
    "errorFlags" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "reviewLabel" TEXT
);

-- Index on createdAt for range queries, metrics aggregation, and retention cleanup
CREATE INDEX IF NOT EXISTS idx_shadow_logs_created_at 
    ON "extraction_shadow_logs" ("createdAt" DESC);

-- Index on fileHash to correlate shadow extractions with CandidateResume and ExtractionJobs
CREATE INDEX IF NOT EXISTS idx_shadow_logs_file_hash 
    ON "extraction_shadow_logs" ("fileHash");

-- Enable Row Level Security (RLS) - restricts access to service_role
ALTER TABLE "extraction_shadow_logs" ENABLE ROW LEVEL SECURITY;

-- Service Role Policy: Only internal service role key can read, insert, update, or delete
DROP POLICY IF EXISTS "Service role access on extraction_shadow_logs" ON "extraction_shadow_logs";
CREATE POLICY "Service role access on extraction_shadow_logs"
    ON "extraction_shadow_logs"
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
