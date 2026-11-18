-- Migration: 003_create_extraction_jobs.sql
-- Description: Persistent extraction job queue for restart resilience and polling recovery
-- Schema: ai_chatbot

CREATE TABLE IF NOT EXISTS ai_chatbot."ExtractionJobs" (
    "jobId" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "fileHash" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "storageObjectPath" TEXT,
    "extractorVersion" TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    "extractionSource" TEXT,
    attempts INT NOT NULL DEFAULT 0,
    result JSONB,
    error TEXT,
    "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "ExtractionJobs_status_check" 
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    CONSTRAINT "ExtractionJobs_source_check" 
        CHECK ("extractionSource" IS NULL OR "extractionSource" IN ('slm', 'fallback', 'cache', 'legacy'))
);

-- Partial unique index to guarantee only one active job per fileHash
CREATE UNIQUE INDEX IF NOT EXISTS idx_extraction_jobs_active_file_hash 
    ON ai_chatbot."ExtractionJobs"("fileHash") 
    WHERE status IN ('pending', 'processing');

-- Index for queue polling and worker recovery
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_status_created_at 
    ON ai_chatbot."ExtractionJobs"(status, "createdAt");

-- Index for general lookup by file hash
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_file_hash 
    ON ai_chatbot."ExtractionJobs"("fileHash");

-- Row Level Security (Service role only)
ALTER TABLE ai_chatbot."ExtractionJobs" ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'ai_chatbot' 
          AND tablename = 'ExtractionJobs' 
          AND policyname = 'Service role full access on ExtractionJobs'
    ) THEN
        CREATE POLICY "Service role full access on ExtractionJobs"
            ON ai_chatbot."ExtractionJobs"
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END $$;
