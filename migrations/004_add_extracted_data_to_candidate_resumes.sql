-- Migration: 004_add_extracted_data_to_candidate_resumes.sql
-- Description: Additive columns for rich extraction caching and version-aware invalidation
-- Schema: ai_chatbot

ALTER TABLE ai_chatbot."CandidateResume"
    ADD COLUMN IF NOT EXISTS "extractedData" JSONB,
    ADD COLUMN IF NOT EXISTS "extractorVersion" TEXT,
    ADD COLUMN IF NOT EXISTS "extractionSource" TEXT;

CREATE INDEX IF NOT EXISTS idx_candidate_resumes_version 
    ON ai_chatbot."CandidateResume"("extractorVersion");
