-- Migration: 001_create_candidate_resumes.sql
-- Description: Add CandidateResume table and storage bucket for persistent resume storage (Batch 1)
-- Schema: ai_chatbot

-- 1. Ensure schema exists
CREATE SCHEMA IF NOT EXISTS ai_chatbot;

-- 2. Create CandidateResume table
CREATE TABLE IF NOT EXISTS ai_chatbot."CandidateResume" (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "fileHash" TEXT NOT NULL,
    "originalFilename" TEXT NOT NULL,
    "storageObjectPath" TEXT NOT NULL,
    "candidateName" TEXT NOT NULL DEFAULT 'Unknown',
    email TEXT,
    "resumeText" TEXT,
    skills TEXT[] NOT NULL DEFAULT '{}',
    "jobRole" TEXT,
    "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "CandidateResume_fileHash_unique" UNIQUE ("fileHash")
);

-- 3. Indexes for fast lookup and deduplication
CREATE INDEX IF NOT EXISTS idx_candidate_resumes_file_hash 
    ON ai_chatbot."CandidateResume"("fileHash");

CREATE INDEX IF NOT EXISTS idx_candidate_resumes_email 
    ON ai_chatbot."CandidateResume"(email) 
    WHERE email IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_candidate_resumes_created_at 
    ON ai_chatbot."CandidateResume"("createdAt" DESC);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE ai_chatbot."CandidateResume" ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access (Backend uses service_role key)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'ai_chatbot' 
          AND tablename = 'CandidateResume' 
          AND policyname = 'Service role full access on CandidateResume'
    ) THEN
        CREATE POLICY "Service role full access on CandidateResume"
            ON ai_chatbot."CandidateResume"
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END $$;

-- 5. Storage Bucket setup (resumes - private)
-- Run this in Supabase SQL editor to create the private bucket:
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'resumes',
    'resumes',
    false,
    20971520, -- 20 MB limit
    ARRAY['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword', 'text/plain']
)
ON CONFLICT (id) DO NOTHING;

-- 6. Explicit schema permissions for PostgREST / service_role
GRANT USAGE ON SCHEMA ai_chatbot TO postgres, anon, authenticated, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA ai_chatbot TO postgres, anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ai_chatbot TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA ai_chatbot GRANT ALL ON TABLES TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA ai_chatbot GRANT ALL ON SEQUENCES TO postgres, anon, authenticated, service_role;

