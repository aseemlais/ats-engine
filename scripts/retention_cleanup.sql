-- retention_cleanup.sql
-- Description: Data retention policy for completed/failed jobs (default 30 days) to prevent PII accumulation
-- Schema: ai_chatbot

-- Option 1: Manual Runbook Execution (run periodically in Supabase SQL editor):
DELETE FROM ai_chatbot."ExtractionJobs"
WHERE "createdAt" < NOW() - INTERVAL '30 days';

-- Option 2: pg_cron (if pg_cron extension is enabled in your Supabase project):
-- SELECT cron.schedule(
--     'cleanup-extraction-jobs-nightly',
--     '0 3 * * *', -- Nightly at 3:00 AM UTC
--     $$DELETE FROM ai_chatbot."ExtractionJobs" WHERE "createdAt" < NOW() - INTERVAL '30 days'$$
-- );
