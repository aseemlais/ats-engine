-- scripts/retention_cleanup_shadow_logs.sql
-- =============================================================================
-- Scheduled Retention Cleanup Script for extraction_shadow_logs
-- Default retention window: 30 days
-- =============================================================================
-- Recommended execution:
--   Option A: via pg_cron extension on Supabase:
--     SELECT cron.schedule('cleanup-shadow-logs-daily', '0 3 * * *', $$
--       DELETE FROM "extraction_shadow_logs" WHERE "createdAt" < NOW() - INTERVAL '30 days';
--     $$);
--
--   Option B: via scheduled maintenance job / GitHub Action / scripts/cleanup_retention.py

DELETE FROM "extraction_shadow_logs"
WHERE "createdAt" < NOW() - INTERVAL '30 days';
