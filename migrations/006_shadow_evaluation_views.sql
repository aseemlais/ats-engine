-- migrations/006_shadow_evaluation_views.sql
-- =============================================================================
-- Migration 006: Additive columns and evaluation views for Phase 4 Shadow Mode
-- =============================================================================

-- 1. Ensure all harmonization columns exist on extraction_shadow_logs
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "jobId" TEXT;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "primarySource" TEXT DEFAULT 'legacy';
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "shadowSource" TEXT DEFAULT 'slm';
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "primaryDurationMs" DOUBLE PRECISION DEFAULT 0.0;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "shadowDurationMs" DOUBLE PRECISION DEFAULT 0.0;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "rssMbAtRun" DOUBLE PRECISION DEFAULT 0.0;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "isValidJson" BOOLEAN DEFAULT true;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "agreementScores" JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "discrepancies" JSONB DEFAULT '[]'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "shadowOutput" JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "legacyVersion" TEXT DEFAULT 'ats-legacy:heuristic-v1';
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "slmVersion" TEXT DEFAULT 'unsloth-qwen2.5-1.5b-q4_k_m';
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "legacyOutput" JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "slmOutput" JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "fieldAgreements" JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "validityFlags" JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "groundingFlags" JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "slmLatencyMs" NUMERIC(10, 2);
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "tokensOut" INT;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "errorFlags" TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "reviewLabel" TEXT;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "reviewerNotes" TEXT;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "labeledAt" TIMESTAMPTZ;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "docFormat" TEXT DEFAULT 'pdf';
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "pageCount" INT DEFAULT 1;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "isTwoColumn" BOOLEAN DEFAULT false;
ALTER TABLE public.extraction_shadow_logs ADD COLUMN IF NOT EXISTS "charLength" INT DEFAULT 0;

-- 2. Indexes for efficient slice querying and review labeling
CREATE INDEX IF NOT EXISTS idx_shadow_logs_review_label 
    ON public.extraction_shadow_logs ("reviewLabel");

CREATE INDEX IF NOT EXISTS idx_shadow_logs_doc_format 
    ON public.extraction_shadow_logs ("docFormat");

-- 3. SQL View: Agreement rate by field
CREATE OR REPLACE VIEW public.view_shadow_field_agreements AS
SELECT
    COUNT(*) AS total_samples,
    ROUND(AVG(CASE WHEN COALESCE(( "fieldAgreements"->>'emailExactMatch' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS email_exact_match_pct,
    ROUND(AVG(CASE WHEN COALESCE(( "fieldAgreements"->>'phoneExactMatch' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS phone_exact_match_pct,
    ROUND(AVG(COALESCE(( "fieldAgreements"->>'nameTokenF1' )::NUMERIC, 0.0)), 4) AS avg_name_token_f1,
    ROUND(AVG(COALESCE(( "fieldAgreements"->>'titleTokenF1' )::NUMERIC, 0.0)), 4) AS avg_title_token_f1,
    ROUND(AVG(COALESCE(( "fieldAgreements"->>'skillsJaccard' )::NUMERIC, 0.0)), 4) AS avg_skills_jaccard,
    ROUND(AVG(COALESCE(( "fieldAgreements"->>'expDeltaYears' )::NUMERIC, 0.0)), 2) AS avg_exp_delta_years,
    ROUND(AVG(CASE WHEN COALESCE(( "fieldAgreements"->>'educationCountMatch' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS education_count_match_pct,
    ROUND(AVG(CASE WHEN COALESCE(( "fieldAgreements"->>'workExperienceCountMatch' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS work_experience_count_match_pct
FROM public.extraction_shadow_logs;

-- 4. SQL View: Validity-rate comparison (Legacy vs SLM)
CREATE OR REPLACE VIEW public.view_shadow_validity_rates AS
SELECT
    COUNT(*) AS total_samples,
    ROUND(AVG(CASE WHEN COALESCE(( "validityFlags"->>'emailValid' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS slm_email_valid_pct,
    ROUND(AVG(CASE WHEN COALESCE(( "validityFlags"->>'phoneValid' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS slm_phone_valid_pct,
    ROUND(AVG(CASE WHEN COALESCE(( "validityFlags"->>'namePlausible' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS slm_name_plausible_pct,
    ROUND(AVG(CASE WHEN COALESCE(( "validityFlags"->>'datesParse' )::BOOLEAN, false) THEN 1.0 ELSE 0.0 END) * 100, 2) AS slm_dates_parse_pct
FROM public.extraction_shadow_logs;

-- 5. SQL View: Agreement rate by slice (Format, Page count, Columns, Length)
CREATE OR REPLACE VIEW public.view_shadow_slice_breakdown AS
SELECT
    COALESCE("docFormat", 'unknown') AS doc_format,
    COALESCE("pageCount", 1) AS page_count,
    COALESCE("isTwoColumn", false) AS is_two_column,
    COUNT(*) AS sample_count,
    ROUND(AVG(COALESCE(( "fieldAgreements"->>'titleTokenF1' )::NUMERIC, 0.0)), 4) AS avg_title_f1,
    ROUND(AVG(COALESCE(( "fieldAgreements"->>'skillsJaccard' )::NUMERIC, 0.0)), 4) AS avg_skills_jaccard,
    ROUND(AVG(COALESCE("slmLatencyMs", "shadowDurationMs", 0.0)::NUMERIC), 1) AS avg_latency_ms
FROM public.extraction_shadow_logs
GROUP BY "docFormat", "pageCount", "isTwoColumn"
ORDER BY sample_count DESC;

-- 6. SQL View: Error rates, timeouts, grounding, and memory
CREATE OR REPLACE VIEW public.view_shadow_error_rates AS
SELECT
    COUNT(*) AS total_runs,
    SUM(CASE WHEN "isValidJson" = false OR 'json_invalid' = ANY("errorFlags") THEN 1 ELSE 0 END) AS invalid_json_count,
    ROUND((SUM(CASE WHEN "isValidJson" = false OR 'json_invalid' = ANY("errorFlags") THEN 1 ELSE 0 END)::NUMERIC / GREATEST(COUNT(*), 1)) * 100, 3) AS invalid_json_rate_pct,
    SUM(CASE WHEN 'timeout' = ANY("errorFlags") THEN 1 ELSE 0 END) AS timeout_count,
    ROUND((SUM(CASE WHEN 'timeout' = ANY("errorFlags") THEN 1 ELSE 0 END)::NUMERIC / GREATEST(COUNT(*), 1)) * 100, 3) AS timeout_rate_pct,
    SUM(CASE WHEN COALESCE(( "groundingFlags"->>'allGrounded' )::BOOLEAN, true) = false THEN 1 ELSE 0 END) AS grounding_failure_count,
    ROUND((SUM(CASE WHEN COALESCE(( "groundingFlags"->>'allGrounded' )::BOOLEAN, true) = false THEN 1 ELSE 0 END)::NUMERIC / GREATEST(COUNT(*), 1)) * 100, 3) AS grounding_failure_rate_pct
FROM public.extraction_shadow_logs;

-- 7. SQL View: Review Queue prioritizing disagreements and ungrounded entities
CREATE OR REPLACE VIEW public.view_shadow_review_queue AS
SELECT
    "id",
    "jobId",
    "fileHash",
    "docFormat",
    "pageCount",
    "isTwoColumn",
    "createdAt",
    "fieldAgreements",
    "validityFlags",
    "groundingFlags",
    "legacyOutput",
    "slmOutput",
    "reviewLabel",
    "reviewerNotes"
FROM public.extraction_shadow_logs
WHERE "reviewLabel" IS NULL
ORDER BY
    -- Prioritize records with grounding failures or severe title/contact mismatches
    (CASE WHEN COALESCE(( "groundingFlags"->>'allGrounded' )::BOOLEAN, true) = false THEN 0 ELSE 1 END) ASC,
    COALESCE(( "fieldAgreements"->>'titleTokenF1' )::NUMERIC, 1.0) ASC,
    COALESCE(( "fieldAgreements"->>'skillsJaccard' )::NUMERIC, 1.0) ASC,
    "createdAt" DESC;
