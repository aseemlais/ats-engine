import { useState, useCallback, useRef, useEffect } from 'react';
import { apiFetch } from '@/lib/api';

export interface EducationEntry {
  degree: string;
  tier?: 'doctorate' | 'master' | 'bachelor' | 'associate' | 'diploma' | 'none';
  institution?: string | null;
  fieldOfStudy?: string | null;
  year?: string | null;
}

export interface WorkExperienceEntry {
  title: string;
  company?: string | null;
  dates?: string | null;
  durationYears?: number;
  isCurrent?: boolean;
  isInternship?: boolean;
  description?: string | null;
}

export interface ExtractedResumeData {
  candidateName: string | null;
  email: string | null;
  phone: string | null;
  location: string | null;
  jobTitle: string | null;
  totalExperienceYears: number;
  workExperienceYears: number;
  internshipYears: number;
  skills: string[];
  education: EducationEntry[];
  degrees: string[];
  workExperience: WorkExperienceEntry[];
  certifications: string[];
  projects: string[];
  source?: string;
  storageObjectPath?: string | null;
}

export type ExtractionJobStatus =
  | 'idle'
  | 'uploading'
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed';

export interface UseResumeExtractionOptions {
  pollIntervalMs?: number;
  maxAttempts?: number;
  apiBaseUrl?: string;
  onSuccess?: (data: ExtractedResumeData) => void;
  onError?: (err: Error) => void;
}

export function useResumeExtraction(options: UseResumeExtractionOptions = {}) {
  const {
    pollIntervalMs = 1500,
    maxAttempts = 60,
    apiBaseUrl = '',
    onSuccess,
    onError,
  } = options;

  const [status, setStatus] = useState<ExtractionJobStatus>('idle');
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractedResumeData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);

  const activeTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isCancelledRef = useRef<boolean>(false);

  const clearTimer = useCallback(() => {
    if (activeTimerRef.current) {
      clearTimeout(activeTimerRef.current);
      activeTimerRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    isCancelledRef.current = true;
    clearTimer();
    setStatus('idle');
    setProgress(0);
  }, [clearTimer]);

  const reset = useCallback(() => {
    cancel();
    setJobId(null);
    setResult(null);
    setError(null);
    setStatus('idle');
    setProgress(0);
  }, [cancel]);

  useEffect(() => {
    return () => {
      isCancelledRef.current = true;
      clearTimer();
    };
  }, [clearTimer]);

  const pollJobStatus = useCallback(
    async (id: string, attempt = 1): Promise<ExtractedResumeData> => {
      if (isCancelledRef.current) {
        throw new Error('Extraction was cancelled.');
      }

      if (attempt > maxAttempts) {
        throw new Error('Extraction timed out. The server is still processing your resume.');
      }

      const pollUrl = `${apiBaseUrl}/jobs/${id}`;
      const response = await apiFetch(pollUrl);

      if (!response.ok) {
        if (response.status === 404 && attempt < 5) {
          await new Promise((resolve) => {
            activeTimerRef.current = setTimeout(resolve, pollIntervalMs);
          });
          return pollJobStatus(id, attempt + 1);
        }
        throw new Error(`Failed to query job status (${response.status})`);
      }

      const jobData = await response.json();
      const currentStatus: ExtractionJobStatus = jobData.status;

      if (currentStatus === 'completed' && jobData.result) {
        setStatus('completed');
        setProgress(100);
        setResult(jobData.result);
        onSuccess?.(jobData.result);
        return jobData.result;
      }

      if (currentStatus === 'failed') {
        const errorMsg = jobData.error || 'Resume extraction failed on server.';
        setStatus('failed');
        setError(errorMsg);
        const err = new Error(errorMsg);
        onError?.(err);
        throw err;
      }

      setStatus(currentStatus);
      if (currentStatus === 'processing') {
        setProgress(Math.min(85, 30 + attempt * 2));
      } else {
        setProgress(20);
      }

      await new Promise((resolve) => {
        activeTimerRef.current = setTimeout(resolve, pollIntervalMs);
      });

      return pollJobStatus(id, attempt + 1);
    },
    [apiBaseUrl, maxAttempts, pollIntervalMs, onSuccess, onError],
  );

  const extractResume = useCallback(
    async (file: File): Promise<ExtractedResumeData> => {
      isCancelledRef.current = false;
      clearTimer();
      setStatus('uploading');
      setError(null);
      setResult(null);
      setProgress(10);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const submitUrl = `${apiBaseUrl}/extract`;
        const response = await apiFetch(submitUrl, {
          method: 'POST',
          body: formData,
        });

        if (response.status === 429) {
          throw new Error('Server queue is at maximum capacity. Please retry in a few moments.');
        }

        if (!response.ok && response.status !== 202) {
          const errBody = await response.text();
          throw new Error(`Upload failed (${response.status}): ${errBody}`);
        }

        const data = await response.json();

        // 1. Immediate Cache Hit (200 OK)
        if (response.status === 200 && data.status === 'completed' && data.result) {
          setStatus('completed');
          setProgress(100);
          setJobId(data.job_id);
          setResult(data.result);
          onSuccess?.(data.result);
          return data.result;
        }

        // 2. Queued Job (202 Accepted)
        if (response.status === 202 && data.job_id) {
          setJobId(data.job_id);
          setStatus('pending');
          setProgress(25);
          return await pollJobStatus(data.job_id);
        }

        throw new Error('Unexpected response structure from extraction server.');
      } catch (err: any) {
        if (!isCancelledRef.current) {
          setStatus('failed');
          setError(err.message || 'Failed to extract resume.');
          onError?.(err);
        }
        throw err;
      }
    },
    [apiBaseUrl, clearTimer, pollJobStatus, onSuccess, onError],
  );

  return {
    extractResume,
    cancel,
    reset,
    status,
    jobId,
    result,
    error,
    progress,
    isLoading: status === 'uploading' || status === 'pending' || status === 'processing',
    isCompleted: status === 'completed',
    isFailed: status === 'failed',
  };
}
