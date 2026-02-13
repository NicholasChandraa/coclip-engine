-- CoClip Database Schema
-- PostgreSQL 

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id VARCHAR(36) PRIMARY KEY,
    video_name VARCHAR(255) NOT NULL,
    language VARCHAR(10),
    duration FLOAT,
    total_segments INTEGER,
    status VARCHAR(50) DEFAULT 'queued',
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Clips table
CREATE TABLE IF NOT EXISTS clips (
    id SERIAL PRIMARY KEY,
    clip_id VARCHAR(100) UNIQUE NOT NULL,
    job_id VARCHAR(36) NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    clip_number INTEGER NOT NULL,
    start FLOAT NOT NULL,
    "end" FLOAT NOT NULL,
    duration FLOAT NOT NULL,
    title VARCHAR(255) NOT NULL,
    reasoning TEXT,
    viral_score FLOAT,
    suggested_caption TEXT,
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER,
    has_subtitles BOOLEAN DEFAULT FALSE,
    status VARCHAR(50) DEFAULT 'ready',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_clips_job_id ON clips(job_id);
CREATE INDEX IF NOT EXISTS idx_clips_clip_number ON clips(clip_number);
