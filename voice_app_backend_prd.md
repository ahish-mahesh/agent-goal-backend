# Voice Recording + Note Taking App Backend - PRD

## Overview
High-performance Rust backend for student note-taking app. Real-time audio transcription, LLM note enhancement, academic content organization. Always-available standalone service.

**Tech Stack:** Actix-web, whisper.cpp, Ollama, SQLite with FTS5

**Recommended Models:**
- **Whisper:** Medium model (accuracy for technical vocabulary)
- **LLM:** Llama 3.1 8B (primary), Phi-3 Medium (alternative)

## Architecture
- Real-time WebSocket audio streaming
- Background LLM processing queues
- Academic hierarchy: Courses > Weeks > Topics > Notes
- No audio storage (real-time only)
- Configurable model selection

## Database Schema
```sql
Courses: id, name, description, created_at, updated_at
Weeks: id, course_id, week_number, title, start_date, end_date  
Topics: id, week_id, title, description, created_at
Notes: id, topic_id, title, content, transcript, created_at, updated_at, metadata_json
Transcription_Sessions: id, note_id, start_time, end_time, status, model_used
```

## Development Phases

### Phase 1: Foundation Service
**Goal:** Core Actix-web service with configuration and monitoring

**Implementation Tasks:**
- Create main.rs with Actix-web HttpServer setup, configurable host/port binding
- Implement config.rs module using serde for JSON/TOML configuration files + env vars
- Add structured logging with tracing crate, configure log levels and output formats
- Create health.rs module for system status, model availability checks, uptime tracking
- Implement graceful shutdown handler for SIGTERM/SIGINT with cleanup routines
- Add basic middleware: CORS, request logging, error handling
- Create service state struct for shared configuration and resources

**Configuration Structure:**
```toml
[server]
host = "127.0.0.1"
port = 8080

[models]
whisper_model = "medium"
llm_model = "llama3.1:8b"

[performance]
max_concurrent_sessions = 10
audio_buffer_size = 4096
```

**Endpoints:**
- `GET /health` - Service status, model status, memory usage
- `GET /config` - Current configuration display  
- `PUT /config` - Runtime configuration updates with validation
- `GET /metrics` - Performance stats, request counts, error rates

### Phase 2: Audio Pipeline
**Goal:** Real-time transcription via whisper.cpp

**Implementation Tasks:**
- Setup whisper.cpp FFI bindings or use existing Rust wrapper crate
- Create audio.rs module for WebSocket handler accepting binary audio data
- Implement circular audio buffer with configurable size and overlap for context
- Add audio preprocessing: format validation, resampling to 16kHz, noise reduction
- Create transcription.rs with model loading, inference queue, result streaming
- Implement concurrent processing: separate threads for audio buffering and inference
- Add audio session management: start/stop/pause with proper resource cleanup
- Create message protocol for WebSocket: audio data, transcription results, status updates

**Audio Processing Flow:**
1. WebSocket receives audio chunks (PCM 16-bit, 16kHz)
2. Buffer audio with 1-2 second overlap for context
3. Queue audio segments for Whisper inference
4. Stream transcription results back via WebSocket
5. Handle model loading/unloading and concurrent sessions

**Endpoints:**
- `WebSocket /ws/audio` - Binary audio in, JSON transcription out
- `GET /models/whisper` - List available models, current loaded model
- `POST /models/whisper/load` - Load specific model with size/performance info
- `POST /transcribe/file` - Test endpoint for audio file transcription

### Phase 3: Data Management
**Goal:** SQLite with academic content APIs

**Implementation Tasks:**
- Create database.rs with SQLx connection pool, migration system
- Implement SQL migrations for schema creation with proper indexes
- Create models.rs with Rust structs matching database schema using serde
- Add repositories for each entity with full CRUD operations and validation
- Setup SQLite FTS5 virtual tables for full-text search across content
- Implement foreign key constraints and cascading deletes
- Add database connection health checks and connection recovery
- Create API handlers for all CRUD operations with proper error handling

**Database Schema Implementation:**
```sql
-- Add FTS5 virtual table for search
CREATE VIRTUAL TABLE notes_fts USING fts5(title, content, transcript);

-- Add indexes for performance
CREATE INDEX idx_weeks_course_id ON weeks(course_id);
CREATE INDEX idx_topics_week_id ON topics(week_id);
CREATE INDEX idx_notes_topic_id ON notes(topic_id);
CREATE INDEX idx_notes_created_at ON notes(created_at);
```

**Endpoints:**
- `GET/POST/PUT/DELETE /courses` - Course management
- `GET/POST/PUT/DELETE /weeks` - Week management with course filtering
- `GET/POST/PUT/DELETE /topics` - Topic management with week filtering
- `GET/POST/PUT/DELETE /notes` - Note management with topic filtering
- `GET /search?q={query}&course_id={id}` - Full-text search with filtering
- `GET /courses/{id}/stats` - Course statistics, note counts, recent activity

### Phase 4: LLM Integration
**Goal:** Ollama integration for summarization and keywords

**Implementation Tasks:**
- Create ollama.rs client module using reqwest for HTTP API communication
- Implement prompt templates for academic content: lectures, study notes, technical content
- Setup background job queue using tokio tasks for non-blocking LLM processing
- Add error handling for Ollama connection failures, timeouts, model loading issues
- Create structured output parsing for summaries and keywords with JSON validation
- Implement model management: list models, load/unload, check availability
- Add processing triggers: manual request, auto after transcription, scheduled batch
- Create llm.rs service layer with retry logic and fallback strategies

**Prompt Templates:**
```rust
const SUMMARY_PROMPT: &str = "Summarize this lecture content in 3-5 bullet points focusing on key concepts: {content}";
const KEYWORDS_PROMPT: &str = "Extract 5-10 important keywords and concepts from this academic content: {content}";
```

**Background Processing:**
- Use tokio::spawn for async LLM processing
- Implement job queue with priority levels
- Add progress tracking and status updates
- Handle partial results and streaming responses

**Endpoints:**
- `POST /llm/summarize` - Generate summary for note ID or raw content
- `POST /llm/keywords` - Extract keywords with confidence scores
- `GET /llm/models` - Available models, capabilities, performance metrics
- `POST /llm/process-batch` - Batch process multiple notes with progress tracking

### Phase 5: Real-time Integration
**Goal:** Seamless real-time transcription + LLM processing

**Implementation Tasks:**
- Enhance WebSocket handler to multiplex audio, transcription, and LLM results
- Implement intelligent audio segmentation using voice activity detection (VAD)
- Create adaptive LLM triggering: process on natural pauses, sentence boundaries
- Add message queuing system for ordered delivery of different content types
- Implement progressive enhancement: immediate transcription, delayed summary/keywords
- Create session state management tracking multiple concurrent processing streams
- Add client synchronization protocols for message ordering and delivery confirmation
- Implement adaptive quality: adjust processing based on audio quality and speech rate

**WebSocket Message Protocol:**
```json
// Outbound message types
{"type": "transcription", "text": "...", "confidence": 0.95, "timestamp": 1234567890}
{"type": "summary", "content": "...", "processing_time": 1.2}
{"type": "keywords", "keywords": ["concept1", "concept2"], "relevance": [0.9, 0.8]}
{"type": "status", "session_id": "...", "processing": ["summary", "keywords"]}
```

**Processing Strategy:**
- Immediate: Stream transcription as it's generated
- Background: Trigger summary on 30+ second pauses
- Progressive: Update keywords as content accumulates
- Adaptive: Adjust timing based on speech patterns

### Phase 6: Chat with Notes
**Goal:** RAG-based chat system for note interaction

**Implementation Tasks:**
- Create chat.rs module with conversation session management
- Implement semantic search using vector embeddings or BM25 scoring for note retrieval
- Setup context window management for efficient token usage with LLM
- Create retrieval-augmented generation pipeline: query → search → context → response
- Add conversation history tracking with context preservation across queries
- Implement source attribution linking responses back to specific notes/courses
- Create specialized query handlers: factual questions, clarification, study assistance
- Add query classification to route different question types appropriately

**RAG Implementation:**
1. Process user query to extract intent and key terms
2. Search relevant notes using FTS5 + semantic similarity
3. Rank and select top-k most relevant note excerpts
4. Construct prompt with user query + context + conversation history
5. Generate response with source citations and confidence scores

**Context Management:**
- Maintain sliding window of conversation history
- Include course/week/topic context for academic relevance
- Smart context truncation to fit LLM token limits
- Cache frequent queries and responses

**Endpoints:**
- `WebSocket /ws/chat` - Real-time chat with typing indicators, message status
- `POST /chat/query` - Single question with context specification
- `GET /chat/sessions` - List conversation sessions with metadata
- `POST /chat/context` - Set course/week/topic context for focused responses

### Phase 7: Categorization
**Goal:** Intelligent note organization and recommendations

**Implementation Tasks:**
- Create categorization.rs with academic subject classification using LLM
- Implement content analysis for difficulty assessment, prerequisite detection
- Add recommendation engine suggesting related notes and study sequences
- Create tagging system with automatic tag generation and manual override
- Implement knowledge gap analysis by comparing course content coverage
- Add study priority scoring based on upcoming deadlines, content difficulty
- Create export/import functionality for academic content portability
- Add analytics for study patterns, content engagement, learning progress

**Content Analysis Features:**
- Academic subject detection using educational taxonomy
- Difficulty scoring based on vocabulary complexity, concept density
- Prerequisite mapping using content dependency analysis
- Study sequence optimization based on learning pathways

**Recommendation System:**
- Related note suggestions using content similarity
- Study priority ranking with deadline awareness
- Knowledge gap identification and remediation suggestions
- Progress tracking and achievement analytics

### Phase 8: Performance Optimization
**Goal:** Production-ready performance and monitoring

**Implementation Tasks:**
- Implement multi-level caching: in-memory LRU, disk-based persistence
- Add comprehensive metrics collection using prometheus-style metrics
- Create database query optimization with explain analysis and index tuning
- Implement connection pooling for all external services (Ollama, database)
- Add memory usage optimization for long-running sessions and large notes
- Create performance profiling endpoints for bottleneck identification
- Implement graceful degradation under high load conditions
- Add alerting system for performance thresholds and error rates

**Caching Strategy:**
- Transcription results: Cache recent audio fingerprints to avoid re-processing
- LLM responses: Cache summary/keyword results with content-based invalidation
- Database queries: Cache frequent searches and note lookups
- Model outputs: Cache embeddings and processed content

**Monitoring Implementation:**
- Request latency histograms for all endpoints
- Concurrent session tracking and resource usage
- Model inference timing and success rates
- Database query performance and connection health
- Memory usage patterns and garbage collection metrics

## Success Criteria
- Sub-second transcription latency
- Real-time LLM processing
- Efficient local-only operation
- Academic workflow optimization
- Stable concurrent processing