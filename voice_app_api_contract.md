# Voice Recording + Note Taking App - API Contract

## Overview
Complete API specification for real-time voice recording and note-taking application. Defines all REST endpoints, WebSocket protocols, data models, and error formats for frontend-backend communication.

**Base URL:** `http://localhost:8080`
**WebSocket URL:** `ws://localhost:8080`

## Data Models

### Core Entities
```typescript
interface Course {
  id: number;
  name: string;
  description: string;
  created_at: string; // ISO 8601
  updated_at: string; // ISO 8601
}

interface Week {
  id: number;
  course_id: number;
  week_number: number;
  title: string;
  start_date: string; // ISO 8601
  end_date: string; // ISO 8601
}

interface Topic {
  id: number;
  week_id: number;
  title: string;
  description: string;
  created_at: string; // ISO 8601
}

interface Note {
  id: number;
  topic_id: number;
  title: string;
  content: string; // Markdown content
  transcript: string; // Raw transcribed text
  created_at: string; // ISO 8601
  updated_at: string; // ISO 8601
  metadata_json: Record<string, any>; // Summaries, keywords, etc.
}

interface TranscriptionSession {
  id: number;
  note_id: number;
  start_time: string; // ISO 8601
  end_time: string | null; // ISO 8601
  status: "recording" | "processing" | "completed" | "error";
  model_used: string;
}
```

### Response Wrappers
```typescript
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string; // ISO 8601
}

interface PaginatedResponse<T> {
  success: boolean;
  data: T[];
  pagination: {
    total: number;
    page: number;
    limit: number;
    total_pages: number;
  };
  timestamp: string;
}

interface ErrorResponse {
  success: false;
  error: string;
  code: string;
  details?: Record<string, any>;
  timestamp: string;
}
```

## REST API Endpoints

### System Management

#### Health Check
```http
GET /health
```
**Response:**
```typescript
{
  success: true,
  data: {
    status: "healthy" | "degraded" | "unhealthy",
    uptime: number, // seconds
    models: {
      whisper: {
        loaded: string | null,
        available: string[],
        status: "ready" | "loading" | "error"
      },
      llm: {
        loaded: string | null,
        available: string[],
        status: "ready" | "loading" | "error"
      }
    },
    memory_usage: {
      used: number, // bytes
      available: number // bytes
    }
  }
}
```

#### Configuration
```http
GET /config
```
**Response:**
```typescript
{
  success: true,
  data: {
    server: {
      host: string,
      port: number
    },
    models: {
      whisper_model: string,
      llm_model: string
    },
    performance: {
      max_concurrent_sessions: number,
      audio_buffer_size: number
    }
  }
}
```

```http
PUT /config
Content-Type: application/json
```
**Request:**
```typescript
{
  models?: {
    whisper_model?: string,
    llm_model?: string
  },
  performance?: {
    max_concurrent_sessions?: number,
    audio_buffer_size?: number
  }
}
```

#### Metrics
```http
GET /metrics
```
**Response:**
```typescript
{
  success: true,
  data: {
    requests: {
      total: number,
      per_second: number,
      errors: number
    },
    sessions: {
      active: number,
      total: number
    },
    models: {
      whisper_inference_time: number, // ms average
      llm_inference_time: number // ms average
    }
  }
}
```

### Audio & Transcription

#### Model Management
```http
GET /models/whisper
```
**Response:**
```typescript
{
  success: true,
  data: {
    current: string | null,
    available: {
      name: string,
      size: string,
      description: string,
      performance: "fast" | "balanced" | "accurate"
    }[]
  }
}
```

```http
POST /models/whisper/load
Content-Type: application/json
```
**Request:**
```typescript
{
  model: string
}
```

#### File Transcription (Testing)
```http
POST /transcribe/file
Content-Type: multipart/form-data
```
**Request:** Audio file in form data
**Response:**
```typescript
{
  success: true,
  data: {
    text: string,
    confidence: number,
    processing_time: number // seconds
  }
}
```

### Academic Content Management

#### Courses
```http
GET /courses
GET /courses?search={query}
```
**Response:** `PaginatedResponse<Course>`

```http
POST /courses
Content-Type: application/json
```
**Request:**
```typescript
{
  name: string,
  description: string
}
```
**Response:** `ApiResponse<Course>`

```http
PUT /courses/{id}
DELETE /courses/{id}
GET /courses/{id}
```

```http
GET /courses/{id}/stats
```
**Response:**
```typescript
{
  success: true,
  data: {
    total_weeks: number,
    total_topics: number,
    total_notes: number,
    total_words: number,
    recent_activity: {
      date: string,
      action: string,
      item_type: string,
      item_name: string
    }[]
  }
}
```

#### Weeks
```http
GET /weeks?course_id={id}
POST /weeks
PUT /weeks/{id}
DELETE /weeks/{id}
GET /weeks/{id}
```
**POST Request:**
```typescript
{
  course_id: number,
  week_number: number,
  title: string,
  start_date: string, // ISO 8601
  end_date: string // ISO 8601
}
```

#### Topics
```http
GET /topics?week_id={id}
POST /topics
PUT /topics/{id}
DELETE /topics/{id}
GET /topics/{id}
```
**POST Request:**
```typescript
{
  week_id: number,
  title: string,
  description: string
}
```

#### Notes
```http
GET /notes?topic_id={id}
POST /notes
PUT /notes/{id}
DELETE /notes/{id}
GET /notes/{id}
```
**POST Request:**
```typescript
{
  topic_id: number,
  title: string,
  content?: string,
  transcript?: string
}
```

#### Search
```http
GET /search?q={query}&course_id={id}&limit={limit}&offset={offset}
```
**Response:**
```typescript
{
  success: true,
  data: {
    notes: {
      id: number,
      title: string,
      content_snippet: string,
      highlight: string,
      course_name: string,
      week_title: string,
      topic_title: string,
      relevance_score: number
    }[],
    total: number
  }
}
```

### LLM Processing

#### Content Processing
```http
POST /llm/summarize
Content-Type: application/json
```
**Request:**
```typescript
{
  note_id?: number,
  content?: string,
  max_length?: number
}
```
**Response:**
```typescript
{
  success: true,
  data: {
    summary: string,
    processing_time: number,
    confidence: number
  }
}
```

```http
POST /llm/keywords
Content-Type: application/json
```
**Request:**
```typescript
{
  note_id?: number,
  content?: string,
  max_keywords?: number
}
```
**Response:**
```typescript
{
  success: true,
  data: {
    keywords: {
      keyword: string,
      relevance: number,
      category?: string
    }[],
    processing_time: number
  }
}
```

#### Model Management
```http
GET /llm/models
```
**Response:**
```typescript
{
  success: true,
  data: {
    current: string | null,
    available: {
      name: string,
      size: string,
      capabilities: string[],
      performance: "fast" | "balanced" | "accurate"
    }[]
  }
}
```

#### Batch Processing
```http
POST /llm/process-batch
Content-Type: application/json
```
**Request:**
```typescript
{
  note_ids: number[],
  operations: ("summarize" | "keywords")[]
}
```
**Response:**
```typescript
{
  success: true,
  data: {
    job_id: string,
    estimated_time: number, // seconds
    status: "queued"
  }
}
```

### Chat System

#### Chat Operations
```http
POST /chat/query
Content-Type: application/json
```
**Request:**
```typescript
{
  query: string,
  context?: {
    course_id?: number,
    week_id?: number,
    topic_id?: number,
    note_id?: number
  },
  session_id?: string
}
```
**Response:**
```typescript
{
  success: true,
  data: {
    response: string,
    sources: {
      note_id: number,
      title: string,
      excerpt: string,
      relevance: number
    }[],
    session_id: string
  }
}
```

```http
GET /chat/sessions
GET /chat/sessions/{session_id}
```

```http
POST /chat/context
Content-Type: application/json
```
**Request:**
```typescript
{
  session_id: string,
  context: {
    course_id?: number,
    week_id?: number,
    topic_id?: number,
    note_id?: number
  }
}
```

## WebSocket Protocols

### Audio Streaming (`/ws/audio`)

#### Connection Setup
```typescript
// Connection headers
{
  "Sec-WebSocket-Protocol": "audio-transcription"
}
```

#### Inbound Messages (Frontend → Backend)
```typescript
// Start recording session
{
  type: "start_session",
  session_id: string,
  note_id?: number,
  audio_config: {
    sample_rate: 16000,
    channels: 1,
    bit_depth: 16
  }
}

// Audio data chunk
{
  type: "audio_data",
  session_id: string,
  data: ArrayBuffer, // PCM audio data
  timestamp: number
}

// Pause/resume session
{
  type: "pause_session" | "resume_session",
  session_id: string
}

// End session
{
  type: "end_session",
  session_id: string
}
```

#### Outbound Messages (Backend → Frontend)
```typescript
// Session status
{
  type: "session_status",
  session_id: string,
  status: "started" | "recording" | "paused" | "processing" | "completed" | "error",
  message?: string
}

// Real-time transcription
{
  type: "transcription",
  session_id: string,
  text: string,
  confidence: number,
  is_partial: boolean,
  timestamp: number
}

// Processing updates
{
  type: "processing_status",
  session_id: string,
  operations: {
    operation: "summary" | "keywords",
    status: "queued" | "processing" | "completed" | "error",
    progress?: number, // 0-100
    estimated_time?: number // seconds remaining
  }[]
}

// Generated summary
{
  type: "summary",
  session_id: string,
  content: string,
  processing_time: number,
  confidence: number
}

// Extracted keywords
{
  type: "keywords",
  session_id: string,
  keywords: {
    keyword: string,
    relevance: number,
    category?: string
  }[],
  processing_time: number
}

// Error handling
{
  type: "error",
  session_id: string,
  error: string,
  code: string,
  recoverable: boolean
}
```

### Chat Streaming (`/ws/chat`)

#### Inbound Messages (Frontend → Backend)
```typescript
// Start chat session
{
  type: "start_chat",
  session_id: string,
  context?: {
    course_id?: number,
    week_id?: number,
    topic_id?: number,
    note_id?: number
  }
}

// Send message
{
  type: "message",
  session_id: string,
  content: string,
  message_id: string
}

// Update context
{
  type: "update_context",
  session_id: string,
  context: {
    course_id?: number,
    week_id?: number,
    topic_id?: number,
    note_id?: number
  }
}

// Typing indicator
{
  type: "typing",
  session_id: string,
  is_typing: boolean
}
```

#### Outbound Messages (Backend → Frontend)
```typescript
// Chat response (streaming)
{
  type: "response_chunk",
  session_id: string,
  message_id: string,
  content: string,
  is_complete: boolean
}

// Complete response with sources
{
  type: "response_complete",
  session_id: string,
  message_id: string,
  sources: {
    note_id: number,
    title: string,
    excerpt: string,
    relevance: number
  }[],
  processing_time: number
}

// Context update confirmation
{
  type: "context_updated",
  session_id: string,
  context: {
    course_name?: string,
    week_title?: string,
    topic_title?: string,
    note_title?: string
  }
}

// Typing indicator
{
  type: "typing",
  session_id: string,
  is_typing: boolean
}
```

## Error Codes

```typescript
enum ErrorCode {
  // General
  INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR",
  INVALID_REQUEST = "INVALID_REQUEST",
  NOT_FOUND = "NOT_FOUND",
  VALIDATION_ERROR = "VALIDATION_ERROR",
  
  // Audio/Transcription
  AUDIO_FORMAT_UNSUPPORTED = "AUDIO_FORMAT_UNSUPPORTED",
  MODEL_NOT_LOADED = "MODEL_NOT_LOADED",
  MODEL_LOADING_FAILED = "MODEL_LOADING_FAILED",
  TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED",
  SESSION_NOT_FOUND = "SESSION_NOT_FOUND",
  
  // LLM Processing
  LLM_UNAVAILABLE = "LLM_UNAVAILABLE",
  LLM_TIMEOUT = "LLM_TIMEOUT",
  CONTEXT_TOO_LARGE = "CONTEXT_TOO_LARGE",
  PROCESSING_FAILED = "PROCESSING_FAILED",
  
  // Database
  DATABASE_ERROR = "DATABASE_ERROR",
  FOREIGN_KEY_CONSTRAINT = "FOREIGN_KEY_CONSTRAINT",
  DUPLICATE_ENTRY = "DUPLICATE_ENTRY",
  
  // Chat
  CHAT_SESSION_NOT_FOUND = "CHAT_SESSION_NOT_FOUND",
  INVALID_CONTEXT = "INVALID_CONTEXT"
}
```

## Rate Limits

- **REST API**: 100 requests per minute per client
- **WebSocket Audio**: 1 concurrent session per client
- **WebSocket Chat**: 10 messages per minute per session
- **LLM Processing**: 20 requests per minute per client

## Authentication

Currently no authentication required. All endpoints accept requests without authorization headers.

**Future:** JWT-based authentication with header: `Authorization: Bearer <token>`