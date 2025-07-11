# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust backend for a voice recording and note-taking application designed for students. The system provides real-time audio transcription using whisper.cpp, LLM-powered note enhancement via Ollama, and academic content organization with SQLite FTS5 search.

**Tech Stack:** Actix-web, whisper.cpp, Ollama, SQLite with FTS5

## Development Commands

```bash
# Build the project
cargo build

# Run the development server
cargo run

# Run tests
cargo test

# Check code (lint equivalent)
cargo check

# Format code
cargo fmt

# Run clippy for additional linting
cargo clippy

# Build for release
cargo build --release
```

## Project Architecture

The application follows a modular architecture with planned phases:

### Core Components Structure

- **config.rs** - Configuration management (TOML/JSON + env vars)
- **health.rs** - System status and model availability checks
- **audio.rs** - WebSocket audio streaming and processing
- **transcription.rs** - Whisper.cpp integration for real-time transcription
- **database.rs** - SQLite with FTS5 for academic content storage
- **models.rs** - Database schema structs and validation
- **ollama.rs** - LLM integration for summarization and keywords
- **chat.rs** - RAG-based chat system for note interaction
- **categorization.rs** - Academic content classification and recommendations

### Data Hierarchy

Academic content is organized as: **Courses → Weeks → Topics → Notes**

### Database Schema

```sql
Courses: id, name, description, created_at, updated_at
Weeks: id, course_id, week_number, title, start_date, end_date
Topics: id, week_id, title, description, created_at
Notes: id, topic_id, title, content, transcript, created_at, updated_at, metadata_json
Transcription_Sessions: id, note_id, start_time, end_time, status, model_used
```

## Configuration Structure

The application uses TOML configuration with the following structure:

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

## Key Endpoints by Phase

### Phase 1: Foundation Service

- `GET /health` - Service status, model status, memory usage
- `GET /config` - Current configuration display
- `PUT /config` - Runtime configuration updates
- `GET /metrics` - Performance statistics

### Phase 2: Audio Pipeline

- `WebSocket /ws/audio` - Binary audio in, JSON transcription out
- `GET /models/whisper` - List available models
- `POST /models/whisper/load` - Load specific model

### Phase 3: Data Management

- `GET/POST/PUT/DELETE /courses, /weeks, /topics, /notes` - CRUD operations
- `GET /search?q={query}&course_id={id}` - Full-text search

### Phase 4: LLM Integration

- `POST /llm/summarize` - Generate summary for content
- `POST /llm/keywords` - Extract keywords
- `GET /llm/models` - Available models

### Phase 6: Chat with Notes

- `WebSocket /ws/chat` - Real-time chat interface
- `POST /chat/query` - Single question with context

## WebSocket Message Protocol

The application uses structured JSON messages for real-time communication:

```json
{"type": "transcription", "text": "...", "confidence": 0.95, "timestamp": 1234567890}
{"type": "summary", "content": "...", "processing_time": 1.2}
{"type": "keywords", "keywords": ["concept1", "concept2"], "relevance": [0.9, 0.8]}
{"type": "status", "session_id": "...", "processing": ["summary", "keywords"]}
```

## Development Notes

- No audio storage (real-time processing only)
- Concurrent processing using tokio for non-blocking operations
- Audio format: PCM 16-bit, 16kHz
- FTS5 virtual tables for full-text search across content
- Background job queues for LLM processing
- Real-time transcription with 1-2 second audio overlap for context
- RAG-based chat system with conversation history tracking

## Recommended Models

- **Whisper:** Medium model (good accuracy for technical vocabulary)
- **LLM:** Llama 3.1 8B (primary), Phi-3 Medium (alternative)

## Rust Best Practices (2025)

### Code Style and Formatting

- **Use `rustfmt`**: Always format code automatically. Each level of indentation must be 4 spaces
- **Maximum line width**: 100 characters
- **Comments**: Prefer line comments (`//`) over block comments (`/* */`). Use `///` for documentation comments
- **Use `clippy`**: Run `cargo clippy` to catch common mistakes and get improvement suggestions
- **Documentation**: Add concise documentation for all methods and variables, and mention what is being done in each complex logic

### Naming Conventions

- **Clear descriptive names**: Choose variable, function, and type names that clearly describe their purpose
- **Avoid abbreviations**: Don't use single-letter names unless universally understood (like `i` for iteration)
- **Crate naming**: Don't use `-rs` or `-rust` as suffix/prefix in crate names

### Error Handling Best Practices

1. **Avoid `unwrap()` in production**: Use graceful error handling methods instead
2. **Prefer `expect()` over `unwrap()`**: When panicking is acceptable, provide descriptive error messages
3. **Use `?` operator**: For error propagation in functions returning `Result` or `Option`
4. **Pattern matching**: Use `match` or `if let` for explicit error handling
5. **Graceful fallbacks**: Use `unwrap_or()`, `unwrap_or_else()` for default values
6. **Modern `let-else` statements**: For early returns when extracting values from `Option`/`Result`

```rust
// Good: Graceful error handling
let user = match get_user(id) {
    Some(user) => user,
    None => return Err("User not found".into()),
};

// Good: Using let-else (2025 pattern)
let Some(user) = get_user(id) else {
    return Err("User not found".into());
};

// Good: Error propagation
let data = fetch_data()?;
```

### Ownership and Borrowing

- **Minimize borrowing complexity**: Keep borrowing simple, avoid multiple reference types simultaneously
- **Borrow only when needed**: Don't hold references longer than necessary
- **Function parameters**: Prefer borrowing (`&T`) over owned types when function doesn't need ownership
- **Use explicit lifetime annotations**: When compiler can't infer lifetimes or multiple input references exist
- **Memory safety first**: Leverage Rust's ownership system to prevent use-after-free and memory leaks

```rust
// Good: Borrowing for read-only access
fn process_data(data: &[u8]) -> Result<String, Error> {
    // Function can read data without taking ownership
}

// Good: Taking ownership when needed
fn consume_data(data: Vec<u8>) -> Result<(), Error> {
    // Function needs to own and potentially move data
}
```

### Function Design

- **Single responsibility**: Each function should perform one clear task
- **Small functions**: Break complex logic into smaller, testable functions
- **Return `Result<T, E>`**: For operations that can fail
- **Use `Option<T>`**: For values that may or may not exist

### Async Programming

- **Use `tokio`**: For non-blocking operations and concurrent processing
- **Async closures**: Use modern `async ||` blocks (2024+ edition feature)
- **Background tasks**: Use `tokio::spawn` for concurrent processing
- **Resource management**: Ensure proper cleanup in async contexts

### Security Practices

- **Minimize `unsafe`**: Only use when absolutely necessary and document why
- **Review `unsafe` code**: Thoroughly audit any unsafe blocks for undefined behavior
- **Input validation**: Always validate external input and API parameters
- **No secrets in code**: Never commit API keys, tokens, or sensitive data

### Testing

- **Unit tests**: Use `#[test]` attribute for comprehensive unit testing
- **Integration tests**: Create separate test files for integration scenarios
- **Property-based testing**: Consider using `proptest` for complex scenarios
- **Error case testing**: Test failure conditions and edge cases

### Documentation

- **Module documentation**: Document public APIs with `///` comments
- **Examples in docs**: Include usage examples in documentation
- **README updates**: Keep project documentation current
- **Code comments**: Explain why, not what (the code shows what)

### Performance Considerations

- **Profile before optimizing**: Use tools like `cargo bench` and `perf`
- **Choose appropriate data structures**: Vec vs HashMap vs BTreeMap based on access patterns
- **Minimize allocations**: Reuse buffers and use iterators when possible
- **Connection pooling**: For database and external service connections
