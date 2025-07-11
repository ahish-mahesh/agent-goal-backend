# Agent Notes Backend

A high-performance Rust backend for a voice recording and note-taking application designed for students. This service provides real-time audio transcription, LLM-powered note enhancement, and academic content organization.

## 🚀 Current Status: Phase 1 Complete

**✅ Foundation Service (Phase 1)** - Core Actix-web service with configuration and monitoring
- HTTP server with configurable host/port binding
- Structured logging with tracing
- Health monitoring and metrics collection
- Graceful shutdown handling (SIGTERM/SIGINT)
- CORS, request logging, and error handling middleware
- Runtime configuration updates via API
- Comprehensive beginner-friendly documentation

## 🛠 Tech Stack

- **Web Framework**: Actix-web 4.9
- **Configuration**: TOML files + environment variables (serde)
- **Logging**: Tracing with structured output
- **Error Handling**: Custom error types with HTTP response mapping
- **Concurrency**: Tokio async runtime with Arc<RwLock> shared state
- **Future Integrations**: whisper.cpp, Ollama, SQLite with FTS5

## 📋 API Endpoints

### Foundation Service (Phase 1)
- `GET /health` - Service status, model status, memory usage, uptime
- `GET /api/v1/config` - Current configuration display
- `PUT /api/v1/config` - Runtime configuration updates with validation
- `GET /api/v1/metrics` - Detailed performance metrics and endpoint statistics

## ⚙️ Configuration

The application uses a layered configuration system with the following priority (highest to lowest):

1. **Environment variables** (APP_SERVER_HOST, APP_SERVER_PORT, etc.)
2. **Configuration file** (config.toml)
3. **Default values** (built into code)

### Example config.toml
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

### Environment Variables
```bash
# Server configuration
export APP_SERVER_HOST="0.0.0.0"
export APP_SERVER_PORT="3000"

# Or use deployment platform conventions
export HOST="0.0.0.0"
export PORT="3000"

# Model configuration  
export APP_MODELS_WHISPER_MODEL="large"
export APP_MODELS_LLM_MODEL="llama3.1:8b"

# Logging
export RUST_LOG="agent_notes_backend=debug,actix_web=info"
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Rust 1.70+ 
- Cargo

### Development
```bash
# Clone and navigate to project
cd agent-notes-backend

# Run with default configuration
cargo run

# Run with custom configuration
RUST_LOG=debug cargo run

# Run tests
cargo test

# Check code quality
cargo check
cargo clippy
cargo fmt
```

### Production
```bash
# Build optimized binary
cargo build --release

# Run with environment configuration
export HOST="0.0.0.0"
export PORT="8080"
./target/release/agent-notes-backend
```

## 🏗 Architecture

### Data Hierarchy
Academic content is organized as: **Courses → Weeks → Topics → Notes**

### Key Rust Patterns Used
- **Arc<RwLock<T>>**: Thread-safe shared state management
- **Custom Error Types**: Structured error handling with HTTP mapping
- **Async/Await**: Non-blocking request processing
- **Middleware Pattern**: Request logging, metrics, and CORS
- **Builder Pattern**: Configuration loading and HTTP responses

### Project Structure
```
src/
├── main.rs           # Server setup, async runtime, graceful shutdown
├── config.rs         # Configuration management (TOML + env vars)  
├── error.rs          # Custom error types and HTTP error handling
├── state.rs          # Thread-safe shared application state
├── health.rs         # Health monitoring and system metrics
├── middleware/       # Custom middleware (logging, metrics)
│   ├── mod.rs
│   ├── logging.rs    # Request logging middleware
│   └── metrics.rs    # Performance metrics collection
└── handlers/         # HTTP request handlers
    ├── mod.rs
    └── config.rs     # Configuration API handlers
```

## 📈 Roadmap

### Phase 2: Audio Pipeline (Next)
- WebSocket audio streaming
- whisper.cpp integration for real-time transcription
- Audio preprocessing and buffering
- Model management (load/unload)

### Phase 3: Data Management
- SQLite database with FTS5 search
- Academic content CRUD APIs
- Database migrations and connection pooling

### Phase 4: LLM Integration
- Ollama integration for text processing
- Background job queues for non-blocking processing
- Prompt templates for academic content

### Phase 5: Real-time Integration
- Combined audio transcription + LLM processing
- WebSocket message multiplexing
- Progressive enhancement (transcription → summary → keywords)

### Phase 6: Chat with Notes
- RAG-based chat system using note content
- Conversation history and context management
- Source attribution and confidence scoring

## 🧠 For Rust Beginners

This project serves as an excellent learning resource for Rust concepts:

- **Comprehensive documentation**: Every major file includes beginner explanations
- **Real-world patterns**: Arc<RwLock>, async/await, error handling, traits
- **Best practices**: Following 2025 Rust coding standards
- **Production patterns**: Configuration, logging, graceful shutdown, testing

Key learning files:
- `src/main.rs` - Async programming and server setup
- `src/state.rs` - Thread-safe shared state management  
- `src/error.rs` - Custom error types and HTTP responses
- `src/config.rs` - Serde and configuration management

## 🧪 Testing

```bash
# Run all tests
cargo test

# Test with output
cargo test -- --nocapture

# Test specific module
cargo test config::tests
```

## 📊 Monitoring

The application provides comprehensive monitoring:

- **Health checks**: `/health` endpoint with system status
- **Performance metrics**: Request counts, response times, error rates  
- **Per-endpoint statistics**: Detailed metrics for each API endpoint
- **Memory usage**: Platform-specific memory monitoring (Linux)
- **Uptime tracking**: Server start time and duration

## 🤝 Contributing

1. Follow the existing code style and documentation patterns
2. Add tests for new functionality
3. Update documentation for any new concepts
4. Run `cargo clippy` and `cargo fmt` before committing
5. Ensure all tests pass with `cargo test`

## 📄 License

[Add your license here]
