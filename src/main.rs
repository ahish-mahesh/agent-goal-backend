//! # Agent Notes Backend - Main Application Entry Point
//! 
//! This is the main entry point for the agent-notes-backend web server.
//! It sets up an Actix-web HTTP server with the following key features:
//! 
//! ## Key Rust Concepts Used:
//! - **async/await**: The entire application is asynchronous for better performance
//! - **modules**: Code is organized into separate modules (mod statements)
//! - **Result<T, E>**: Error handling using Rust's Result type
//! - **Arc & RwLock**: Thread-safe shared state management
//! - **static**: Global variables that live for the entire program duration
//! 
//! ## Application Architecture:
//! - **config**: Handles application configuration (TOML files + environment variables)
//! - **state**: Manages shared application state and metrics
//! - **health**: Provides system health monitoring endpoints
//! - **middleware**: Custom request processing logic (logging, metrics)
//! - **handlers**: HTTP request handlers for API endpoints
//! - **error**: Custom error types and HTTP error responses

// Module declarations - These tell Rust about our other source files
mod config;      // Configuration management (config.rs)
mod error;       // Error handling types (error.rs) 
mod state;       // Application state management (state.rs)
mod health;      // Health check endpoints (health.rs)
mod middleware;  // Custom middleware (middleware/ directory)
mod handlers;    // HTTP request handlers (handlers/ directory)

// External crate imports - These are dependencies from Cargo.toml
use actix_cors::Cors;  // Cross-Origin Resource Sharing support
use actix_web::{web, App, HttpServer, middleware::Logger};  // Web framework
use anyhow::Result;    // Better error handling with context
use config::AppConfig; // Our custom configuration struct
use state::AppState;   // Our custom application state
use std::sync::atomic::{AtomicBool, Ordering};  // Thread-safe boolean for shutdown
use tracing::{info, error};  // Structured logging
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};  // Logging setup

/// Global shutdown signal that can be accessed from anywhere in the program.
/// AtomicBool is thread-safe, meaning multiple threads can safely read/write to it.
/// This is used to signal when the server should gracefully shut down.
static SHUTDOWN_SIGNAL: AtomicBool = AtomicBool::new(false);

/// The main application entry point.
/// 
/// ## What this function does:
/// 1. **Loads configuration** from files and environment variables
/// 2. **Sets up logging** for debugging and monitoring
/// 3. **Creates shared application state** that all requests can access
/// 4. **Configures the HTTP server** with middleware and routes
/// 5. **Handles graceful shutdown** when receiving system signals
/// 
/// ## Key Rust Concepts:
/// - `#[actix_web::main]`: This macro sets up the async runtime (like a JavaScript event loop)
/// - `async fn`: This function can be paused and resumed, allowing other work to happen
/// - `Result<()>`: This function returns either success (Ok(())) or an error (Err(error))
/// - `?`: The question mark operator automatically returns early if there's an error
/// 
/// ## Error Handling:
/// If any step fails (config loading, server binding, etc.), the function returns an error
/// and the program exits gracefully with an error message.
#[actix_web::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file (if it exists)
    // .ok() means "ignore errors" - it's fine if there's no .env file
    dotenv::dotenv().ok();
    
    // Set up structured logging (tracing) for debugging and monitoring
    // The ? operator means "if this fails, return the error immediately"
    init_tracing()?;
    
    // Load application configuration from config.toml and environment variables
    let config = AppConfig::load()?;
    // Validate that the configuration makes sense (e.g., port isn't 0)
    config.validate()?;
    
    // Log startup information - these appear in the console when you run the server
    info!("Starting agent-notes-backend v{}", env!("CARGO_PKG_VERSION"));
    info!("Configuration loaded: {}:{}", config.server.host, config.server.port);
    
    // Create the shared application state that all HTTP requests can access
    // This includes configuration, metrics, and the server start time
    let app_state = AppState::new(config.clone());
    let bind_addr = format!("{}:{}", config.server.host, config.server.port);
    
    // Set up signal handlers for graceful shutdown (Ctrl+C, SIGTERM, etc.)
    setup_signal_handlers();
    
    info!("Starting HTTP server on {}", bind_addr);
    
    // Create the HTTP server with all its configuration
    let server = HttpServer::new(move || {
        // Configure CORS (Cross-Origin Resource Sharing) to allow web browsers to connect
        let cors = Cors::default()
            .allow_any_origin()    // Allow requests from any domain
            .allow_any_method()    // Allow GET, POST, PUT, DELETE, etc.
            .allow_any_header()    // Allow any HTTP headers
            .max_age(3600);        // Cache CORS settings for 1 hour

        // Create the main application with all its configuration
        App::new()
            // Share our application state with all request handlers
            .app_data(web::Data::new(app_state.clone()))
            // Add middleware in order (they execute in reverse order for responses)
            .wrap(cors)                                    // Handle CORS
            .wrap(Logger::default())                       // Log HTTP requests
            .wrap(middleware::MetricsMiddleware)           // Collect performance metrics
            .wrap(middleware::RequestLogging)              // Custom request logging
            // Define API routes under /api/v1 prefix
            .service(
                web::scope("/api/v1")
                    .route("/health", web::get().to(health::health_check))
                    .route("/metrics", web::get().to(health::detailed_metrics))
                    .route("/config", web::get().to(handlers::get_config))
                    .route("/config", web::put().to(handlers::update_config))
            )
            // Also provide health check at root level for convenience
            .route("/health", web::get().to(health::health_check))
    })
    .bind(&bind_addr)?  // Bind to the configured host and port
    .run();             // Start the server (but don't block here)

    // Get a handle to control the server and spawn it in a separate task
    let server_handle = server.handle();
    let server_task = tokio::spawn(server);

    // Wait for either the server to finish OR a shutdown signal
    // tokio::select! is like a "race" - whichever finishes first wins
    tokio::select! {
        // If the server task finishes (which usually means an error)
        result = server_task => {
            match result {
                Ok(server_result) => {
                    if let Err(e) = server_result {
                        error!("Server error: {}", e);
                    }
                }
                Err(e) => {
                    error!("Server task error: {}", e);
                }
            }
        }
        // If we receive a shutdown signal (Ctrl+C, SIGTERM, etc.)
        _ = wait_for_shutdown() => {
            info!("Shutdown signal received, stopping server...");
            server_handle.stop(true).await;  // Gracefully stop the server
        }
    }

    info!("Server stopped gracefully");
    Ok(())  // Return success
}

/// Initialize the tracing (logging) system for the application.
/// 
/// ## What this does:
/// - Sets up structured logging that outputs to the console
/// - Configures log levels (debug, info, warn, error)
/// - Reads log configuration from environment variables
/// 
/// ## Environment Variables:
/// - `RUST_LOG`: Controls what gets logged (e.g., "debug", "info", "agent_notes_backend=debug")
/// - If not set, defaults to "agent_notes_backend=debug,actix_web=info"
/// 
/// ## Rust Concepts:
/// - **Builder pattern**: Methods chained together to configure the logger
/// - **unwrap_or_else**: Provides a default value if the environment variable doesn't exist
/// - **into()**: Converts the string into the expected type
fn init_tracing() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            // Try to read RUST_LOG environment variable, or use defaults
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "agent_notes_backend=debug,actix_web=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())  // Format logs nicely for console output
        .init();  // Actually start the logging system

    Ok(())
}

/// Set up signal handlers for graceful shutdown.
/// 
/// ## What this does:
/// - Listens for SIGTERM (termination signal from system)
/// - Listens for SIGINT (interrupt signal, usually Ctrl+C)
/// - When either signal is received, sets the global shutdown flag
/// 
/// ## Rust Concepts:
/// - **tokio::spawn**: Runs this code in a separate async task
/// - **async block**: Code that can be paused and resumed
/// - **tokio::select!**: Waits for whichever signal comes first
/// - **Ordering::SeqCst**: Memory ordering for atomic operations (ensures thread safety)
/// 
/// ## Why this matters:
/// Graceful shutdown means the server can finish processing current requests
/// before shutting down, rather than just stopping immediately.
fn setup_signal_handlers() {
    tokio::spawn(async {
        // Set up handlers for different types of shutdown signals
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler");
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("Failed to install SIGINT handler");

        // Wait for either signal to arrive
        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM");
            }
            _ = sigint.recv() => {
                info!("Received SIGINT");
            }
        }

        // Set the global shutdown flag so other parts of the program know to stop
        // SeqCst (Sequential Consistency) ensures this change is visible to all threads
        SHUTDOWN_SIGNAL.store(true, Ordering::SeqCst);
    });
}

/// Wait for the shutdown signal to be set.
/// 
/// ## What this does:
/// - Continuously checks if the shutdown signal has been set
/// - Sleeps for 100ms between checks to avoid busy-waiting
/// - Returns when shutdown has been requested
/// 
/// ## Rust Concepts:
/// - **async fn**: This function can be paused without blocking the thread
/// - **while loop**: Keeps running until the condition becomes false
/// - **Ordering::SeqCst**: Safe way to read atomic variables from multiple threads
/// - **tokio::time::sleep**: Async sleep that doesn't block other tasks
/// 
/// ## Why polling instead of events:
/// This is a simple polling approach. In a more complex system, you might use
/// async channels or other event-driven mechanisms.
async fn wait_for_shutdown() {
    while !SHUTDOWN_SIGNAL.load(Ordering::SeqCst) {
        // Sleep for 100ms, then check again
        // This prevents busy-waiting (constantly checking) which would waste CPU
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}
