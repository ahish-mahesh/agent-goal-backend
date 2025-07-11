//! # Application State Management
//! 
//! This module manages shared state that needs to be accessed by multiple HTTP request handlers
//! simultaneously. This is one of the most complex parts of the application from a Rust perspective.
//! 
//! ## Key Rust Concepts (IMPORTANT for beginners):
//! 
//! ### Arc (Atomically Reference Counted)
//! - **Purpose**: Allows multiple parts of the program to safely share ownership of data
//! - **Why needed**: Multiple HTTP requests run simultaneously and all need access to the same state
//! - **Memory safety**: Automatically cleans up data when the last reference is dropped
//! - **Thread safety**: Safe to share between threads
//! 
//! ### RwLock (Reader-Writer Lock)
//! - **Purpose**: Allows multiple readers OR one writer at a time (but not both)
//! - **Why needed**: Multiple requests can read config simultaneously, but only one can update it
//! - **Performance**: Reading is fast (no blocking), writing blocks everything else
//! - **Deadlock prevention**: Rust's type system helps prevent common locking mistakes
//! 
//! ### Arc<RwLock<T>> Pattern
//! - **Arc**: Multiple ownership (many HTTP handlers can hold a reference)
//! - **RwLock**: Thread-safe read/write access
//! - **T**: The actual data type being protected
//! - **Result**: Thread-safe shared mutable state - the "holy grail" of concurrent programming
//! 
//! ## Why this is hard in other languages:
//! - C++: Easy to get segfaults, data races, or memory leaks
//! - Java: Requires careful synchronization, easy to get deadlocks  
//! - JavaScript: Single-threaded, so not an issue
//! - Rust: Type system prevents most common mistakes at compile time!

use crate::config::AppConfig;        // Our configuration types
use std::sync::{Arc, RwLock};        // Thread-safe shared ownership and locking
use std::time::Instant;              // For tracking server uptime
use std::collections::HashMap;       // For storing per-endpoint metrics

/// The main application state that's shared across all HTTP request handlers.
/// 
/// ## Thread Safety Pattern:
/// This struct uses Arc<RwLock<T>> for all mutable data, which means:
/// - Multiple HTTP requests can read the same data simultaneously
/// - Only one request can modify data at a time
/// - No data races or memory corruption possible
/// 
/// ## Rust Concepts:
/// - **#[derive(Debug, Clone)]**: Automatically implements debug printing and cloning
/// - **pub struct**: Public struct that other modules can use
/// - **Arc<RwLock<T>>**: Thread-safe shared mutable data
/// - **Instant**: A point in time (for measuring duration)
#[derive(Debug, Clone)]
pub struct AppState {
    /// Application configuration (can be updated at runtime)
    /// Arc<RwLock<AppConfig>> means:
    /// - Arc: Multiple HTTP handlers can hold a reference to this
    /// - RwLock: Multiple readers OR one writer (thread-safe)
    /// - AppConfig: The actual configuration data
    pub config: Arc<RwLock<AppConfig>>,
    
    /// Performance metrics (constantly being updated by requests)
    /// This needs to be mutable because every HTTP request updates the metrics
    pub metrics: Arc<RwLock<AppMetrics>>,
    
    /// When the server started (never changes, so no Arc<RwLock> needed)
    /// Instant is Copy, so it's safe to share directly
    pub start_time: Instant,
}

/// Performance metrics collected across all HTTP requests.
/// 
/// ## Rust Concepts:
/// - **#[derive(Debug, Default)]**: Automatically implements:
///   - `Debug`: Can be printed with {:?} for debugging
///   - `Default`: Can create with AppMetrics::default() (all zeros)
/// - **u64**: 64-bit unsigned integer (can count up to 18 quintillion)
/// - **u32**: 32-bit unsigned integer (sufficient for concurrent sessions)
/// - **HashMap**: Key-value map (like a dictionary in Python)
/// 
/// ## Why these metrics matter:
/// - **request_count**: Total requests processed (for load monitoring)
/// - **error_count**: Total errors (for reliability monitoring)  
/// - **active_sessions**: Current concurrent connections (for capacity planning)
/// - **endpoint_metrics**: Per-endpoint statistics (for performance optimization)
#[derive(Debug, Default)]
pub struct AppMetrics {
    /// Total number of HTTP requests processed since server start
    pub request_count: u64,
    
    /// Total number of errors encountered since server start
    pub error_count: u64,
    
    /// Current number of active audio transcription sessions
    pub active_sessions: u32,
    
    /// Detailed metrics for each API endpoint (URL path)
    /// Key: endpoint name (e.g., "GET /health")
    /// Value: detailed metrics for that endpoint
    pub endpoint_metrics: HashMap<String, EndpointMetric>,
}

/// Detailed performance metrics for a specific API endpoint.
/// 
/// ## Rust Concepts:
/// - **Clone**: Can make copies of this struct (needed for sending metrics over HTTP)
/// - **total_duration_ms**: Cumulative time spent processing requests (in milliseconds)
/// 
/// ## Performance calculations:
/// - **Average response time**: total_duration_ms / request_count
/// - **Error rate**: error_count / request_count
/// - **Requests per second**: Can be calculated using server uptime
#[derive(Debug, Default, Clone)]
pub struct EndpointMetric {
    /// Number of requests to this specific endpoint
    pub request_count: u64,
    
    /// Total time spent processing all requests to this endpoint (milliseconds)
    pub total_duration_ms: u64,
    
    /// Number of errors that occurred for this endpoint
    pub error_count: u64,
}

/// Implementation of methods for AppState.
impl AppState {
    /// Create a new AppState with the given configuration.
    /// 
    /// ## What this does:
    /// 1. Wraps the config in Arc<RwLock<>> for thread-safe sharing
    /// 2. Creates empty metrics (also thread-safe)
    /// 3. Records the current time as the server start time
    /// 
    /// ## Rust Concepts:
    /// - **Arc::new()**: Creates a new reference-counted pointer
    /// - **RwLock::new()**: Creates a new reader-writer lock
    /// - **AppMetrics::default()**: Creates metrics with all zeros
    /// - **Instant::now()**: Captures the current moment in time
    /// - **Self**: Refers to the AppState type
    pub fn new(config: AppConfig) -> Self {
        Self {
            // Wrap config for thread-safe sharing and updating
            config: Arc::new(RwLock::new(config)),
            // Start with empty metrics
            metrics: Arc::new(RwLock::new(AppMetrics::default())),
            // Record when the server started
            start_time: Instant::now(),
        }
    }

    /// Get a copy of the current configuration.
    /// 
    /// ## Thread Safety:
    /// - Uses `.read()` to get a read lock (multiple readers allowed)
    /// - `.unwrap()` assumes the lock isn't poisoned (safe in practice)
    /// - `.clone()` makes a copy so we don't hold the lock longer than needed
    /// 
    /// ## Rust Concepts:
    /// - **&self**: Immutable reference to self (this method doesn't change AppState)
    /// - **.read()**: Acquire a read lock (blocks if someone is writing)
    /// - **.unwrap()**: Convert Result to the value (panics on error)
    /// - **.clone()**: Make a copy of the AppConfig
    /// 
    /// ## Why clone:
    /// Cloning releases the lock immediately, so other threads aren't blocked.
    /// AppConfig is designed to be cheap to clone.
    pub fn get_config(&self) -> AppConfig {
        self.config.read().unwrap().clone()
    }

    /// Update the configuration with validation.
    /// 
    /// ## Thread Safety:
    /// - Uses `.write()` to get exclusive write access
    /// - Only one thread can write at a time
    /// - All readers are blocked until the write completes
    /// 
    /// ## Rust Concepts:
    /// - **Result<(), String>**: Returns either success () or an error message
    /// - **match**: Pattern matching on the validation result
    /// - **&self**: Immutable reference (the method doesn't change AppState itself)
    /// - **write()**: Acquire exclusive write lock
    /// - **dereference (*)**:  Directly assigns to the locked value
    /// 
    /// ## Error handling:
    /// Configuration is validated before updating to ensure it's always valid.
    pub fn update_config(&self, new_config: AppConfig) -> Result<(), String> {
        match new_config.validate() {
            Ok(_) => {
                // Validation passed, update the config
                *self.config.write().unwrap() = new_config;
                Ok(())
            }
            Err(e) => {
                // Validation failed, return the error
                Err(e.to_string())
            }
        }
    }

    /// Increment the total request counter (called by middleware for every request).
    /// 
    /// ## Thread Safety:
    /// - Gets exclusive write access to metrics
    /// - Blocks other threads until the increment is complete
    /// - Very fast operation (just adding 1 to a number)
    /// 
    /// ## Rust Concepts:
    /// - **let mut**: Creates a mutable binding to the locked metrics
    /// - **write()**: Gets exclusive write access (no other reads or writes allowed)
    /// - **+=**: Compound assignment (same as metrics.request_count = metrics.request_count + 1)
    pub fn increment_request_count(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.request_count += 1;
    }

    /// Increment the total error counter (called when any request fails).
    /// 
    /// ## When this is called:
    /// - HTTP 4xx errors (client errors like 404 Not Found)
    /// - HTTP 5xx errors (server errors like 500 Internal Server Error)
    /// - Any unhandled exceptions during request processing
    /// 
    /// ## Thread safety:
    /// Same pattern as increment_request_count - fast exclusive write access.
    pub fn increment_error_count(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.error_count += 1;
    }

    /// Record detailed metrics for a specific endpoint.
    /// 
    /// ## Parameters:
    /// - **endpoint**: The API endpoint (e.g., "GET /health", "POST /config")
    /// - **duration_ms**: How long the request took to process (in milliseconds)
    /// - **is_error**: Whether this request resulted in an error
    /// 
    /// ## Rust Concepts:
    /// - **&str**: String slice (borrowed string, no allocation needed)
    /// - **.entry()**: Gets or creates a HashMap entry
    /// - **.or_default()**: Creates default value if the key doesn't exist
    /// - **if**: Conditional execution
    /// 
    /// ## HashMap operations:
    /// The first time we see an endpoint, we create a new EndpointMetric with default values.
    /// Subsequent requests update the existing metrics for that endpoint.
    pub fn record_endpoint_request(&self, endpoint: &str, duration_ms: u64, is_error: bool) {
        let mut metrics = self.metrics.write().unwrap();
        
        // Get or create metrics for this specific endpoint
        let endpoint_metric = metrics.endpoint_metrics.entry(endpoint.to_string()).or_default();
        
        // Update the metrics for this endpoint
        endpoint_metric.request_count += 1;
        endpoint_metric.total_duration_ms += duration_ms;
        
        if is_error {
            endpoint_metric.error_count += 1;
        }
    }

    /// Increment the active sessions counter (called when a new audio session starts).
    /// 
    /// ## When this is used:
    /// - User starts audio transcription
    /// - WebSocket connection established for real-time processing
    /// - Any long-running operation that needs to be tracked
    /// 
    /// ## Session tracking:
    /// Active sessions help monitor current load and prevent overloading the server.
    pub fn increment_active_sessions(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.active_sessions += 1;
    }

    /// Decrement the active sessions counter (called when an audio session ends).
    /// 
    /// ## Safety check:
    /// Includes a check to prevent underflow (going below zero).
    /// This prevents bugs where we might decrement more than we increment.
    /// 
    /// ## Rust Concepts:
    /// - **if condition**: Only execute the decrement if sessions > 0
    /// - **Underflow protection**: u32 would panic on underflow, so we check first
    pub fn decrement_active_sessions(&self) {
        let mut metrics = self.metrics.write().unwrap();
        if metrics.active_sessions > 0 {
            metrics.active_sessions -= 1;
        }
    }

    /// Get a snapshot of current metrics (used for the /metrics endpoint).
    /// 
    /// ## Why a snapshot:
    /// - Takes a read lock to get consistent data
    /// - Clones the data so we don't hold the lock while sending HTTP response
    /// - Ensures metrics don't change while we're serializing them to JSON
    /// 
    /// ## Rust Concepts:
    /// - **.read()**: Read-only access (other readers allowed)
    /// - **Manual struct creation**: Creating AppMetrics field by field
    /// - **.clone()**: Deep copy of the HashMap (expensive but necessary)
    /// 
    /// ## Performance note:
    /// Cloning the endpoint_metrics HashMap can be expensive with many endpoints,
    /// but it's necessary to avoid holding locks during HTTP response generation.
    pub fn get_metrics_snapshot(&self) -> AppMetrics {
        let metrics = self.metrics.read().unwrap();
        AppMetrics {
            request_count: metrics.request_count,
            error_count: metrics.error_count,
            active_sessions: metrics.active_sessions,
            endpoint_metrics: metrics.endpoint_metrics.clone(),
        }
    }

    /// Get server uptime in seconds.
    /// 
    /// ## How it works:
    /// - Calculates the difference between now and start_time
    /// - Converts the duration to seconds
    /// 
    /// ## Rust Concepts:
    /// - **.elapsed()**: Returns a Duration since start_time
    /// - **.as_secs()**: Converts Duration to seconds (u64)
    /// - **No locking needed**: start_time never changes, so it's safe to read directly
    pub fn get_uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

/// Implementation of utility methods for EndpointMetric.
impl EndpointMetric {
    /// Calculate the average response time for this endpoint.
    /// 
    /// ## Formula:
    /// Average = Total Duration ÷ Number of Requests
    /// 
    /// ## Rust Concepts:
    /// - **f64**: 64-bit floating point number (for precise decimals)
    /// - **as f64**: Type conversion from integer to float
    /// - **Division by zero check**: Prevents runtime panic
    /// 
    /// ## Example:
    /// If an endpoint has processed 10 requests taking a total of 500ms,
    /// the average response time would be 50ms per request.
    pub fn average_duration_ms(&self) -> f64 {
        if self.request_count > 0 {
            self.total_duration_ms as f64 / self.request_count as f64
        } else {
            0.0  // No requests yet, so no average to calculate
        }
    }

    /// Calculate the error rate for this endpoint as a percentage (0.0 to 1.0).
    /// 
    /// ## Formula:
    /// Error Rate = Number of Errors ÷ Total Requests
    /// 
    /// ## Return values:
    /// - 0.0 = No errors (0% error rate)
    /// - 0.5 = Half the requests failed (50% error rate)  
    /// - 1.0 = All requests failed (100% error rate)
    /// 
    /// ## Example:
    /// If an endpoint has processed 100 requests and 5 failed,
    /// the error rate would be 0.05 (5%).
    pub fn error_rate(&self) -> f64 {
        if self.request_count > 0 {
            self.error_count as f64 / self.request_count as f64
        } else {
            0.0  // No requests yet, so no errors possible
        }
    }
}