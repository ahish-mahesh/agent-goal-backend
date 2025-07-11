//! # Configuration Management
//! 
//! This module handles loading and managing application configuration from multiple sources:
//! - TOML configuration files (config.toml)
//! - Environment variables (with APP_ prefix)
//! - Default values (built into the code)
//! 
//! ## Key Rust Concepts Used:
//! - **Serde**: Serialization/deserialization library for converting between Rust structs and data formats
//! - **derive macros**: Automatically generate code for common traits (Debug, Clone, Serialize, Deserialize)
//! - **struct**: Custom data types that group related fields together
//! - **impl blocks**: Add methods to structs
//! - **Result<T, E>**: Error handling that forces you to handle potential failures
//! 
//! ## Configuration Priority (highest to lowest):
//! 1. Environment variables (APP_SERVER_HOST, APP_SERVER_PORT, etc.)
//! 2. Configuration file (config.toml)
//! 3. Default values (defined in the Default impl)

use anyhow::Result;              // Better error handling with context
use serde::{Deserialize, Serialize};  // For converting to/from TOML, JSON, etc.
use std::env;                    // For reading environment variables

/// Main application configuration that contains all settings.
/// 
/// ## Rust Concepts:
/// - **#[derive(...)]**: Automatically implements common traits:
///   - `Debug`: Allows printing with {:?} for debugging
///   - `Clone`: Allows making copies of the struct
///   - `Serialize`: Can convert this struct to JSON, TOML, etc.
///   - `Deserialize`: Can create this struct from JSON, TOML, etc.
/// - **pub struct**: Public struct that other modules can use
/// - **pub fields**: Public fields that can be accessed directly
/// 
/// ## Why separate config structs:
/// Breaking configuration into logical groups (server, models, performance)
/// makes it easier to understand and maintain as the application grows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub models: ModelsConfig,
    pub performance: PerformanceConfig,
}

/// Server-specific configuration settings.
/// 
/// ## Fields:
/// - `host`: IP address or hostname to bind the server to (e.g., "127.0.0.1", "0.0.0.0")
/// - `port`: TCP port number to listen on (1-65535, typically 8080 for development)
/// 
/// ## Common values:
/// - `host = "127.0.0.1"`: Only accept connections from localhost (development)
/// - `host = "0.0.0.0"`: Accept connections from any IP address (production)
/// - `port = 8080`: Common development port (production often uses 80 or 443)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,  // u16 = unsigned 16-bit integer (0-65535), perfect for port numbers
}

/// AI model configuration settings.
/// 
/// ## Fields:
/// - `whisper_model`: Which Whisper model to use for speech recognition ("tiny", "base", "small", "medium", "large")
/// - `llm_model`: Which language model to use for text processing (e.g., "llama3.1:8b", "phi-3:medium")
/// 
/// ## Model size trade-offs:
/// - Smaller models: Faster processing, less memory, lower accuracy
/// - Larger models: Slower processing, more memory, higher accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub whisper_model: String,
    pub llm_model: String,
}

/// Performance tuning configuration.
/// 
/// ## Fields:
/// - `max_concurrent_sessions`: Maximum number of audio transcription sessions to handle simultaneously
/// - `audio_buffer_size`: Size of audio buffers in bytes (affects latency vs. reliability)
/// 
/// ## Tuning guidelines:
/// - Higher concurrent sessions: More users, but requires more CPU/memory
/// - Larger audio buffers: More reliable processing, but higher latency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_concurrent_sessions: usize,  // usize = platform-specific unsigned integer (usually 64-bit)
    pub audio_buffer_size: usize,
}

/// Provides default configuration values.
/// 
/// ## Rust Concepts:
/// - **impl Default**: Implements the Default trait, which provides a `default()` method
/// - **Self**: Refers to the current type (AppConfig)
/// - **to_string()**: Converts string literals (&str) to owned String objects
/// 
/// ## Why defaults matter:
/// Default values ensure the application can start even if no configuration file exists.
/// They also serve as documentation of reasonable starting values.
impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),  // Localhost only (safe for development)
                port: 8080,                     // Common development port
            },
            models: ModelsConfig {
                whisper_model: "medium".to_string(),     // Good balance of accuracy and speed
                llm_model: "llama3.1:8b".to_string(),    // Recommended LLM model
            },
            performance: PerformanceConfig {
                max_concurrent_sessions: 10,   // Reasonable for most development machines
                audio_buffer_size: 4096,       // 4KB buffer (good starting point)
            },
        }
    }
}

/// Implementation block for AppConfig - adds methods to the struct.
impl AppConfig {
    /// Load configuration from multiple sources in priority order.
    /// 
    /// ## Configuration Loading Process:
    /// 1. Start with built-in defaults
    /// 2. Override with values from config.toml (if it exists)
    /// 3. Override with environment variables prefixed with APP_
    /// 4. Handle special cases for HOST and PORT environment variables
    /// 
    /// ## Rust Concepts:
    /// - **Builder pattern**: Chain method calls to configure the config loader
    /// - **?**: Early return on error (if any step fails, return the error)
    /// - **env::var()**: Read environment variables, returns Result<String, VarError>
    /// - **if let Ok(...)**: Only execute if the environment variable exists
    /// 
    /// ## Environment Variable Examples:
    /// - `APP_SERVER_HOST=0.0.0.0`: Override server host
    /// - `APP_SERVER_PORT=3000`: Override server port  
    /// - `APP_MODELS_WHISPER_MODEL=large`: Override whisper model
    /// - `HOST=0.0.0.0`: Special case for deployment platforms
    /// - `PORT=3000`: Special case for deployment platforms
    pub fn load() -> Result<Self> {
        let mut settings = config::Config::builder()
            // 1. Start with defaults - converts our Default impl to config format
            .add_source(config::Config::try_from(&AppConfig::default())?)
            // 2. Load from config.toml file (if it exists) - required(false) means "don't error if missing"
            .add_source(config::File::with_name("config").required(false))
            // 3. Load from environment variables with APP_ prefix
            // Example: APP_SERVER_HOST becomes server.host in the config
            .add_source(config::Environment::with_prefix("APP").separator("_"));

        // Handle special environment variables used by deployment platforms
        // These don't follow the APP_ prefix convention but are commonly used
        if let Ok(host) = env::var("HOST") {
            settings = settings.set_override("server.host", host)?;
        }
        
        if let Ok(port) = env::var("PORT") {
            settings = settings.set_override("server.port", port)?;
        }

        // Build the final configuration and convert it back to our AppConfig struct
        let config = settings.build()?.try_deserialize()?;
        Ok(config)
    }

    /// Validate that the configuration values make sense.
    /// 
    /// ## What this checks:
    /// - Server port is not 0 (port 0 is reserved and can't be used)
    /// - Max concurrent sessions is greater than 0 (must allow at least one session)
    /// - Audio buffer size is greater than 0 (must have some buffer space)
    /// 
    /// ## Rust Concepts:
    /// - **&self**: Borrowed reference (read-only access to the struct)
    /// - **anyhow::anyhow!**: Creates an error with a custom message
    /// - **Early return**: Return immediately if validation fails
    /// 
    /// ## Why validate:
    /// Catching configuration errors early prevents runtime failures and
    /// provides clear error messages about what's wrong.
    pub fn validate(&self) -> Result<()> {
        if self.server.port == 0 {
            return Err(anyhow::anyhow!("Server port cannot be 0"));
        }

        if self.performance.max_concurrent_sessions == 0 {
            return Err(anyhow::anyhow!("Max concurrent sessions must be greater than 0"));
        }

        if self.performance.audio_buffer_size == 0 {
            return Err(anyhow::anyhow!("Audio buffer size must be greater than 0"));
        }

        Ok(())  // All validation passed
    }

    /// Update configuration from a JSON string (used for runtime config updates).
    /// 
    /// ## What this does:
    /// 1. Parse the JSON string into a generic value
    /// 2. Extract individual configuration fields if they exist
    /// 3. Update only the fields that were provided
    /// 4. Validate the updated configuration
    /// 
    /// ## Rust Concepts:
    /// - **&mut self**: Mutable reference (allows modifying the struct)
    /// - **serde_json::Value**: Generic JSON value that can hold any JSON data
    /// - **if let Some(...)**: Only execute if the field exists in the JSON
    /// - **and_then()**: Chain operations that might fail
    /// - **as_str(), as_u64()**: Convert JSON values to specific types
    /// 
    /// ## Partial updates:
    /// This method allows updating only some fields, not the entire configuration.
    /// For example, you can send just `{"server": {"port": 9000}}` to change only the port.
    pub fn update_from_json(&mut self, json_str: &str) -> Result<()> {
        // Parse the JSON string into a generic value
        let partial_config: serde_json::Value = serde_json::from_str(json_str)?;
        
        // Update server configuration if provided
        if let Some(server) = partial_config.get("server") {
            if let Some(host) = server.get("host").and_then(|v| v.as_str()) {
                self.server.host = host.to_string();
            }
            if let Some(port) = server.get("port").and_then(|v| v.as_u64()) {
                self.server.port = port as u16;  // Convert u64 to u16 for port number
            }
        }

        // Update model configuration if provided
        if let Some(models) = partial_config.get("models") {
            if let Some(whisper) = models.get("whisper_model").and_then(|v| v.as_str()) {
                self.models.whisper_model = whisper.to_string();
            }
            if let Some(llm) = models.get("llm_model").and_then(|v| v.as_str()) {
                self.models.llm_model = llm.to_string();
            }
        }

        // Update performance configuration if provided
        if let Some(performance) = partial_config.get("performance") {
            if let Some(sessions) = performance.get("max_concurrent_sessions").and_then(|v| v.as_u64()) {
                self.performance.max_concurrent_sessions = sessions as usize;
            }
            if let Some(buffer) = performance.get("audio_buffer_size").and_then(|v| v.as_u64()) {
                self.performance.audio_buffer_size = buffer as usize;
            }
        }

        // Validate the updated configuration to ensure it's still valid
        self.validate()?;
        Ok(())
    }
}

/// Tests for the configuration module.
/// 
/// ## Rust Concepts:
/// - **#[cfg(test)]**: Only compile this code when running tests
/// - **mod tests**: A module containing test functions
/// - **#[test]**: Marks a function as a test case
/// - **assert_eq!**: Checks that two values are equal
/// - **assert!**: Checks that a condition is true
/// - **is_ok(), is_err()**: Check if a Result is success or error
/// 
/// ## Testing philosophy:
/// Tests ensure that the configuration system works correctly and
/// catches errors before they reach production.
#[cfg(test)]
mod tests {
    use super::*;  // Import everything from the parent module

    /// Test that the default configuration is valid and has expected values.
    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8080);
        // Ensure the default config passes validation
        assert!(config.validate().is_ok());
    }

    /// Test that validation catches invalid configurations.
    #[test]
    fn test_config_validation() {
        let mut config = AppConfig::default();
        config.server.port = 0;  // Invalid port
        // Validation should fail for port 0
        assert!(config.validate().is_err());
    }

    /// Test that runtime configuration updates work correctly.
    #[test]
    fn test_config_update() {
        let mut config = AppConfig::default();
        let json = r#"{"server": {"port": 9090}}"#;  // Update only the port
        assert!(config.update_from_json(json).is_ok());
        assert_eq!(config.server.port, 9090);  // Port should be updated
        // Other fields should remain unchanged
        assert_eq!(config.server.host, "127.0.0.1");
    }
}