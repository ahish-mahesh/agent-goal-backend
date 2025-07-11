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
    pub audio: AudioConfig,
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
/// - `max_model_memory_mb`: Maximum memory to allocate for loaded models (in MB)
/// 
/// ## Tuning guidelines:
/// - Higher concurrent sessions: More users, but requires more CPU/memory
/// - Higher model memory limit: Can load larger/more models, but uses more RAM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_concurrent_sessions: usize,  // usize = platform-specific unsigned integer (usually 64-bit)
    pub max_model_memory_mb: u32,        // Memory limit for AI models in megabytes
}

/// Audio processing configuration settings.
/// 
/// ## Fields:
/// - `sample_rate`: Audio sample rate in Hz (16000 for Whisper compatibility)
/// - `channels`: Number of audio channels (1 for mono, 2 for stereo)
/// - `bit_depth`: Audio bit depth (16 for 16-bit PCM)
/// - `buffer_duration_ms`: Audio buffer duration in milliseconds
/// - `overlap_duration_ms`: Overlap between audio segments in milliseconds
/// - `min_transcription_duration`: Minimum audio duration before transcription (seconds)
/// - `max_transcription_duration`: Maximum audio duration per transcription (seconds)
/// 
/// ## Whisper Requirements:
/// Whisper models work best with 16kHz, 16-bit, mono PCM audio.
/// The buffer and overlap settings control real-time transcription behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,                    // Audio sample rate (Hz)
    pub channels: u8,                        // Number of audio channels
    pub bit_depth: u8,                       // Audio bit depth
    pub buffer_duration_ms: u32,             // Buffer duration (milliseconds)
    pub overlap_duration_ms: u32,            // Overlap duration (milliseconds)
    pub min_transcription_duration: f64,     // Minimum audio duration for transcription (seconds)
    pub max_transcription_duration: f64,     // Maximum audio duration per transcription (seconds)
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
                max_model_memory_mb: 2048,     // 2GB memory limit for models
            },
            audio: AudioConfig {
                sample_rate: 16000,            // 16kHz (Whisper's preferred sample rate)
                channels: 1,                   // Mono audio
                bit_depth: 16,                 // 16-bit PCM
                buffer_duration_ms: 2000,      // 2 second audio buffer
                overlap_duration_ms: 500,      // 500ms overlap between segments
                min_transcription_duration: 0.5,  // Process audio >= 0.5 seconds
                max_transcription_duration: 30.0, // Max 30 seconds per segment
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

        if self.performance.max_model_memory_mb == 0 {
            return Err(anyhow::anyhow!("Max model memory must be greater than 0"));
        }

        // Validate audio configuration
        if self.audio.sample_rate == 0 {
            return Err(anyhow::anyhow!("Audio sample rate must be greater than 0"));
        }

        if self.audio.channels == 0 {
            return Err(anyhow::anyhow!("Audio channels must be greater than 0"));
        }

        if self.audio.bit_depth == 0 {
            return Err(anyhow::anyhow!("Audio bit depth must be greater than 0"));
        }

        if self.audio.buffer_duration_ms == 0 {
            return Err(anyhow::anyhow!("Audio buffer duration must be greater than 0"));
        }

        if self.audio.min_transcription_duration <= 0.0 {
            return Err(anyhow::anyhow!("Minimum transcription duration must be greater than 0"));
        }

        if self.audio.max_transcription_duration <= self.audio.min_transcription_duration {
            return Err(anyhow::anyhow!("Maximum transcription duration must be greater than minimum"));
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
            if let Some(memory) = performance.get("max_model_memory_mb").and_then(|v| v.as_u64()) {
                self.performance.max_model_memory_mb = memory as u32;
            }
        }

        // Update audio configuration if provided
        if let Some(audio) = partial_config.get("audio") {
            if let Some(sample_rate) = audio.get("sample_rate").and_then(|v| v.as_u64()) {
                self.audio.sample_rate = sample_rate as u32;
            }
            if let Some(channels) = audio.get("channels").and_then(|v| v.as_u64()) {
                self.audio.channels = channels as u8;
            }
            if let Some(bit_depth) = audio.get("bit_depth").and_then(|v| v.as_u64()) {
                self.audio.bit_depth = bit_depth as u8;
            }
            if let Some(buffer_duration) = audio.get("buffer_duration_ms").and_then(|v| v.as_u64()) {
                self.audio.buffer_duration_ms = buffer_duration as u32;
            }
            if let Some(overlap_duration) = audio.get("overlap_duration_ms").and_then(|v| v.as_u64()) {
                self.audio.overlap_duration_ms = overlap_duration as u32;
            }
            if let Some(min_duration) = audio.get("min_transcription_duration").and_then(|v| v.as_f64()) {
                self.audio.min_transcription_duration = min_duration;
            }
            if let Some(max_duration) = audio.get("max_transcription_duration").and_then(|v| v.as_f64()) {
                self.audio.max_transcription_duration = max_duration;
            }
        }

        // Validate the updated configuration to ensure it's still valid
        self.validate()?;
        Ok(())
    }
}

impl AudioConfig {
    /// Convert to AudioBufferConfig for use with audio processing modules.
    pub fn to_buffer_config(&self) -> crate::audio::buffer::AudioBufferConfig {
        crate::audio::buffer::AudioBufferConfig {
            sample_rate: self.sample_rate,
            channels: self.channels,
            bit_depth: self.bit_depth,
            buffer_duration_ms: self.buffer_duration_ms,
            overlap_duration_ms: self.overlap_duration_ms,
        }
    }

    /// Convert to TranscriptionConfig for use with transcription engine.
    pub fn to_transcription_config(&self) -> crate::transcription::engine::TranscriptionConfig {
        crate::transcription::engine::TranscriptionConfig {
            language: Some("en".to_string()), // Default to English
            min_audio_duration: self.min_transcription_duration,
            max_audio_duration: self.max_transcription_duration,
            enable_punctuation: true,
            temperature: 0.0,
            enable_word_timestamps: false,
        }
    }

    /// Calculate the buffer size in samples.
    pub fn buffer_size_samples(&self) -> usize {
        (self.buffer_duration_ms as usize * self.sample_rate as usize) / 1000
    }

    /// Calculate the overlap size in samples.
    pub fn overlap_size_samples(&self) -> usize {
        (self.overlap_duration_ms as usize * self.sample_rate as usize) / 1000
    }

    /// Calculate bytes per second for this audio configuration.
    pub fn bytes_per_second(&self) -> usize {
        self.sample_rate as usize * self.channels as usize * (self.bit_depth as usize / 8)
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
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.audio.channels, 1);
        assert_eq!(config.audio.bit_depth, 16);
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

    /// Test audio configuration utility methods.
    #[test]
    fn test_audio_config_utilities() {
        let audio_config = AudioConfig {
            sample_rate: 16000,
            channels: 1,
            bit_depth: 16,
            buffer_duration_ms: 2000,
            overlap_duration_ms: 500,
            min_transcription_duration: 0.5,
            max_transcription_duration: 30.0,
        };

        // Test buffer size calculation: 2000ms * 16000Hz / 1000 = 32000 samples
        assert_eq!(audio_config.buffer_size_samples(), 32000);
        
        // Test overlap size calculation: 500ms * 16000Hz / 1000 = 8000 samples
        assert_eq!(audio_config.overlap_size_samples(), 8000);
        
        // Test bytes per second: 16000Hz * 1 channel * 2 bytes (16-bit) = 32000 bytes/sec
        assert_eq!(audio_config.bytes_per_second(), 32000);
    }
}