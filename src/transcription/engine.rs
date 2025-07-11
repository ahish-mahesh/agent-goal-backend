//! # Transcription Engine
//! 
//! Core transcription processing engine that coordinates between audio buffers,
//! Whisper models, and result handling. Provides high-level transcription
//! operations with confidence scoring and error handling.
//! 
//! ## Key Responsibilities:
//! - **Audio-to-text conversion**: Process audio segments using loaded models
//! - **Confidence scoring**: Evaluate transcription quality and reliability
//! - **Result formatting**: Structure transcription results for API responses
//! - **Performance monitoring**: Track transcription speed and accuracy

use crate::transcription::model::{WhisperModel, ModelSize};
use crate::audio::buffer::AudioBuffer;
use candle_core::Device;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use anyhow::{Result, anyhow};

/// Configuration for transcription processing.
/// 
/// ## Tuning Parameters:
/// Allows fine-tuning of transcription behavior for different use cases
/// (real-time vs accuracy, language-specific settings, etc.)
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    /// Target language (ISO 639-1 code like "en", "es", "fr")
    pub language: Option<String>,
    
    /// Minimum audio duration to process (seconds)
    pub min_audio_duration: f64,
    
    /// Maximum audio duration per segment (seconds) 
    pub max_audio_duration: f64,
    
    /// Whether to enable automatic punctuation
    pub enable_punctuation: bool,
    
    /// Temperature for model sampling (0.0 = deterministic, 1.0 = creative)
    pub temperature: f32,
    
    /// Whether to timestamp individual words
    pub enable_word_timestamps: bool,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            language: Some("en".to_string()),
            min_audio_duration: 0.5,    // Process segments >= 0.5 seconds
            max_audio_duration: 30.0,   // Whisper works best with <30s segments
            enable_punctuation: true,
            temperature: 0.0,           // Deterministic output for consistency
            enable_word_timestamps: false, // Disabled for simplicity initially
        }
    }
}

/// Result of a transcription operation.
/// 
/// ## API Integration:
/// This struct is designed to match the API contract format
/// for transcription responses.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptionResult {
    /// The transcribed text
    pub text: String,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    
    /// Duration of audio processed (seconds)
    pub audio_duration: f64,
    
    /// Time taken for transcription (milliseconds)
    pub processing_time_ms: u64,
    
    /// Language detected/used
    pub language: String,
    
    /// Model used for transcription
    pub model_name: String,
    
    /// Timestamp when transcription was completed
    pub timestamp: u64,
    
    /// Optional word-level timestamps
    pub word_timestamps: Option<Vec<WordTimestamp>>,
}

/// Timestamp information for individual words.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f32,
}

/// High-level transcription engine that coordinates model loading and audio processing.
/// 
/// ## Thread Safety:
/// Uses Arc<RwLock<>> to allow safe concurrent access to the loaded model.
/// Multiple transcription requests can be processed simultaneously.
/// 
/// ## Resource Management:
/// - Manages model loading/unloading based on usage
/// - Tracks performance metrics across transcriptions
/// - Handles model switching and cleanup
pub struct TranscriptionEngine {
    /// Currently loaded Whisper model
    /// RwLock allows multiple concurrent reads (transcriptions) or one write (model change)
    model: Arc<RwLock<Option<WhisperModel>>>,
    
    /// Configuration for transcription behavior
    config: TranscriptionConfig,
    
    /// Device to use for model inference (CPU/GPU)
    device: Device,
    
    /// Performance metrics tracking
    metrics: Arc<RwLock<EngineMetrics>>,
}

/// Performance tracking for the transcription engine.
#[derive(Debug, Default)]
struct EngineMetrics {
    /// Total number of transcription requests processed
    total_requests: u64,
    
    /// Total audio duration processed (seconds)
    total_audio_duration: f64,
    
    /// Total processing time (milliseconds)
    total_processing_time: u64,
    
    /// Number of failed transcriptions
    failed_requests: u64,
    
    /// Average confidence score
    average_confidence: f32,
}

impl TranscriptionEngine {
    /// Create a new transcription engine.
    /// 
    /// ## Parameters:
    /// - **config**: Transcription behavior configuration
    /// - **device**: Device for model inference (CPU/GPU)
    pub fn new(config: TranscriptionConfig, device: Device) -> Self {
        Self {
            model: Arc::new(RwLock::new(None)),
            config,
            device,
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
        }
    }
    
    /// Load a specific Whisper model for transcription.
    /// 
    /// ## Model Management:
    /// - Unloads any previously loaded model to free memory
    /// - Downloads and loads the new model from HuggingFace
    /// - Validates the model works before making it available
    /// 
    /// ## Parameters:
    /// - **model_size**: Which Whisper model variant to load
    pub async fn load_model(&self, model_size: ModelSize) -> Result<()> {
        tracing::info!("Loading {} model for transcription engine", model_size);
        let start_time = Instant::now();
        
        // print the model size being loaded
        tracing::info!("Loading model size: {}", model_size);

        // Load the new model (validation is done internally)
        let new_model = WhisperModel::load(model_size, self.device.clone()).await?;
        
        // Replace the current model
        {
            let mut model_guard = self.model.write().await;
            *model_guard = Some(new_model);
        }
        
        let load_time = start_time.elapsed();
        tracing::info!("Model loaded and validated in {:.2}s", load_time.as_secs_f64());
        
        Ok(())
    }
    
    /// Unload the current model to free memory.
    pub async fn unload_model(&self) {
        tracing::info!("Unloading transcription model");
        let mut model_guard = self.model.write().await;
        *model_guard = None;
    }
    
    /// Check if a model is currently loaded and ready.
    pub async fn is_model_loaded(&self) -> bool {
        let model_guard = self.model.read().await;
        model_guard.is_some()
    }
    
    /// Get information about the currently loaded model.
    pub async fn get_current_model_info(&self) -> Option<String> {
        let model_guard = self.model.read().await;
        model_guard.as_ref().map(|model| {
            format!("{} ({})", 
                model.info().name, 
                model.info().description
            )
        })
    }
    
    /// Transcribe audio data to text using the loaded model.
    /// 
    /// ## Process:
    /// 1. Validate audio meets minimum requirements
    /// 2. Check that a model is loaded and ready
    /// 3. Process audio through the Whisper model
    /// 4. Calculate confidence score and format results
    /// 5. Update performance metrics
    /// 
    /// ## Parameters:
    /// - **audio_data**: Audio samples as 32-bit floats (16kHz, mono)
    /// 
    /// ## Returns:
    /// - **Ok(TranscriptionResult)**: Successful transcription with metadata
    /// - **Err(anyhow::Error)**: Transcription failed with error details
    pub async fn transcribe(&self, audio_data: &[f32]) -> Result<TranscriptionResult> {
        let start_time = Instant::now();
        
        // Validate input
        if audio_data.is_empty() {
            return Err(anyhow!("Audio data is empty"));
        }
        
        let audio_duration = audio_data.len() as f64 / 16000.0; // Assuming 16kHz sample rate
        
        // Check duration constraints
        if audio_duration < self.config.min_audio_duration {
            return Err(anyhow!(
                "Audio too short: {:.2}s (minimum: {:.2}s)", 
                audio_duration, 
                self.config.min_audio_duration
            ));
        }
        
        if audio_duration > self.config.max_audio_duration {
            return Err(anyhow!(
                "Audio too long: {:.2}s (maximum: {:.2}s)", 
                audio_duration, 
                self.config.max_audio_duration
            ));
        }
        
        // Check if model is loaded and get model info
        let model_loaded = {
            let model_guard = self.model.read().await;
            model_guard.is_some()
        };
        
        if !model_loaded {
            return Err(anyhow!("No model loaded for transcription"));
        }
        
        tracing::debug!(
            "Starting transcription of {:.2}s audio", 
            audio_duration
        );
        
        // Perform transcription using the loaded model
        let transcription_text = {
            let mut model_guard = self.model.write().await;
            match model_guard.as_mut() {
                Some(model) => {
                    model.transcribe(audio_data, self.config.language.as_deref()).await?
                }
                None => {
                    return Err(anyhow!("No model loaded for transcription"));
                }
            }
        };

        // Print the transcription result
        tracing::debug!("Transcription result: '{}'", transcription_text);
        
        // Calculate processing time
        let processing_time = start_time.elapsed();
        let processing_time_ms = processing_time.as_millis() as u64;
        
        // Calculate confidence score (simplified heuristic)
        let confidence = self.estimate_confidence(&transcription_text, audio_duration);
        
        tracing::debug!(
            "Transcription confidence: {:.2} (processing time: {}ms)", 
            confidence, 
            processing_time_ms
        );

        // Get current timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        // Create result
        let result = TranscriptionResult {
            text: transcription_text.trim().to_string(),
            confidence,
            audio_duration,
            processing_time_ms,
            language: self.config.language.clone().unwrap_or("en".to_string()),
            model_name: {
                let model_guard = self.model.read().await;
                model_guard.as_ref()
                    .map(|m| m.info().name.clone())
                    .unwrap_or_else(|| "unknown".to_string())
            },
            timestamp,
            word_timestamps: None, // Not implemented yet
        };
        
        // Update metrics
        self.update_metrics(&result).await;
        
        tracing::info!(
            "Transcription completed: {:.2}s audio -> {} chars in {}ms (confidence: {:.2})", 
            audio_duration,
            result.text.len(),
            processing_time_ms,
            confidence
        );
        
        Ok(result)
    }
    
    /// Transcribe audio from a buffer with automatic segment management.
    /// 
    /// ## Buffer Processing:
    /// Gets the current audio segment from the buffer and transcribes it.
    /// Handles overlap management and segment boundary detection automatically.
    pub async fn transcribe_from_buffer(&self, buffer: &AudioBuffer) -> Result<TranscriptionResult> {
        // Get audio segment from buffer
        let audio_data = buffer.get_transcription_segment();
        
        if audio_data.is_empty() {
            return Err(anyhow!("No audio data available in buffer"));
        }
        
        // Check if we have sufficient audio
        if !buffer.has_sufficient_audio() {
            return Err(anyhow!("Insufficient audio data for transcription"));
        }
        
        // Transcribe the segment
        self.transcribe(&audio_data).await
    }
    
    /// Estimate confidence score for a transcription result.
    /// 
    /// ## Heuristic Approach:
    /// This is a simplified confidence estimation. A more sophisticated
    /// approach would use model attention weights and output probabilities.
    fn estimate_confidence(&self, text: &str, audio_duration: f64) -> f32 {
        // Handle empty strings explicitly - they should always have low confidence
        if text.trim().is_empty() {
            return 0.2; // Well below 0.5 for empty text
        }
        
        // Base confidence starts at 0.5
        let mut confidence = 0.5f32;
        
        // Factor 1: Text length vs audio duration ratio
        let chars_per_second = text.len() as f64 / audio_duration;
        if chars_per_second > 10.0 && chars_per_second < 100.0 {
            confidence += 0.2; // Reasonable speaking pace
        } else {
            confidence -= 0.1; // Too fast, too slow, or empty
        }
        
        // Factor 2: Presence of common words
        let common_words = ["the", "and", "to", "of", "a", "in", "is", "it", "you", "that"];
        let text_lower = text.to_lowercase();
        let common_word_count = common_words.iter()
            .filter(|&&word| text_lower.contains(word))
            .count();
        
        if common_word_count > 0 {
            confidence += 0.1 * (common_word_count as f32 / common_words.len() as f32);
        }
        
        // Factor 3: Proper capitalization and punctuation
        if text.chars().next().map_or(false, |c| c.is_uppercase()) {
            confidence += 0.1;
        }
        
        // Only award punctuation bonus for single punctuation marks, not repeated ones
        if (text.contains('.') && !text.contains("..")) || 
           (text.contains('?') && !text.contains("??")) || 
           (text.contains('!') && !text.contains("!!")) {
            confidence += 0.1;
        }
        
        // Factor 4: Check for obvious errors and poor quality indicators
        if text.contains("...") || text.contains("???") || text.contains("asd") {
            confidence -= 0.2; // Heavy penalty for obvious errors/garbled text
        } else {
            confidence += 0.1; // Bonus for clean text
        }
        
        // Clamp to valid range
        confidence.clamp(0.0, 1.0)
    }
    
    /// Update performance metrics with a new transcription result.
    async fn update_metrics(&self, result: &TranscriptionResult) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_requests += 1;
        metrics.total_audio_duration += result.audio_duration;
        metrics.total_processing_time += result.processing_time_ms;
        
        // Update rolling average confidence
        let total_requests = metrics.total_requests as f32;
        metrics.average_confidence = 
            (metrics.average_confidence * (total_requests - 1.0) + result.confidence) / total_requests;
    }
    
    /// Get performance statistics for the transcription engine.
    pub async fn get_performance_stats(&self) -> EnginePerformanceStats {
        let metrics = self.metrics.read().await;
        
        EnginePerformanceStats {
            total_requests: metrics.total_requests,
            total_audio_hours: metrics.total_audio_duration / 3600.0,
            average_processing_time_ms: if metrics.total_requests > 0 {
                metrics.total_processing_time / metrics.total_requests
            } else {
                0
            },
            success_rate: if metrics.total_requests > 0 {
                (metrics.total_requests - metrics.failed_requests) as f32 / metrics.total_requests as f32
            } else {
                0.0
            },
            average_confidence: metrics.average_confidence,
            throughput_factor: if metrics.total_processing_time > 0 {
                (metrics.total_audio_duration * 1000.0) / metrics.total_processing_time as f64
            } else {
                0.0
            },
        }
    }
}

/// Performance statistics for the transcription engine.
#[derive(Debug)]
pub struct EnginePerformanceStats {
    pub total_requests: u64,
    pub total_audio_hours: f64,
    pub average_processing_time_ms: u64,
    pub success_rate: f32,
    pub average_confidence: f32,
    /// Ratio of audio time to processing time (>1.0 = real-time capable)
    pub throughput_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_transcription_config_default() {
        let config = TranscriptionConfig::default();
        assert_eq!(config.language, Some("en".to_string()));
        assert_eq!(config.min_audio_duration, 0.5);
        assert_eq!(config.temperature, 0.0);
    }
    
    #[test]
    fn test_confidence_estimation() {
        let engine = TranscriptionEngine::new(
            TranscriptionConfig::default(),
            Device::Cpu
        );
        
        // Test reasonable transcription
        let confidence1 = engine.estimate_confidence("Hello, how are you today?", 2.0);
        assert!(confidence1 > 0.5);
        
        // Test empty transcription
        let confidence2 = engine.estimate_confidence("", 2.0);
        assert!(confidence2 < 0.5);
        
        // Test garbled transcription
        let confidence3 = engine.estimate_confidence("asdkjhasd... ???", 2.0);
        assert!(confidence3 < 0.5);
    }
}