//! # Model Registry
//! 
//! Tracks available Whisper models, their loading status, and provides
//! centralized model management across the application. Handles model
//! discovery, status tracking, and resource allocation.
//! 
//! ## Key Features:
//! - **Model availability tracking**: Monitor which models are installed/cached
//! - **Loading status management**: Track download and loading progress
//! - **Resource usage monitoring**: Memory usage and performance metrics
//! - **Model metadata storage**: Size, capabilities, and performance characteristics

use crate::transcription::model::{ModelSize, ModelInfo};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// Current status of a model in the registry.
/// 
/// ## State Transitions:
/// Available → Downloading → Loading → Loaded → Unloaded
/// Any state can transition to Error if something goes wrong.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model is available for download but not cached locally
    Available,
    
    /// Model is currently being downloaded from HuggingFace
    Downloading {
        /// Download progress as percentage (0-100)
        progress: u8,
        /// Estimated time remaining in seconds
        eta_seconds: Option<u32>,
    },
    
    /// Model files downloaded, currently loading into memory
    Loading,
    
    /// Model successfully loaded and ready for inference
    Loaded {
        /// When the model was loaded
        loaded_at: u64,
        /// Estimated memory usage in bytes
        memory_usage_bytes: usize,
    },
    
    /// Model was loaded but has been unloaded to free memory
    Unloaded {
        /// When the model was unloaded
        unloaded_at: u64,
    },
    
    /// Model is in an error state
    Error {
        /// Error message describing what went wrong
        message: String,
        /// When the error occurred
        error_at: u64,
    },
}

impl ModelStatus {
    /// Check if the model is currently available for inference.
    pub fn is_loaded(&self) -> bool {
        matches!(self, ModelStatus::Loaded { .. })
    }
    
    /// Check if the model is currently being prepared (downloading or loading).
    pub fn is_preparing(&self) -> bool {
        matches!(self, ModelStatus::Downloading { .. } | ModelStatus::Loading)
    }
    
    /// Check if the model can be loaded (downloaded but not in memory).
    pub fn can_be_loaded(&self) -> bool {
        matches!(self, ModelStatus::Available | ModelStatus::Unloaded { .. })
    }
    
    /// Get a human-readable status description.
    pub fn description(&self) -> String {
        match self {
            ModelStatus::Available => "Available for download".to_string(),
            ModelStatus::Downloading { progress, .. } => {
                format!("Downloading {}%", progress)
            }
            ModelStatus::Loading => "Loading into memory".to_string(),
            ModelStatus::Loaded { .. } => "Ready for inference".to_string(),
            ModelStatus::Unloaded { .. } => "Downloaded but not loaded".to_string(),
            ModelStatus::Error { message, .. } => {
                format!("Error: {}", message)
            }
        }
    }
}

/// Registry entry for a Whisper model.
/// 
/// ## Purpose:
/// Combines model metadata with current status and performance tracking.
/// This allows the application to make intelligent decisions about which
/// models to load based on requirements and available resources.
#[derive(Debug, Clone)]
pub struct ModelRegistryEntry {
    /// Basic model information (size, description, etc.)
    pub info: ModelInfo,
    
    /// Current status of this model
    pub status: ModelStatus,
    
    /// Performance metrics for this model
    pub metrics: ModelMetrics,
    
    /// When this entry was last updated
    pub last_updated: u64,
}

/// Performance metrics for a model.
/// 
/// ## Usage:
/// Tracks how well a model performs in practice, which helps
/// with automatic model selection and optimization decisions.
#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    /// Total number of transcription requests processed
    pub total_requests: u64,
    
    /// Total audio duration processed (seconds)
    pub total_audio_duration: f64,
    
    /// Total processing time (milliseconds)
    pub total_processing_time: u64,
    
    /// Average confidence score across all transcriptions
    pub average_confidence: f32,
    
    /// Number of failed transcription attempts
    pub failed_requests: u64,
    
    /// Average memory usage when loaded (bytes)
    pub average_memory_usage: usize,
}

impl ModelMetrics {
    /// Calculate the success rate for this model.
    pub fn success_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.total_requests - self.failed_requests) as f32 / self.total_requests as f32
        }
    }
    
    /// Calculate the average processing speed (audio seconds per CPU second).
    pub fn processing_speed_factor(&self) -> f64 {
        if self.total_processing_time == 0 {
            0.0
        } else {
            (self.total_audio_duration * 1000.0) / self.total_processing_time as f64
        }
    }
    
    /// Update metrics with a new transcription result.
    pub fn update(&mut self, audio_duration: f64, processing_time_ms: u64, confidence: f32, success: bool) {
        self.total_requests += 1;
        self.total_audio_duration += audio_duration;
        self.total_processing_time += processing_time_ms;
        
        if success {
            // Update rolling average confidence
            let total_successful = self.total_requests - self.failed_requests;
            self.average_confidence = 
                (self.average_confidence * (total_successful - 1) as f32 + confidence) / total_successful as f32;
        } else {
            self.failed_requests += 1;
        }
    }
}

/// Central registry for managing Whisper model availability and status.
/// 
/// ## Thread Safety:
/// Uses RwLock to allow multiple concurrent readers (status checks) or
/// one writer (status updates) at a time.
/// 
/// ## Responsibilities:
/// - Track which models are available, downloading, loaded, etc.
/// - Provide model recommendation based on requirements
/// - Handle model lifecycle events (load, unload, error)
/// - Maintain performance statistics for each model
pub struct ModelRegistry {
    /// Registry entries indexed by model size
    models: Arc<RwLock<HashMap<ModelSize, ModelRegistryEntry>>>,
    
    /// Currently loaded model (if any)
    current_loaded: Arc<RwLock<Option<ModelSize>>>,
    
    /// Maximum number of models that can be loaded simultaneously
    max_concurrent_models: usize,
    
    /// Total memory limit for loaded models (bytes)
    memory_limit_bytes: usize,
}

impl ModelRegistry {
    /// Create a new model registry.
    /// 
    /// ## Parameters:
    /// - **max_concurrent_models**: Usually 1 to avoid memory issues
    /// - **memory_limit_bytes**: Total memory budget for models
    pub fn new(max_concurrent_models: usize, memory_limit_bytes: usize) -> Self {
        let mut models = HashMap::new();
        
        // Initialize registry with all available model sizes
        for &size in &[ModelSize::Tiny, ModelSize::Base, ModelSize::Small, ModelSize::Medium, ModelSize::Large] {
            let entry = ModelRegistryEntry {
                info: ModelInfo::new(size),
                status: ModelStatus::Available,
                metrics: ModelMetrics::default(),
                last_updated: current_timestamp(),
            };
            models.insert(size, entry);
        }
        
        Self {
            models: Arc::new(RwLock::new(models)),
            current_loaded: Arc::new(RwLock::new(None)),
            max_concurrent_models,
            memory_limit_bytes,
        }
    }
    
    /// Get information about all models in the registry.
    pub async fn list_models(&self) -> Vec<ModelRegistryEntry> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }
    
    /// Get information about a specific model.
    pub async fn get_model(&self, size: ModelSize) -> Option<ModelRegistryEntry> {
        let models = self.models.read().await;
        models.get(&size).cloned()
    }
    
    /// Get the currently loaded model.
    pub async fn get_current_loaded(&self) -> Option<ModelSize> {
        let current = self.current_loaded.read().await;
        current.clone()
    }
    
    /// Update the status of a model.
    /// 
    /// ## Usage:
    /// Called by the transcription engine when model states change
    /// (downloading, loading, loaded, error, etc.)
    pub async fn update_model_status(&self, size: ModelSize, status: ModelStatus) {
        let mut models = self.models.write().await;
        if let Some(entry) = models.get_mut(&size) {
            entry.status = status.clone();
            entry.last_updated = current_timestamp();
            
            // Update currently loaded tracking
            match status {
                ModelStatus::Loaded { .. } => {
                    let mut current = self.current_loaded.write().await;
                    *current = Some(size);
                }
                ModelStatus::Unloaded { .. } | ModelStatus::Error { .. } => {
                    let mut current = self.current_loaded.write().await;
                    if *current == Some(size) {
                        *current = None;
                    }
                }
                _ => {} // No change to loaded status
            }
        }
    }
    
    /// Update performance metrics for a model.
    pub async fn update_model_metrics(
        &self, 
        size: ModelSize, 
        audio_duration: f64, 
        processing_time_ms: u64, 
        confidence: f32, 
        success: bool
    ) {
        let mut models = self.models.write().await;
        if let Some(entry) = models.get_mut(&size) {
            entry.metrics.update(audio_duration, processing_time_ms, confidence, success);
            entry.last_updated = current_timestamp();
        }
    }
    
    /// Recommend the best model for given requirements.
    /// 
    /// ## Selection Criteria:
    /// 1. **Loaded models**: Prefer already loaded models to avoid loading delay
    /// 2. **Performance requirements**: Balance speed vs accuracy based on use case
    /// 3. **Memory constraints**: Ensure model fits within memory limits
    /// 4. **Historical performance**: Consider past success rates and speeds
    /// 
    /// ## Parameters:
    /// - **prefer_speed**: Prioritize fast processing over accuracy
    /// - **min_confidence**: Minimum acceptable confidence score
    /// - **max_memory_mb**: Maximum memory usage allowed
    pub async fn recommend_model(
        &self, 
        prefer_speed: bool, 
        min_confidence: f32,
        max_memory_mb: Option<u32>
    ) -> Option<ModelSize> {
        let models = self.models.read().await;
        let current_loaded = self.current_loaded.read().await;
        
        // If we have a loaded model that meets requirements, prefer it
        if let Some(loaded_size) = *current_loaded {
            if let Some(entry) = models.get(&loaded_size) {
                if entry.status.is_loaded() && 
                   entry.metrics.average_confidence >= min_confidence {
                    return Some(loaded_size);
                }
            }
        }
        
        // Find best available model based on criteria
        let mut candidates: Vec<_> = models.iter()
            .filter(|(_, entry)| {
                // Filter by confidence requirement
                entry.metrics.average_confidence >= min_confidence || entry.metrics.total_requests == 0
            })
            .filter(|(_, entry)| {
                // Filter by memory limit
                if let Some(max_mb) = max_memory_mb {
                    entry.info.size_mb <= max_mb
                } else {
                    true
                }
            })
            .collect();
        
        if candidates.is_empty() {
            return None;
        }
        
        // Sort by preference criteria
        candidates.sort_by(|a, b| {
            if prefer_speed {
                // For speed: prefer smaller models with good performance
                let speed_a = a.1.metrics.processing_speed_factor();
                let speed_b = b.1.metrics.processing_speed_factor();
                speed_b.partial_cmp(&speed_a).unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.info.size_mb.cmp(&b.1.info.size_mb))
            } else {
                // For accuracy: prefer larger models with high confidence
                let conf_a = a.1.metrics.average_confidence;
                let conf_b = b.1.metrics.average_confidence;
                conf_b.partial_cmp(&conf_a).unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.1.info.size_mb.cmp(&a.1.info.size_mb))
            }
        });
        
        candidates.first().map(|(size, _)| **size)
    }
    
    /// Check if we can load a new model given current memory usage.
    pub async fn can_load_model(&self, size: ModelSize) -> bool {
        let models = self.models.read().await;
        
        // Check if we're at the concurrent model limit
        let loaded_count = models.values()
            .filter(|entry| entry.status.is_loaded())
            .count();
            
        if loaded_count >= self.max_concurrent_models {
            return false;
        }
        
        // Check memory limit
        let current_memory_usage: usize = models.values()
            .filter_map(|entry| {
                if let ModelStatus::Loaded { memory_usage_bytes, .. } = entry.status {
                    Some(memory_usage_bytes)
                } else {
                    None
                }
            })
            .sum();
            
        let target_model_size = models.get(&size)
            .map(|entry| entry.info.size_mb as usize * 1024 * 1024)
            .unwrap_or(0);
            
        current_memory_usage + target_model_size <= self.memory_limit_bytes
    }
    
    /// Get a summary of registry status.
    pub async fn get_registry_summary(&self) -> RegistrySummary {
        let models = self.models.read().await;
        let current_loaded = self.current_loaded.read().await;
        
        let total_models = models.len();
        let loaded_models = models.values()
            .filter(|entry| entry.status.is_loaded())
            .count();
        let available_models = models.values()
            .filter(|entry| entry.status.can_be_loaded())
            .count();
        let total_memory_usage = models.values()
            .filter_map(|entry| {
                if let ModelStatus::Loaded { memory_usage_bytes, .. } = entry.status {
                    Some(memory_usage_bytes)
                } else {
                    None
                }
            })
            .sum();
            
        RegistrySummary {
            total_models,
            loaded_models,
            available_models,
            current_loaded: current_loaded.clone(),
            total_memory_usage_bytes: total_memory_usage,
            memory_limit_bytes: self.memory_limit_bytes,
            max_concurrent_models: self.max_concurrent_models,
        }
    }
}

/// Summary of the model registry state.
#[derive(Debug, Serialize)]
pub struct RegistrySummary {
    pub total_models: usize,
    pub loaded_models: usize,
    pub available_models: usize,
    pub current_loaded: Option<ModelSize>,
    pub total_memory_usage_bytes: usize,
    pub memory_limit_bytes: usize,
    pub max_concurrent_models: usize,
}

/// Helper function to get current Unix timestamp.
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_status() {
        let status = ModelStatus::Available;
        assert!(!status.is_loaded());
        assert!(status.can_be_loaded());
        
        let loaded_status = ModelStatus::Loaded { 
            loaded_at: 12345, 
            memory_usage_bytes: 1000000 
        };
        assert!(loaded_status.is_loaded());
        assert!(!loaded_status.can_be_loaded());
    }
    
    #[test]
    fn test_model_metrics() {
        let mut metrics = ModelMetrics::default();
        
        // Add some successful transcriptions
        metrics.update(2.0, 1000, 0.95, true);
        metrics.update(3.0, 1500, 0.90, true);
        
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.total_audio_duration, 5.0);
        assert_eq!(metrics.success_rate(), 1.0);
        assert!((metrics.average_confidence - 0.925).abs() < 0.001);
        
        // Add a failed transcription
        metrics.update(1.0, 500, 0.0, false);
        
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.failed_requests, 1);
        assert!((metrics.success_rate() - 0.667).abs() < 0.001);
    }
    
    #[tokio::test]
    async fn test_registry_basic_operations() {
        let registry = ModelRegistry::new(1, 1024 * 1024 * 1024); // 1GB limit
        
        // Test listing models
        let models = registry.list_models().await;
        assert_eq!(models.len(), 5); // All model sizes
        
        // Test getting specific model
        let tiny_model = registry.get_model(ModelSize::Tiny).await;
        assert!(tiny_model.is_some());
        
        // Test updating status
        registry.update_model_status(
            ModelSize::Tiny, 
            ModelStatus::Loaded { loaded_at: 12345, memory_usage_bytes: 50000000 }
        ).await;
        
        let current = registry.get_current_loaded().await;
        assert_eq!(current, Some(ModelSize::Tiny));
        
        // Test model recommendation
        let recommended = registry.recommend_model(true, 0.0, Some(100)).await;
        assert_eq!(recommended, Some(ModelSize::Tiny)); // Should recommend loaded model
    }
}