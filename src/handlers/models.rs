//! # Model Management REST API Handlers
//! 
//! Provides HTTP endpoints for managing Whisper models: loading, unloading,
//! status checking, and configuration. These endpoints work with the model
//! registry and transcription engine to provide centralized model management.
//! 
//! ## Available Endpoints:
//! - `GET /models/whisper` - List available models and current status
//! - `POST /models/whisper/load` - Load specific Whisper model  
//! - `POST /models/whisper/unload` - Unload current model
//! - `GET /models/status` - Model registry status and memory usage

use crate::{error::AppError, state::AppState};
use crate::transcription::model::ModelSize;
use crate::transcription::engine::TranscriptionEngine;
use crate::transcription::registry::{ModelRegistry};
use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;
use candle_core::Device;
use anyhow::Result;

/// Request body for loading a specific model.
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    /// Model size to load (tiny, base, small, medium, large)
    pub model_size: String,
    /// Optional device preference (cpu, gpu)
    pub device: Option<String>,
}

/// Request body for model configuration updates.
#[derive(Debug, Deserialize)]
pub struct ModelConfigRequest {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<u32>,
    /// Maximum concurrent models
    pub max_concurrent_models: Option<usize>,
}

/// Response structure for model information.
#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    pub size: String,
    pub name: String,
    pub description: String,
    pub size_mb: u32,
    pub performance: String,
    pub status: String,
    pub loaded: bool,
}

/// Response structure for model loading operation.
#[derive(Debug, Serialize)]
pub struct ModelLoadResponse {
    pub success: bool,
    pub message: String,
    pub model_size: String,
    pub load_time_seconds: Option<f64>,
    pub memory_usage_mb: Option<u32>,
}

/// List all available Whisper models with their current status.
/// 
/// ## Endpoint: `GET /api/v1/models/whisper`
/// 
/// ## Response:
/// ```json
/// {
///   "models": [
///     {
///       "size": "medium",
///       "name": "medium", 
///       "description": "Good accuracy, handles technical vocabulary",
///       "size_mb": 769,
///       "performance": "accurate",
///       "status": "available",
///       "loaded": false
///     }
///   ],
///   "current_loaded": "medium",
///   "total_memory_usage_mb": 769
/// }
/// ```
pub async fn list_whisper_models(
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, AppError> {
    // Get the global model registry (we'll need to create this)
    let config = app_state.get_config();
    let model_registry = Arc::new(RwLock::new(
        ModelRegistry::new(1, config.performance.max_model_memory_mb as usize * 1024 * 1024)
    ));
    
    let models = model_registry.read().await.list_models().await;
    let current_loaded = model_registry.read().await.get_current_loaded().await;
    let summary = model_registry.read().await.get_registry_summary().await;
    
    let model_responses: Vec<ModelInfoResponse> = models
        .into_iter()
        .map(|entry| ModelInfoResponse {
            size: entry.info.size.to_string(),
            name: entry.info.name,
            description: entry.info.description,
            size_mb: entry.info.size_mb,
            performance: entry.info.performance,
            status: entry.status.description(),
            loaded: entry.status.is_loaded(),
        })
        .collect();
    
    Ok(HttpResponse::Ok().json(json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "models": model_responses,
        "current_loaded": current_loaded.map(|s| s.to_string()),
        "total_memory_usage_mb": summary.total_memory_usage_bytes / (1024 * 1024),
        "memory_limit_mb": summary.memory_limit_bytes / (1024 * 1024),
        "max_concurrent_models": summary.max_concurrent_models
    })))
}

/// Load a specific Whisper model.
/// 
/// ## Endpoint: `POST /api/v1/models/whisper/load`
/// 
/// ## Request Body:
/// ```json
/// {
///   "model_size": "medium",
///   "device": "cpu"
/// }
/// ```
/// 
/// ## Response:
/// ```json
/// {
///   "success": true,
///   "message": "Model loaded successfully",
///   "model_size": "medium",
///   "load_time_seconds": 2.5,
///   "memory_usage_mb": 769
/// }
/// ```
pub async fn load_whisper_model(
    app_state: web::Data<AppState>,
    request: web::Json<LoadModelRequest>,
) -> Result<HttpResponse, AppError> {
    let start_time = std::time::Instant::now();
    
    // Parse model size
    let model_size: ModelSize = request.model_size.parse()
        .map_err(|e| AppError::ValidationError(format!("Invalid model size: {}", e)))?;
    
    // Determine device
    let device = match request.device.as_deref() {
        Some("gpu") => {
            // TODO: Implement GPU detection
            Device::Cpu // Fallback to CPU for now
        }
        Some("cpu") | None => Device::Cpu,
        Some(other) => {
            return Err(AppError::ValidationError(format!("Invalid device: {}", other)));
        }
    };
    
    // Create transcription engine and load model
    let config = app_state.get_config();
    let transcription_config = config.audio.to_transcription_config();
    let transcription_engine = Arc::new(RwLock::new(
        TranscriptionEngine::new(transcription_config, device)
    ));
    
    // Load the model
    let result = transcription_engine.write().await.load_model(model_size).await;
    
    match result {
        Ok(()) => {
            let load_time = start_time.elapsed();
            
            // TODO: Get actual memory usage from model
            let estimated_memory_mb = match model_size {
                ModelSize::Tiny => 39,
                ModelSize::Base => 74,
                ModelSize::Small => 244,
                ModelSize::Medium => 769,
                ModelSize::Large => 1550,
            };
            
            Ok(HttpResponse::Ok().json(ModelLoadResponse {
                success: true,
                message: format!("Whisper {} model loaded successfully", model_size),
                model_size: model_size.to_string(),
                load_time_seconds: Some(load_time.as_secs_f64()),
                memory_usage_mb: Some(estimated_memory_mb),
            }))
        }
        Err(e) => {
            Ok(HttpResponse::InternalServerError().json(ModelLoadResponse {
                success: false,
                message: format!("Failed to load model: {}", e),
                model_size: model_size.to_string(),
                load_time_seconds: None,
                memory_usage_mb: None,
            }))
        }
    }
}

/// Unload the currently loaded Whisper model.
/// 
/// ## Endpoint: `POST /api/v1/models/whisper/unload`
/// 
/// ## Response:
/// ```json
/// {
///   "success": true,
///   "message": "Model unloaded successfully",
///   "freed_memory_mb": 769
/// }
/// ```
pub async fn unload_whisper_model(
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, AppError> {
    // Create transcription engine
    let config = app_state.get_config();
    let transcription_config = config.audio.to_transcription_config();
    let transcription_engine = Arc::new(RwLock::new(
        TranscriptionEngine::new(transcription_config, Device::Cpu)
    ));
    
    // Check if model is loaded first
    let is_loaded = transcription_engine.read().await.is_model_loaded().await;
    
    if !is_loaded {
        return Ok(HttpResponse::BadRequest().json(json!({
            "success": false,
            "message": "No model is currently loaded",
            "freed_memory_mb": 0
        })));
    }
    
    // Unload the model
    transcription_engine.write().await.unload_model().await;
    
    Ok(HttpResponse::Ok().json(json!({
        "success": true,
        "message": "Model unloaded successfully",
        "freed_memory_mb": "unknown", // TODO: Track actual memory usage
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

/// Get overall model registry status and system information.
/// 
/// ## Endpoint: `GET /api/v1/models/status`
/// 
/// ## Response:
/// ```json
/// {
///   "system": {
///     "total_models": 5,
///     "loaded_models": 1,
///     "available_models": 4,
///     "memory_usage_mb": 769,
///     "memory_limit_mb": 4096
///   },
///   "current_model": {
///     "size": "medium",
///     "loaded_at": "2024-01-15T10:30:00Z",
///     "memory_usage_mb": 769
///   }
/// }
/// ```
pub async fn get_model_status(
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, AppError> {
    let config = app_state.get_config();
    let model_registry = Arc::new(RwLock::new(
        ModelRegistry::new(1, config.performance.max_model_memory_mb as usize * 1024 * 1024)
    ));
    
    let summary = model_registry.read().await.get_registry_summary().await;
    let current_loaded = model_registry.read().await.get_current_loaded().await;
    
    // Get current model info if one is loaded
    let current_model_info = if let Some(loaded_size) = current_loaded {
        if let Some(model_entry) = model_registry.read().await.get_model(loaded_size).await {
            Some(json!({
                "size": loaded_size.to_string(),
                "name": model_entry.info.name,
                "description": model_entry.info.description,
                "status": model_entry.status.description(),
                "last_updated": model_entry.last_updated
            }))
        } else {
            None
        }
    } else {
        None
    };
    
    Ok(HttpResponse::Ok().json(json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "system": {
            "total_models": summary.total_models,
            "loaded_models": summary.loaded_models,
            "available_models": summary.available_models,
            "memory_usage_mb": summary.total_memory_usage_bytes / (1024 * 1024),
            "memory_limit_mb": summary.memory_limit_bytes / (1024 * 1024),
            "max_concurrent_models": summary.max_concurrent_models
        },
        "current_model": current_model_info,
        "device": "cpu" // TODO: Dynamic device detection
    })))
}

/// Transcribe an uploaded audio file for testing purposes.
/// 
/// ## Endpoint: `POST /api/v1/transcribe/file`
/// 
/// ## Request:
/// Multipart form data with an audio file field named "audio"
/// 
/// ## Response:
/// ```json
/// {
///   "success": true,
///   "transcription": {
///     "text": "Hello, this is a test transcription.",
///     "confidence": 0.95,
///     "duration_seconds": 3.2,
///     "processing_time_ms": 1500,
///     "model_used": "medium"
///   }
/// }
/// ```
pub async fn transcribe_file(
    mut payload: actix_multipart::Multipart,
) -> Result<HttpResponse, AppError> {
    use actix_multipart::Field;
    use futures_util::stream::StreamExt;
    
    let start_time = std::time::Instant::now();
    
    // Process multipart form data
    let mut audio_data: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    
    while let Some(item) = payload.next().await {
        let mut field: Field = item.map_err(|e| AppError::ValidationError(format!("Multipart error: {}", e)))?;
        
        let content_disposition = field.content_disposition().ok_or_else(|| {
            AppError::ValidationError("Missing content disposition".to_string())
        })?;
        
        let field_name = content_disposition.get_name().ok_or_else(|| {
            AppError::ValidationError("Missing field name".to_string())
        })?;
        
        if field_name == "audio" {
            filename = content_disposition.get_filename().map(|s| s.to_string());
            
            let mut bytes = Vec::new();
            while let Some(chunk) = field.next().await {
                let chunk = chunk.map_err(|e| AppError::ValidationError(format!("Chunk error: {}", e)))?;
                bytes.extend_from_slice(&chunk);
            }
            
            audio_data = Some(bytes);
        }
    }
    
    let audio_bytes = audio_data.ok_or_else(|| {
        AppError::ValidationError("No audio file provided".to_string())
    })?;
    
    let filename = filename.unwrap_or_else(|| "unknown".to_string());
    
    // Validate file size (max 50MB)
    const MAX_FILE_SIZE: usize = 50 * 1024 * 1024;
    if audio_bytes.len() > MAX_FILE_SIZE {
        return Err(AppError::ValidationError(format!(
            "File too large: {} bytes (max: {} bytes)", 
            audio_bytes.len(), 
            MAX_FILE_SIZE
        )));
    }
    
    // For now, create a placeholder response
    // TODO: Implement actual audio file processing and conversion to PCM
    let processing_time = start_time.elapsed();
    
    // Mock transcription result for testing
    let mock_result = json!({
        "success": true,
        "file_info": {
            "filename": filename,
            "size_bytes": audio_bytes.len(),
            "format": "unknown" // TODO: Detect format
        },
        "transcription": {
            "text": format!("Mock transcription for file: {}", filename),
            "confidence": 0.85,
            "duration_seconds": 5.0, // TODO: Get actual duration
            "processing_time_ms": processing_time.as_millis() as u64,
            "model_used": "medium" // TODO: Use actual loaded model
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    Ok(HttpResponse::Ok().json(mock_result))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_model_request_parsing() {
        let json = r#"{"model_size": "medium", "device": "cpu"}"#;
        let request: LoadModelRequest = serde_json::from_str(json).unwrap();
        
        assert_eq!(request.model_size, "medium");
        assert_eq!(request.device, Some("cpu".to_string()));
    }
    
    #[test]
    fn test_model_size_parsing() {
        assert!("medium".parse::<ModelSize>().is_ok());
        assert!("invalid".parse::<ModelSize>().is_err());
    }
    
    #[test]
    fn test_model_info_response_serialization() {
        let response = ModelInfoResponse {
            size: "medium".to_string(),
            name: "medium".to_string(),
            description: "Good accuracy".to_string(),
            size_mb: 769,
            performance: "accurate".to_string(),
            status: "available".to_string(),
            loaded: false,
        };
        
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("medium"));
        assert!(json.contains("769"));
    }
}