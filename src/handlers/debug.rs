//! Debug endpoints for testing transcription engine and model loading
//! 
//! These endpoints are useful for development and debugging without requiring
//! WebSocket connections or complex test setups.

use actix_web::{web, HttpResponse, Result as ActixResult};
use serde::{Deserialize, Serialize};
use candle_core::Device;

use crate::transcription::engine::{TranscriptionEngine, TranscriptionConfig};
use crate::transcription::model::ModelSize;
use crate::error::AppError;

/// Request to test model loading
#[derive(Debug, Deserialize)]
pub struct TestModelLoadRequest {
    /// Model size to test (tiny, base, small, medium, large)
    pub model_size: String,
}

/// Response from model loading test
#[derive(Debug, Serialize)]
pub struct TestModelLoadResponse {
    pub success: bool,
    pub message: String,
    pub model_size: String,
    pub load_time_ms: u64,
    pub error_details: Option<String>,
}

/// Request to test transcription with sample audio
#[derive(Debug, Deserialize)]
pub struct TestTranscriptionRequest {
    /// Model size to use (optional, defaults to medium)
    pub model_size: Option<String>,
    /// Audio duration in seconds for test (optional, defaults to 2.0)
    pub test_audio_duration: Option<f64>,
}

/// Response from transcription test
#[derive(Debug, Serialize)]
pub struct TestTranscriptionResponse {
    pub success: bool,
    pub transcription_result: Option<serde_json::Value>,
    pub error_details: Option<String>,
    pub test_audio_info: String,
}

/// Test model loading endpoint
/// 
/// POST /debug/test-model-load
/// Body: {"model_size": "medium"}
pub async fn test_model_load(
    req: web::Json<TestModelLoadRequest>,
) -> ActixResult<HttpResponse, AppError> {
    let start_time = std::time::Instant::now();
    
    // Parse model size
    let model_size = req.model_size.parse::<ModelSize>()
        .map_err(|e| AppError::ValidationError(format!("Invalid model size '{}': {}", req.model_size, e)))?;
    
    tracing::info!("Debug: Testing model loading for {}", model_size);
    
    // Create transcription engine with default config
    let config = TranscriptionConfig::default();
    let device = Device::Cpu; // Use CPU for debugging to avoid GPU issues
    let engine = TranscriptionEngine::new(config, device);
    
    // Attempt to load the model
    let result = engine.load_model(model_size).await;
    let load_time = start_time.elapsed();
    
    let response = match result {
        Ok(()) => {
            tracing::info!("Debug: Model {} loaded successfully in {:.2}s", model_size, load_time.as_secs_f64());
            TestModelLoadResponse {
                success: true,
                message: format!("Model {} loaded successfully", model_size),
                model_size: model_size.to_string(),
                load_time_ms: load_time.as_millis() as u64,
                error_details: None,
            }
        }
        Err(e) => {
            tracing::error!("Debug: Model {} loading failed: {}", model_size, e);
            TestModelLoadResponse {
                success: false,
                message: format!("Model {} loading failed", model_size),
                model_size: model_size.to_string(),
                load_time_ms: load_time.as_millis() as u64,
                error_details: Some(format!("{}", e)),
            }
        }
    };
    
    Ok(HttpResponse::Ok().json(response))
}

/// Test transcription with synthetic audio endpoint
/// 
/// POST /debug/test-transcription
/// Body: {"model_size": "medium", "test_audio_duration": 2.0}
pub async fn test_transcription(
    req: web::Json<TestTranscriptionRequest>,
) -> ActixResult<HttpResponse, AppError> {
    let model_size_str = req.model_size.as_deref().unwrap_or("medium");
    let audio_duration = req.test_audio_duration.unwrap_or(2.0);
    
    // Parse model size
    let model_size = model_size_str.parse::<ModelSize>()
        .map_err(|e| AppError::ValidationError(format!("Invalid model size '{}': {}", model_size_str, e)))?;
    
    tracing::info!("Debug: Testing transcription with {} model and {:.1}s audio", model_size, audio_duration);
    
    // Create transcription engine
    let config = TranscriptionConfig::default();
    let device = Device::Cpu;
    let engine = TranscriptionEngine::new(config, device);
    
    // Load the model first
    tracing::info!("Debug: Loading model {} for transcription test", model_size);
    if let Err(e) = engine.load_model(model_size).await {
        tracing::error!("Debug: Failed to load model for transcription test: {}", e);
        return Ok(HttpResponse::Ok().json(TestTranscriptionResponse {
            success: false,
            transcription_result: None,
            error_details: Some(format!("Failed to load model: {}", e)),
            test_audio_info: format!("Attempted to test with {:.1}s of synthetic audio", audio_duration),
        }));
    }
    
    // Generate test audio (sine wave at 440Hz)
    let sample_rate = 16000;
    let samples = (audio_duration * sample_rate as f64) as usize;
    let mut test_audio = Vec::with_capacity(samples);
    
    for i in 0..samples {
        let t = i as f64 / sample_rate as f64;
        // Generate a 440Hz sine wave (A note) with some amplitude variation
        let amplitude = 0.3 * (t * 2.0).sin(); // Slow amplitude modulation
        let sample = amplitude * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        test_audio.push(sample as f32);
    }
    
    tracing::info!("Debug: Generated {:.1}s of test audio ({} samples)", audio_duration, samples);
    
    // Attempt transcription
    let transcription_result = engine.transcribe(&test_audio).await;
    
    let response = match transcription_result {
        Ok(result) => {
            tracing::info!("Debug: Transcription successful: '{}'", result.text);
            TestTranscriptionResponse {
                success: true,
                transcription_result: Some(serde_json::to_value(&result).unwrap_or_default()),
                error_details: None,
                test_audio_info: format!("Generated {:.1}s of 440Hz sine wave test audio", audio_duration),
            }
        }
        Err(e) => {
            tracing::error!("Debug: Transcription failed: {}", e);
            TestTranscriptionResponse {
                success: false,
                transcription_result: None,
                error_details: Some(format!("{}", e)),
                test_audio_info: format!("Generated {:.1}s of 440Hz sine wave test audio", audio_duration),
            }
        }
    };
    
    Ok(HttpResponse::Ok().json(response))
}

/// Get debug info about the environment and configuration
/// 
/// GET /debug/environment
pub async fn debug_environment() -> ActixResult<HttpResponse, AppError> {
    let env_info = serde_json::json!({
        "hf_hub_url": std::env::var("HF_HUB_URL").unwrap_or_else(|_| "not set".to_string()),
        "hf_home": std::env::var("HF_HOME").unwrap_or_else(|_| "not set".to_string()),
        "hf_token": if std::env::var("HF_TOKEN").is_ok() { "set" } else { "not set" },
        "http_proxy": std::env::var("HTTP_PROXY").unwrap_or_else(|_| "not set".to_string()),
        "https_proxy": std::env::var("HTTPS_PROXY").unwrap_or_else(|_| "not set".to_string()),
        "no_proxy": std::env::var("NO_PROXY").unwrap_or_else(|_| "not set".to_string()),
        "device": "CPU (forced for debugging)",
        "available_models": ["tiny", "base", "small", "medium", "large"],
    });
    
    Ok(HttpResponse::Ok().json(env_info))
}

/// Configuration for debug routes
pub fn configure_debug_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/debug")
            .route("/test-model-load", web::post().to(test_model_load))
            .route("/test-transcription", web::post().to(test_transcription))
            .route("/environment", web::get().to(debug_environment))
    );
}