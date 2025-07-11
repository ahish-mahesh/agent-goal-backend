use crate::{error::AppError, state::AppState};
use actix_web::{web, HttpResponse};
use serde_json::json;

pub async fn get_config(state: web::Data<AppState>) -> Result<HttpResponse, AppError> {
    let config = state.get_config();
    
    Ok(HttpResponse::Ok().json(json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "config": {
            "server": {
                "host": config.server.host,
                "port": config.server.port
            },
            "models": {
                "whisper_model": config.models.whisper_model,
                "llm_model": config.models.llm_model
            },
            "performance": {
                "max_concurrent_sessions": config.performance.max_concurrent_sessions,
                "audio_buffer_size": config.performance.audio_buffer_size
            }
        }
    })))
}

pub async fn update_config(
    state: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
) -> Result<HttpResponse, AppError> {
    let json_str = serde_json::to_string(&body.into_inner())?;
    
    let mut current_config = state.get_config();
    current_config.update_from_json(&json_str)?;
    
    state.update_config(current_config.clone())
        .map_err(|e| AppError::ValidationError(e))?;

    Ok(HttpResponse::Ok().json(json!({
        "status": "success",
        "message": "Configuration updated successfully",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "updated_config": {
            "server": {
                "host": current_config.server.host,
                "port": current_config.server.port
            },
            "models": {
                "whisper_model": current_config.models.whisper_model,
                "llm_model": current_config.models.llm_model
            },
            "performance": {
                "max_concurrent_sessions": current_config.performance.max_concurrent_sessions,
                "audio_buffer_size": current_config.performance.audio_buffer_size
            }
        }
    })))
}