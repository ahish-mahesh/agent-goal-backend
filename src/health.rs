use crate::state::AppState;
use actix_web::{web, HttpResponse};
use serde_json::json;
use std::process;

pub async fn health_check(state: web::Data<AppState>) -> HttpResponse {
    let metrics = state.get_metrics_snapshot();
    let config = state.get_config();
    let uptime_seconds = state.get_uptime_seconds();
    
    let memory_info = get_memory_info();
    let system_status = get_system_status(&config, &metrics);

    HttpResponse::Ok().json(json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": uptime_seconds,
        "service": {
            "name": "agent-notes-backend",
            "version": env!("CARGO_PKG_VERSION"),
            "host": config.server.host,
            "port": config.server.port
        },
        "metrics": {
            "total_requests": metrics.request_count,
            "total_errors": metrics.error_count,
            "error_rate": if metrics.request_count > 0 {
                metrics.error_count as f64 / metrics.request_count as f64
            } else {
                0.0
            },
            "active_sessions": metrics.active_sessions
        },
        "memory": memory_info,
        "models": {
            "whisper": {
                "model": config.models.whisper_model,
                "status": "not_loaded",
                "available": false
            },
            "llm": {
                "model": config.models.llm_model,
                "status": "not_loaded",
                "available": false
            }
        },
        "system": system_status
    }))
}

pub async fn detailed_metrics(state: web::Data<AppState>) -> HttpResponse {
    let metrics = state.get_metrics_snapshot();
    let uptime_seconds = state.get_uptime_seconds();

    let mut endpoint_stats = Vec::new();
    for (endpoint, metric) in metrics.endpoint_metrics.iter() {
        endpoint_stats.push(json!({
            "endpoint": endpoint,
            "request_count": metric.request_count,
            "error_count": metric.error_count,
            "error_rate": metric.error_rate(),
            "average_duration_ms": metric.average_duration_ms(),
            "total_duration_ms": metric.total_duration_ms
        }));
    }

    HttpResponse::Ok().json(json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": uptime_seconds,
        "overall": {
            "total_requests": metrics.request_count,
            "total_errors": metrics.error_count,
            "error_rate": if metrics.request_count > 0 {
                metrics.error_count as f64 / metrics.request_count as f64
            } else {
                0.0
            },
            "active_sessions": metrics.active_sessions,
            "requests_per_second": if uptime_seconds > 0 {
                metrics.request_count as f64 / uptime_seconds as f64
            } else {
                0.0
            }
        },
        "endpoints": endpoint_stats,
        "memory": get_memory_info(),
        "performance": {
            "max_concurrent_sessions": state.get_config().performance.max_concurrent_sessions,
            "max_model_memory_mb": state.get_config().performance.max_model_memory_mb
        }
    }))
}

fn get_memory_info() -> serde_json::Value {
    let pid = process::id();
    
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string(format!("/proc/{}/status", pid)) {
            let mut vm_rss = 0;
            let mut vm_size = 0;
            
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        vm_rss = kb_str.parse::<u64>().unwrap_or(0) * 1024;
                    }
                } else if line.starts_with("VmSize:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        vm_size = kb_str.parse::<u64>().unwrap_or(0) * 1024;
                    }
                }
            }
            
            return json!({
                "resident_memory_bytes": vm_rss,
                "virtual_memory_bytes": vm_size,
                "available": true
            });
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        json!({
            "resident_memory_bytes": 0,
            "virtual_memory_bytes": 0,
            "available": false,
            "note": "Memory info not available on macOS"
        })
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        json!({
            "resident_memory_bytes": 0,
            "virtual_memory_bytes": 0,
            "available": false,
            "note": "Memory info not available on this platform"
        })
    }
}

fn get_system_status(config: &crate::config::AppConfig, metrics: &crate::state::AppMetrics) -> serde_json::Value {
    let session_usage = if config.performance.max_concurrent_sessions > 0 {
        metrics.active_sessions as f64 / config.performance.max_concurrent_sessions as f64
    } else {
        0.0
    };

    let status = if session_usage > 0.9 {
        "high_load"
    } else if session_usage > 0.7 {
        "moderate_load"
    } else {
        "normal"
    };

    json!({
        "status": status,
        "session_usage_percent": (session_usage * 100.0).round(),
        "max_sessions": config.performance.max_concurrent_sessions,
        "current_sessions": metrics.active_sessions,
        "load_warnings": if session_usage > 0.8 {
            vec!["High session usage - consider increasing max_concurrent_sessions"]
        } else {
            vec![]
        }
    })
}