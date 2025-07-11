//! # WebSocket Audio Streaming Handler
//! 
//! Handles real-time audio streaming via WebSocket for speech-to-text transcription.
//! Clients connect to `/ws/audio` and send binary audio data for real-time processing.
//! 
//! ## WebSocket Protocol:
//! 1. **Connection**: Client connects with optional query parameters
//! 2. **Configuration**: First message should be JSON config with audio format
//! 3. **Audio Streaming**: Subsequent binary messages contain PCM audio data
//! 4. **Transcription Results**: Server sends JSON responses with transcribed text
//! 5. **Session Management**: Each connection represents one audio session
//! 
//! ## Message Format:
//! - **Client → Server**: Binary PCM audio data (16-bit, 16kHz, mono)
//! - **Server → Client**: JSON messages with transcription results and status

use crate::config::AppConfig;
use crate::state::AppState;
use crate::audio::session::{SessionManager, AudioSession};
use crate::audio::buffer::AudioBufferConfig;
use crate::audio::processor::{AudioProcessor, AudioConfig as ProcessorAudioConfig};
use crate::transcription::engine::TranscriptionEngine;
use crate::transcription::registry::ModelRegistry;

use actix_web::{web, HttpRequest, HttpResponse, Result as ActixResult};
use actix_web_actors::ws;
use actix::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::RwLock;
use candle_core::Device;
use tracing::{info, warn, error, debug};

/// WebSocket message types for client-server communication.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Start recording session message from client
    #[serde(rename = "start_session")]
    StartSession {
        /// Session ID for this recording session
        session_id: String,
        /// Optional note ID to associate transcriptions with
        note_id: Option<u64>,
        /// Audio format configuration
        audio_config: ProcessorAudioConfig,
    },
    
    /// Audio data chunk from client
    #[serde(rename = "audio_data")]
    AudioData {
        /// Session ID for this audio chunk
        session_id: String,
        /// Binary audio data (will be sent as binary WebSocket message)
        #[serde(skip_serializing)]
        data: Vec<u8>,
        /// Timestamp when audio was captured
        timestamp: u64,
    },
    
    /// Pause session message from client
    #[serde(rename = "pause_session")]
    PauseSession {
        /// Session ID to pause
        session_id: String,
    },
    
    /// Resume session message from client
    #[serde(rename = "resume_session")]
    ResumeSession {
        /// Session ID to resume
        session_id: String,
    },
    
    /// End session message from client
    #[serde(rename = "end_session")]
    EndSession {
        /// Session ID to end
        session_id: String,
    },
    
    /// Initial configuration message from client (deprecated, use StartSession)
    #[serde(rename = "config")]
    Config {
        /// Audio format configuration
        audio: ProcessorAudioConfig,
        /// Optional note ID to associate transcriptions with
        note_id: Option<u64>,
        /// Preferred model size for transcription
        model_size: Option<String>,
        /// Target language for transcription
        language: Option<String>,
    },
    
    /// Transcription result from server
    #[serde(rename = "transcription")]
    Transcription {
        /// Session ID for this transcription
        session_id: String,
        /// Transcribed text
        text: String,
        /// Confidence score (0.0 to 1.0)
        confidence: f32,
        /// Whether this is a partial or final result
        is_final: bool,
        /// Processing time in milliseconds
        processing_time_ms: u64,
        /// Timestamp when transcription was completed
        timestamp: u64,
    },
    
    /// Session status updates (API contract format)
    #[serde(rename = "session_status")]
    SessionStatus {
        /// Session ID
        session_id: String,
        /// Current session status
        status: String,
        /// Additional status message
        message: Option<String>,
    },
    
    /// Session status updates (deprecated format)
    #[serde(rename = "status")]
    Status {
        /// Session ID
        session_id: String,
        /// Current session status
        status: String,
        /// Additional status message
        message: Option<String>,
    },
    
    /// Error messages
    #[serde(rename = "error")]
    Error {
        /// Error code
        code: String,
        /// Human-readable error message
        message: String,
        /// Optional session ID if error is session-specific
        session_id: Option<String>,
    },
    
    /// Heartbeat/ping message
    #[serde(rename = "ping")]
    Ping {
        /// Timestamp for latency measurement
        timestamp: u64,
    },
    
    /// Heartbeat/pong response
    #[serde(rename = "pong")]
    Pong {
        /// Original timestamp from ping
        timestamp: u64,
    },
}

/// WebSocket actor for handling audio streaming connections.
/// 
/// ## Actor Model:
/// Uses Actix's actor system where each WebSocket connection is an independent actor
/// that can receive and send messages asynchronously.
pub struct AudioWebSocket {
    /// Unique session ID for this connection
    session_id: Option<String>,
    
    /// Audio session for this connection
    audio_session: Option<Arc<AudioSession>>,
    
    /// Audio processor for validation and preprocessing
    audio_processor: Option<AudioProcessor>,
    
    /// Transcription engine
    transcription_engine: Arc<RwLock<TranscriptionEngine>>,
    
    /// Model registry for managing Whisper models
    model_registry: Arc<RwLock<ModelRegistry>>,
    
    /// Session manager for tracking active sessions
    session_manager: Arc<RwLock<SessionManager>>,
    
    /// Application configuration
    config: AppConfig,
    
    /// Last heartbeat time
    last_heartbeat: Instant,
    
    /// Whether the session has been configured
    is_configured: bool,
}

impl AudioWebSocket {
    /// Create a new WebSocket actor.
    pub fn new(
        app_state: web::Data<AppState>,
        transcription_engine: Arc<RwLock<TranscriptionEngine>>,
        model_registry: Arc<RwLock<ModelRegistry>>,
        session_manager: Arc<RwLock<SessionManager>>,
    ) -> Self {
        Self {
            session_id: None,
            audio_session: None,
            audio_processor: None,
            transcription_engine,
            model_registry,
            session_manager,
            config: app_state.get_config(),
            last_heartbeat: Instant::now(),
            is_configured: false,
        }
    }
    
    /// Static version of handle_config_message for use in spawned tasks.
    async fn handle_config_message_static(
        frontend_session_id: String,
        audio_config: ProcessorAudioConfig,
        note_id: Option<u64>,
        model_size: Option<String>,
        _language: Option<String>,
        session_manager: &Arc<RwLock<SessionManager>>,
        config: &AppConfig,
        transcription_engine: &Arc<RwLock<TranscriptionEngine>>,
    ) -> Result<(), String> {
        // Validate audio configuration
        let buffer_config = AudioBufferConfig {
            sample_rate: audio_config.sample_rate,
            channels: audio_config.channels,
            bit_depth: audio_config.bit_depth,
            buffer_duration_ms: config.audio.buffer_duration_ms,
            overlap_duration_ms: config.audio.overlap_duration_ms,
        };
        
        // Load the Whisper model for this session
        let model_name = model_size.unwrap_or_else(|| config.models.whisper_model.clone());
        let model_size_enum = model_name.parse().map_err(|e| format!("Invalid model size: {}", e))?;
        
        // Load model in transcription engine
        transcription_engine.write().await.load_model(model_size_enum).await
            .map_err(|e| format!("Failed to load model: {}", e))?;
        
        // Create audio session using the frontend-provided session ID
        match session_manager.write().await.create_session(Some(frontend_session_id.clone()), note_id, model_name.clone()) {
            Ok(_actual_session_id) => {
                info!("WebSocket session {} configured successfully", frontend_session_id);
                Ok(())
            }
            Err(err) => {
                Err(err)
            }
        }
    }

    /// Handle configuration message from client.
    async fn handle_config_message(
        &mut self,
        audio_config: ProcessorAudioConfig,
        note_id: Option<u64>,
        model_size: Option<String>,
        _language: Option<String>,
        session_manager: &Arc<RwLock<SessionManager>>,
        config: &AppConfig,
    ) -> Result<String, String> {
        // Validate audio configuration
        let buffer_config = AudioBufferConfig {
            sample_rate: audio_config.sample_rate,
            channels: audio_config.channels,
            bit_depth: audio_config.bit_depth,
            buffer_duration_ms: config.audio.buffer_duration_ms,
            overlap_duration_ms: config.audio.overlap_duration_ms,
        };
        
        // Create audio processor
        self.audio_processor = Some(AudioProcessor::new(buffer_config.clone()));
        
        // Load the Whisper model for this session
        let model_name = model_size.unwrap_or_else(|| config.models.whisper_model.clone());
        let model_size_enum = model_name.parse().map_err(|e| format!("Invalid model size: {}", e))?;
        
        // Load model in transcription engine
        self.transcription_engine.write().await.load_model(model_size_enum).await
            .map_err(|e| format!("Failed to load model: {}", e))?;
        
        // Create audio session
        match session_manager.write().await.create_session(None, note_id, model_name.clone()) {
            Ok(session_id) => {
                // Get the audio session
                if let Some(session) = session_manager.read().await.get_session(&session_id) {
                    self.audio_session = Some(session);
                    
                    info!("WebSocket session {} configured successfully", session_id);
                    Ok(session_id)
                } else {
                    Err("Failed to retrieve created session".to_string())
                }
            }
            Err(err) => {
                Err(err)
            }
        }
    }
    
    /// Handle binary audio data from client and trigger transcription if needed.
    fn handle_audio_data(&mut self, data: &[u8], ctx: &mut ws::WebsocketContext<Self>) -> Result<(), String> {
        if !self.is_configured {
            return Err("Session not configured. Send config message first.".to_string());
        }
        
        // Get session_id
        let session_id = match &self.session_id {
            Some(id) => id.clone(),
            None => return Err("No session ID available".to_string()),
        };
        
        // We'll look up the session when needed in the async task
        
        let processor = match &self.audio_processor {
            Some(processor) => processor,
            None => {
                return Err("Audio processor not initialized".to_string());
            }
        };
        
        // Validate audio data
        let audio_config = ProcessorAudioConfig::new(
            self.config.audio.sample_rate,
            self.config.audio.channels,
            self.config.audio.bit_depth,
        );
        
        if let Err(err) = processor.validate_audio_data(data, &audio_config) {
            return Err(format!("Invalid audio: {}", err));
        }
        
        // Clone data for the async task
        let data_clone = data.to_vec();
        let transcription_engine = self.transcription_engine.clone();
        let session_manager = self.session_manager.clone();
        let audio_config = ProcessorAudioConfig::new(
            self.config.audio.sample_rate,
            self.config.audio.channels,
            self.config.audio.bit_depth,
        );
        let addr = ctx.address();
        
        // Spawn async task to handle audio data
        tokio::spawn(async move {
            // Get the session from session manager
            let session = match session_manager.read().await.get_session(&session_id) {
                Some(session) => session,
                None => {
                    error!("Audio session not found: {}", session_id);
                    return;
                }
            };
            
            // Add audio data to session buffer
            if let Err(err) = session.add_audio_data(&data_clone) {
                error!("Buffer error: {}", err);
                return;
            }
            
            // Check if we have sufficient audio for transcription
            if session.buffer.has_sufficient_audio() {
                match transcription_engine.write().await.transcribe_from_buffer(&session.buffer).await {
                    Ok(result) => {
                        // Send transcription result back to WebSocket client
                        let text_len = result.text.len();
                        let msg = SendTranscriptionResult {
                            session_id: session_id.clone(),
                            text: result.text,
                            confidence: result.confidence,
                            processing_time_ms: result.processing_time_ms,
                            timestamp: result.timestamp,
                        };
                        
                        addr.do_send(msg);
                        
                        debug!("Transcription completed: {} chars in {}ms", 
                            text_len, result.processing_time_ms);
                    }
                    Err(err) => {
                        error!("Transcription failed: {}", err);
                    }
                }
            }
        });
        
        debug!("Received {} bytes of audio data", data.len());
        
        Ok(())
    }

    /// Handle pong message from client.
    /// For every pong message received, send a ping message back.
    fn handle_pong(&mut self, timestamp: u64, ctx: &mut ws::WebsocketContext<Self>) {
        let ping_msg = WebSocketMessage::Ping { timestamp };
        if let Ok(json) = serde_json::to_string(&ping_msg) {
            ctx.text(json);
        }
        self.last_heartbeat = Instant::now();
    }
    
    /// Send error message to client.
    fn send_error(&self, ctx: &mut ws::WebsocketContext<Self>, code: &str, message: &str, session_id: Option<String>) {
        let error_msg = WebSocketMessage::Error {
            code: code.to_string(),
            message: message.to_string(),
            session_id,
        };
        
        if let Ok(json) = serde_json::to_string(&error_msg) {
            ctx.text(json);
        }
        
        warn!("WebSocket error {}: {}", code, message);
    }
    
    /// Clean up session when connection closes.
    fn handle_pause_session(&mut self, session_id: String, ctx: &mut ws::WebsocketContext<Self>) {
        // TODO: Implement pause session logic
        tracing::warn!("Pause session not yet implemented: {}", session_id);
        
        let status_msg = WebSocketMessage::SessionStatus {
            session_id: session_id.clone(),
            status: "paused".to_string(),
            message: Some("Session paused (not fully implemented)".to_string()),
        };
        
        if let Ok(json) = serde_json::to_string(&status_msg) {
            ctx.text(json);
        }
    }
    
    fn handle_resume_session(&mut self, session_id: String, ctx: &mut ws::WebsocketContext<Self>) {
        // TODO: Implement resume session logic
        tracing::warn!("Resume session not yet implemented: {}", session_id);
        
        let status_msg = WebSocketMessage::SessionStatus {
            session_id: session_id.clone(),
            status: "resumed".to_string(),
            message: Some("Session resumed (not fully implemented)".to_string()),
        };
        
        if let Ok(json) = serde_json::to_string(&status_msg) {
            ctx.text(json);
        }
    }
    
    fn handle_end_session(&mut self, session_id: String, ctx: &mut ws::WebsocketContext<Self>) {
        // TODO: Implement end session logic
        tracing::warn!("End session not yet implemented: {}", session_id);
        
        let status_msg = WebSocketMessage::SessionStatus {
            session_id: session_id.clone(),
            status: "ended".to_string(),
            message: Some("Session ended (not fully implemented)".to_string()),
        };
        
        if let Ok(json) = serde_json::to_string(&status_msg) {
            ctx.text(json);
        }
    }

    async fn cleanup_session(&mut self) {
        if let Some(session_id) = &self.session_id {
            let session_manager = self.session_manager.clone();
            session_manager.write().await.remove_session(session_id);
            info!("Cleaned up WebSocket session: {}", session_id);
        }
    }
}

/// Message for sending text to WebSocket client.
#[derive(Message)]
#[rtype(result = "()")]
struct SendText(String);

/// Message for sending transcription results to WebSocket client.
#[derive(Message)]
#[rtype(result = "()")]
struct SendTranscriptionResult {
    session_id: String,
    text: String,
    confidence: f32,
    processing_time_ms: u64,
    timestamp: u64,
}

/// Message for setting session as configured.
#[derive(Message)]
#[rtype(result = "()")]
struct SetSessionConfigured {
    session_id: String,
    audio_config: ProcessorAudioConfig,
}

/// Implement Actor trait for WebSocket handling.
impl Actor for AudioWebSocket {
    type Context = ws::WebsocketContext<Self>;
    
    /// Called when the WebSocket connection starts.
    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket connection started");
        
        // Start heartbeat timer
        ctx.run_interval(Duration::from_secs(30), |act, ctx| {
            // Check if client is still responsive
            if Instant::now().duration_since(act.last_heartbeat) > Duration::from_secs(60) {
                warn!("WebSocket heartbeat timeout, closing connection");
                ctx.stop();
            } else {
                // Send ping
                let ping_msg = WebSocketMessage::Ping {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                
                if let Ok(json) = serde_json::to_string(&ping_msg) {
                    ctx.text(json);
                }
            }
        });
    }
    
    /// Called when the WebSocket connection stops.
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("WebSocket connection stopped");
        
        // Clean up session in background
        let session_manager = self.session_manager.clone();
        let session_id = self.session_id.clone();
        
        tokio::spawn(async move {
            if let Some(id) = session_id {
                session_manager.write().await.remove_session(&id);
            }
        });
    }
}

/// Handle incoming WebSocket messages.
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for AudioWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                // Handle JSON messages
                match serde_json::from_str::<WebSocketMessage>(&text) {
                    Ok(WebSocketMessage::StartSession { session_id, note_id, audio_config }) => {
                        // Handle start_session message according to API contract
                        let session_manager = self.session_manager.clone();
                        let config = self.config.clone();
                        let transcription_engine = self.transcription_engine.clone();
                        let session_id_clone = session_id.clone();
                        let ctx_addr = ctx.address();
                        
                        // Send configuration task to be handled asynchronously
                        tokio::spawn(async move {
                            // Create configuration result message
                            let audio_config_clone = audio_config.clone();
                            let result = Self::handle_config_message_static(
                                session_id_clone.clone(),
                                audio_config, 
                                note_id, 
                                None, 
                                None, 
                                &session_manager, 
                                &config,
                                &transcription_engine
                            ).await;
                            
                            match result {
                                Ok(()) => {
                                    // Send message to update actor state - notification will be sent from there
                                    ctx_addr.do_send(SetSessionConfigured {
                                        session_id: session_id_clone,
                                        audio_config: audio_config_clone,
                                    });
                                }
                                Err(err) => {
                                    let error_msg = WebSocketMessage::Error {
                                        code: "session_error".to_string(),
                                        message: err,
                                        session_id: Some(session_id_clone),
                                    };
                                    
                                    if let Ok(json) = serde_json::to_string(&error_msg) {
                                        ctx_addr.do_send(SendText(json));
                                    }
                                }
                            }
                        });
                    }
                    Ok(WebSocketMessage::Config { audio, note_id, model_size, language }) => {
                        // Handle config message (deprecated path) 
                        let session_manager = self.session_manager.clone();
                        let config = self.config.clone();
                        let transcription_engine = self.transcription_engine.clone();
                        let ctx_addr = ctx.address();
                        
                        tokio::spawn(async move {
                            let audio_clone = audio.clone();
                            // For legacy config messages, generate a session ID
                            let generated_session_id = format!("session_{}", std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis());
                            
                            let result = Self::handle_config_message_static(
                                generated_session_id.clone(),
                                audio, 
                                note_id, 
                                model_size, 
                                language, 
                                &session_manager, 
                                &config,
                                &transcription_engine
                            ).await;
                            
                            match result {
                                Ok(()) => {
                                    // Send message to update actor state - notification will be sent from there
                                    ctx_addr.do_send(SetSessionConfigured {
                                        session_id: generated_session_id,
                                        audio_config: audio_clone,
                                    });
                                }
                                Err(err) => {
                                    let error_msg = WebSocketMessage::Error {
                                        code: "config_error".to_string(),
                                        message: err,
                                        session_id: None,
                                    };
                                    
                                    if let Ok(json) = serde_json::to_string(&error_msg) {
                                        ctx_addr.do_send(SendText(json));
                                    }
                                }
                            }
                        });
                    }
                    Ok(WebSocketMessage::PauseSession { session_id }) => {
                        let session_manager = self.session_manager.clone();
                        
                        self.handle_pause_session(session_id, ctx);
                    }
                    Ok(WebSocketMessage::ResumeSession { session_id }) => {
                        self.handle_resume_session(session_id, ctx);
                    }
                    Ok(WebSocketMessage::EndSession { session_id }) => {
                        self.handle_end_session(session_id, ctx);
                    }
                    Ok(WebSocketMessage::Pong { timestamp }) => {
                        self.handle_pong(timestamp, ctx);
                    }
                    Ok(_) => {
                        warn!("Received unexpected message type from client");
                    }
                    Err(err) => {
                        self.send_error(ctx, "invalid_json", &format!("Invalid JSON: {}", err), None);
                    }
                }
            }
            Ok(ws::Message::Binary(data)) => {
                // Handle binary audio data and trigger transcription
                if let Err(err) = self.handle_audio_data(&data, ctx) {
                    self.send_error(ctx, "audio_error", &err, self.session_id.clone());
                }
            }
            Ok(ws::Message::Ping(data)) => {
                ctx.pong(&data);
                self.last_heartbeat = Instant::now();
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_heartbeat = Instant::now();
            }
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket closed: {:?}", reason);
                ctx.stop();
            }
            Ok(ws::Message::Continuation(_)) => {
                // Handle continuation frames (not used in our protocol)
                warn!("Received unexpected continuation frame");
            }
            Ok(ws::Message::Nop) => {
                // Handle no-op frames (usually internal)
            }
            Err(err) => {
                error!("WebSocket protocol error: {}", err);
                ctx.stop();
            }
        }
    }
}

/// Handle SendText messages.
impl Handler<SendText> for AudioWebSocket {
    type Result = ();
    
    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}

/// Handle SendTranscriptionResult messages.
impl Handler<SendTranscriptionResult> for AudioWebSocket {
    type Result = ();
    
    fn handle(&mut self, msg: SendTranscriptionResult, ctx: &mut Self::Context) {
        let transcription_msg = WebSocketMessage::Transcription {
            session_id: msg.session_id,
            text: msg.text,
            confidence: msg.confidence,
            is_final: true, // For now, all results are final
            processing_time_ms: msg.processing_time_ms,
            timestamp: msg.timestamp,
        };
        
        if let Ok(json) = serde_json::to_string(&transcription_msg) {
            ctx.text(json);
        }
    }
}

/// Handle SetSessionConfigured messages.
impl Handler<SetSessionConfigured> for AudioWebSocket {
    type Result = ();
    
    fn handle(&mut self, msg: SetSessionConfigured, ctx: &mut Self::Context) {
        self.session_id = Some(msg.session_id.clone());
        self.is_configured = true;
        
        // Create audio processor
        let buffer_config = AudioBufferConfig {
            sample_rate: msg.audio_config.sample_rate,
            channels: msg.audio_config.channels,
            bit_depth: msg.audio_config.bit_depth,
            buffer_duration_ms: self.config.audio.buffer_duration_ms,
            overlap_duration_ms: self.config.audio.overlap_duration_ms,
        };
        self.audio_processor = Some(AudioProcessor::new(buffer_config));
        
        // Send success notification to frontend
        // Use the appropriate message format based on whether this looks like a generated session ID
        let status_msg = if msg.session_id.starts_with("session_") {
            // Legacy config message - use Status format
            info!("WebSocket session {} configured with legacy config message", msg.session_id);
            WebSocketMessage::Status {
                session_id: msg.session_id.clone(),
                status: "configured".to_string(),
                message: Some("Session configured successfully".to_string()),
            }
        } else {
            // New start_session message - use SessionStatus format
            info!("WebSocket session {} started with new start_session message", msg.session_id);
            WebSocketMessage::SessionStatus {
                session_id: msg.session_id.clone(),
                status: "started".to_string(),
                message: Some("Session started successfully".to_string()),
            }
        };
        
        if let Ok(json) = serde_json::to_string(&status_msg) {
            ctx.text(json);
        }
        
        info!("WebSocket session {} is now fully configured and ready for audio data", msg.session_id);
    }
}

/// WebSocket endpoint handler.
/// 
/// ## HTTP to WebSocket Upgrade:
/// This function handles the initial HTTP request and upgrades it to a WebSocket connection.
/// The actual WebSocket communication is handled by the AudioWebSocket actor.
pub async fn audio_websocket(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    info!("New WebSocket connection request from: {:?}", req.connection_info().peer_addr());
    
    // Extract query parameters
    let query = web::Query::<std::collections::HashMap<String, String>>::from_query(req.query_string())
        .unwrap_or_else(|_| web::Query(std::collections::HashMap::new()));
    
    debug!("WebSocket query parameters: {:?}", query.iter().collect::<Vec<_>>());
    
    // Initialize transcription components
    let device = Device::Cpu; // TODO: Support GPU detection
    let config = app_state.get_config();
    let transcription_config = config.audio.to_transcription_config();
    let transcription_engine = Arc::new(RwLock::new(
        TranscriptionEngine::new(transcription_config, device)
    ));
    
    let config = app_state.get_config();
    let model_registry = Arc::new(RwLock::new(
        ModelRegistry::new(1, config.performance.max_model_memory_mb as usize * 1024 * 1024)
    ));
    
    let session_manager = Arc::new(RwLock::new(
        SessionManager::new(
            config.performance.max_concurrent_sessions,
            config.audio.to_buffer_config(),
        )
    ));
    
    // Create WebSocket actor
    let websocket = AudioWebSocket::new(
        app_state,
        transcription_engine,
        model_registry,
        session_manager,
    );
    
    // Start WebSocket connection
    ws::start(websocket, &req, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_websocket_message_serialization() {
        let config_msg = WebSocketMessage::Config {
            audio: ProcessorAudioConfig::new(16000, 1, 16),
            note_id: Some(123),
            model_size: Some("medium".to_string()),
            language: Some("en".to_string()),
        };
        
        let json = serde_json::to_string(&config_msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::Config { audio, note_id, model_size, language } => {
                assert_eq!(audio.sample_rate, 16000);
                assert_eq!(note_id, Some(123));
                assert_eq!(model_size, Some("medium".to_string()));
                assert_eq!(language, Some("en".to_string()));
            }
            _ => panic!("Wrong message type"),
        }
    }
    
    #[test]
    fn test_error_message_creation() {
        let error_msg = WebSocketMessage::Error {
            code: "test_error".to_string(),
            message: "Test error message".to_string(),
            session_id: Some("session123".to_string()),
        };
        
        let json = serde_json::to_string(&error_msg).unwrap();
        assert!(json.contains("test_error"));
        assert!(json.contains("Test error message"));
        assert!(json.contains("session123"));
    }
}