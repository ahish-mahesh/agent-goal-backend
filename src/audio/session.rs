//! # Audio Session Management
//! 
//! Manages the lifecycle of audio transcription sessions, including session state,
//! resource cleanup, and concurrent session limits. Each session represents one
//! continuous audio recording and transcription process.
//! 
//! ## Session Lifecycle:
//! 1. **Created**: Session initialized, waiting for audio
//! 2. **Recording**: Actively receiving audio data
//! 3. **Paused**: Recording paused, can be resumed
//! 4. **Processing**: Audio being transcribed
//! 5. **Completed**: Session finished successfully
//! 6. **Error**: Session failed and needs cleanup

use crate::audio::buffer::{AudioBuffer, AudioBufferConfig};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Current status of an audio session.
/// 
/// ## Rust Concepts:
/// - **#[derive(Debug, Clone)]**: Automatically implements debug printing and cloning
/// - **enum**: Type that can be one of several variants
/// - **Serialization ready**: Can be converted to JSON for WebSocket messages
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    /// Session created but not yet started
    Created,
    /// Currently recording audio
    Recording,
    /// Recording paused (can be resumed)
    Paused,
    /// Audio is being processed/transcribed
    Processing,
    /// Session completed successfully
    Completed,
    /// Session failed with an error
    Error(String),
}

impl SessionStatus {
    /// Convert status to string for API responses.
    /// 
    /// ## API Contract:
    /// Returns status strings that match the API contract specification.
    pub fn as_str(&self) -> &str {
        match self {
            SessionStatus::Created => "created",
            SessionStatus::Recording => "recording", 
            SessionStatus::Paused => "paused",
            SessionStatus::Processing => "processing",
            SessionStatus::Completed => "completed",
            SessionStatus::Error(_) => "error",
        }
    }
}

/// Represents a single audio transcription session.
/// 
/// ## Thread Safety:
/// Uses Arc<RwLock<>> for safe sharing between WebSocket and transcription threads.
/// Multiple threads can read session data, but only one can modify it at a time.
pub struct AudioSession {
    /// Unique identifier for this session
    pub session_id: String,
    
    /// Associated note ID (if saving to a specific note)
    pub note_id: Option<u64>,
    
    /// Current session status
    status: Arc<RwLock<SessionStatus>>,
    
    /// Audio buffer for this session
    pub buffer: AudioBuffer,
    
    /// When the session was created
    pub created_at: DateTime<Utc>,
    
    /// When recording started (if it has started)
    started_at: Arc<RwLock<Option<DateTime<Utc>>>>,
    
    /// When recording ended (if it has ended)
    ended_at: Arc<RwLock<Option<DateTime<Utc>>>>,
    
    /// Configuration used for this session
    pub config: AudioBufferConfig,
    
    /// Model being used for transcription
    pub model_name: String,
    
    /// Performance tracking
    performance: Arc<RwLock<SessionPerformance>>,
}

/// Performance metrics for a session.
#[derive(Debug, Default)]
struct SessionPerformance {
    /// Total bytes of audio data received
    total_audio_bytes: usize,
    
    /// Number of transcription requests made
    transcription_requests: u32,
    
    /// Total time spent on transcription (milliseconds)
    total_transcription_time_ms: u64,
    
    /// Number of errors encountered
    error_count: u32,
}

impl AudioSession {
    /// Create a new audio session.
    /// 
    /// ## Parameters:
    /// - **session_id**: Unique identifier (typically a UUID)
    /// - **note_id**: Optional note to associate with this session
    /// - **config**: Audio buffer configuration
    /// - **model_name**: Name of the Whisper model to use
    pub fn new(
        session_id: String,
        note_id: Option<u64>,
        config: AudioBufferConfig,
        model_name: String,
    ) -> Self {
        Self {
            session_id,
            note_id,
            status: Arc::new(RwLock::new(SessionStatus::Created)),
            buffer: AudioBuffer::new(config.clone()),
            created_at: Utc::now(),
            started_at: Arc::new(RwLock::new(None)),
            ended_at: Arc::new(RwLock::new(None)),
            config,
            model_name,
            performance: Arc::new(RwLock::new(SessionPerformance::default())),
        }
    }
    
    /// Get the current session status.
    pub fn status(&self) -> SessionStatus {
        self.status.read().unwrap().clone()
    }
    
    /// Update the session status.
    /// 
    /// ## Thread Safety:
    /// Uses write lock to ensure only one thread can change status at a time.
    pub fn set_status(&self, new_status: SessionStatus) {
        *self.status.write().unwrap() = new_status;
    }
    
    /// Start recording audio for this session.
    /// 
    /// ## State Transition:
    /// Created/Paused → Recording
    pub fn start_recording(&self) -> Result<(), String> {
        let current_status = self.status();
        
        match current_status {
            SessionStatus::Created | SessionStatus::Paused => {
                self.set_status(SessionStatus::Recording);
                let mut started_at = self.started_at.write().unwrap();
                if started_at.is_none() {
                    *started_at = Some(Utc::now());
                }
                Ok(())
            }
            _ => Err(format!("Cannot start recording from status: {:?}", current_status)),
        }
    }
    
    /// Pause recording for this session.
    /// 
    /// ## State Transition:
    /// Recording → Paused
    pub fn pause_recording(&self) -> Result<(), String> {
        let current_status = self.status();
        
        match current_status {
            SessionStatus::Recording => {
                self.set_status(SessionStatus::Paused);
                Ok(())
            }
            _ => Err(format!("Cannot pause from status: {:?}", current_status)),
        }
    }
    
    /// End the recording session.
    /// 
    /// ## State Transition:
    /// Recording/Paused → Processing → Completed
    pub fn end_recording(&self) -> Result<(), String> {
        let current_status = self.status();
        
        match current_status {
            SessionStatus::Recording | SessionStatus::Paused => {
                self.set_status(SessionStatus::Processing);
                *self.ended_at.write().unwrap() = Some(Utc::now());
                Ok(())
            }
            _ => Err(format!("Cannot end recording from status: {:?}", current_status)),
        }
    }
    
    /// Mark session as completed.
    pub fn complete(&self) {
        self.set_status(SessionStatus::Completed);
    }
    
    /// Mark session as failed with an error message.
    pub fn set_error(&self, error_message: String) {
        self.set_status(SessionStatus::Error(error_message));
        let mut perf = self.performance.write().unwrap();
        perf.error_count += 1;
    }
    
    /// Add audio data to this session's buffer.
    /// 
    /// ## Performance Tracking:
    /// Updates metrics about total audio bytes received.
    pub fn add_audio_data(&self, data: &[u8]) -> Result<(), String> {
        // Update performance metrics
        {
            let mut perf = self.performance.write().unwrap();
            perf.total_audio_bytes += data.len();
        }
        
        // Add to buffer
        self.buffer.append_pcm_data(data)
    }
    
    /// Record transcription performance metrics.
    pub fn record_transcription(&self, duration_ms: u64) {
        let mut perf = self.performance.write().unwrap();
        perf.transcription_requests += 1;
        perf.total_transcription_time_ms += duration_ms;
    }
    
    /// Get session duration in seconds (from creation to now, or to end time).
    pub fn duration_seconds(&self) -> f64 {
        let end_time = self.ended_at.read().unwrap().unwrap_or_else(Utc::now);
        let duration = end_time.signed_duration_since(self.created_at);
        duration.num_milliseconds() as f64 / 1000.0
    }
    
    /// Get recording duration in seconds (actual audio recording time).
    pub fn recording_duration_seconds(&self) -> f64 {
        if let Some(started) = *self.started_at.read().unwrap() {
            let end_time = self.ended_at.read().unwrap().unwrap_or_else(Utc::now);
            let duration = end_time.signed_duration_since(started);
            duration.num_milliseconds() as f64 / 1000.0
        } else {
            0.0
        }
    }
    
    /// Check if the session can accept more audio data.
    pub fn can_accept_audio(&self) -> bool {
        matches!(self.status(), SessionStatus::Recording)
    }
    
    /// Get performance summary for this session.
    pub fn get_performance_summary(&self) -> SessionPerformanceSummary {
        let perf = self.performance.read().unwrap();
        SessionPerformanceSummary {
            total_audio_bytes: perf.total_audio_bytes,
            transcription_requests: perf.transcription_requests,
            average_transcription_time_ms: if perf.transcription_requests > 0 {
                perf.total_transcription_time_ms / perf.transcription_requests as u64
            } else {
                0
            },
            error_count: perf.error_count,
            session_duration_seconds: self.duration_seconds(),
            recording_duration_seconds: self.recording_duration_seconds(),
        }
    }
}

/// Summary of session performance metrics.
#[derive(Debug)]
pub struct SessionPerformanceSummary {
    pub total_audio_bytes: usize,
    pub transcription_requests: u32,
    pub average_transcription_time_ms: u64,
    pub error_count: u32,
    pub session_duration_seconds: f64,
    pub recording_duration_seconds: f64,
}

/// Manages multiple concurrent audio sessions.
/// 
/// ## Thread Safety:
/// Uses RwLock to allow multiple readers (getting session info) or one writer
/// (creating/removing sessions) at a time.
/// 
/// ## Resource Management:
/// - Enforces maximum concurrent session limits
/// - Automatically cleans up completed/failed sessions
/// - Tracks resource usage across all sessions
pub struct SessionManager {
    /// Active sessions mapped by session ID
    sessions: Arc<RwLock<HashMap<String, Arc<AudioSession>>>>,
    
    /// Maximum number of concurrent sessions allowed
    max_concurrent_sessions: usize,
    
    /// Default configuration for new sessions
    default_config: AudioBufferConfig,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new(max_concurrent_sessions: usize, default_config: AudioBufferConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent_sessions,
            default_config,
        }
    }
    
    /// Create a new audio session.
    /// 
    /// ## Parameters:
    /// - **session_id**: Optional session ID. If None, a UUID will be generated
    /// - **note_id**: Optional note to associate with this session
    /// - **model_name**: Name of the Whisper model to use
    /// 
    /// ## Returns:
    /// - **Ok(session_id)**: Session created successfully
    /// - **Err(message)**: Failed to create (likely due to session limit)
    pub fn create_session(
        &self,
        session_id: Option<String>,
        note_id: Option<u64>,
        model_name: String,
    ) -> Result<String, String> {
        let mut sessions = self.sessions.write().unwrap();
        
        // Check session limit
        if sessions.len() >= self.max_concurrent_sessions {
            return Err(format!(
                "Maximum concurrent sessions ({}) reached",
                self.max_concurrent_sessions
            ));
        }
        
        // Use provided session ID or generate new one
        let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        
        // Check if session ID already exists
        if sessions.contains_key(&session_id) {
            return Err(format!("Session ID '{}' already exists", session_id));
        }
        
        // Create new session
        let session = AudioSession::new(
            session_id.clone(),
            note_id,
            self.default_config.clone(),
            model_name,
        );
        
        // Add to active sessions
        sessions.insert(session_id.clone(), Arc::new(session));
        
        Ok(session_id)
    }
    
    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Option<Arc<AudioSession>> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(session_id).cloned()
    }
    
    /// Remove a session (cleanup when completed or failed).
    pub fn remove_session(&self, session_id: &str) -> bool {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id).is_some()
    }
    
    /// Get the number of active sessions.
    pub fn active_session_count(&self) -> usize {
        let sessions = self.sessions.read().unwrap();
        sessions.len()
    }
    
    /// Get all active session IDs.
    pub fn get_active_session_ids(&self) -> Vec<String> {
        let sessions = self.sessions.read().unwrap();
        sessions.keys().cloned().collect()
    }
    
    /// Clean up completed or failed sessions.
    /// 
    /// ## Cleanup Criteria:
    /// - Sessions in Completed or Error status
    /// - Sessions older than a certain age (even if still active)
    pub fn cleanup_old_sessions(&self, max_age_seconds: u64) -> usize {
        let mut sessions = self.sessions.write().unwrap();
        let now = Utc::now();
        let mut removed_count = 0;
        
        // Collect session IDs to remove
        let mut to_remove = Vec::new();
        
        for (session_id, session) in sessions.iter() {
            let should_remove = match session.status() {
                SessionStatus::Completed | SessionStatus::Error(_) => true,
                _ => {
                    // Check age
                    let age_seconds = now.signed_duration_since(session.created_at).num_seconds();
                    age_seconds > max_age_seconds as i64
                }
            };
            
            if should_remove {
                to_remove.push(session_id.clone());
            }
        }
        
        // Remove sessions
        for session_id in to_remove {
            sessions.remove(&session_id);
            removed_count += 1;
        }
        
        removed_count
    }
    
    /// Get summary of all active sessions.
    pub fn get_sessions_summary(&self) -> SessionManagerSummary {
        let sessions = self.sessions.read().unwrap();
        
        let mut status_counts = HashMap::new();
        let mut total_audio_bytes = 0;
        
        for session in sessions.values() {
            // Count by status
            let status = session.status();
            *status_counts.entry(status.as_str().to_string()).or_insert(0) += 1;
            
            // Sum audio data
            let perf = session.get_performance_summary();
            total_audio_bytes += perf.total_audio_bytes;
        }
        
        SessionManagerSummary {
            total_sessions: sessions.len(),
            max_sessions: self.max_concurrent_sessions,
            status_counts,
            total_audio_bytes,
        }
    }
}

/// Summary of session manager state.
#[derive(Debug)]
pub struct SessionManagerSummary {
    pub total_sessions: usize,
    pub max_sessions: usize,
    pub status_counts: HashMap<String, usize>,
    pub total_audio_bytes: usize,
}