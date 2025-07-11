//! # Audio Processing Module
//! 
//! This module handles real-time audio streaming and processing for voice transcription.
//! It implements the WebSocket audio pipeline as specified in the API contract.
//! 
//! ## Key Components:
//! - **WebSocket Handler**: Manages audio streaming connections (`/ws/audio`)
//! - **Audio Buffer**: Circular buffer with overlap for context preservation
//! - **Audio Processor**: PCM validation, format conversion, preprocessing
//! - **Session Manager**: Tracks active audio sessions and their state
//! 
//! ## Audio Format Requirements:
//! - **Sample Rate**: 16kHz (16,000 Hz)
//! - **Bit Depth**: 16-bit PCM
//! - **Channels**: Mono (1 channel)
//! - **Encoding**: Little-endian signed integers
//! 
//! ## WebSocket Protocol:
//! Implements the audio streaming protocol defined in voice_app_api_contract.md
//! with support for session management, real-time audio data, and error handling.

// WebSocket handler is in src/websocket.rs at the root level
pub mod buffer;       // Circular audio buffer with overlap
pub mod processor;    // Audio preprocessing and validation
pub mod session;      // Session state management
