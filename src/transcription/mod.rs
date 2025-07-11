//! # Transcription Module
//! 
//! Handles speech-to-text transcription using Whisper models via the Candle-rs framework.
//! This module provides a pure Rust implementation without requiring FFI bindings to whisper.cpp.
//! 
//! ## Key Components:
//! - **Model Management**: Loading and unloading Whisper models
//! - **Transcription Engine**: Converting audio to text with confidence scores
//! - **Model Registry**: Tracking available and loaded models
//! - **Performance Monitoring**: Tracking inference times and accuracy
//! 
//! ## Whisper Model Sizes:
//! - **tiny**: ~39MB, fastest but least accurate
//! - **base**: ~74MB, good balance for development
//! - **small**: ~244MB, better accuracy
//! - **medium**: ~769MB, recommended for production (good technical vocabulary)
//! - **large**: ~1550MB, best accuracy but slowest
//! 
//! ## Candle-rs Integration:
//! Uses Candle-rs instead of whisper.cpp FFI for:
//! - Pure Rust implementation (no C++ dependencies)
//! - Better error handling and memory safety
//! - Integration with Rust async ecosystem
//! - Easier cross-platform deployment

pub mod model;       // Whisper model loading and management
pub mod engine;      // Transcription processing engine
pub mod registry;    // Model registry and availability tracking

pub use engine::{TranscriptionEngine, TranscriptionResult, TranscriptionConfig};
pub use registry::{ModelRegistry};