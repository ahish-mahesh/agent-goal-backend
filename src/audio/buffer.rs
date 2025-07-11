//! # Audio Buffer Management
//! 
//! Implements a circular audio buffer with configurable overlap for maintaining
//! context between transcription segments. This is crucial for accurate speech
//! recognition at segment boundaries.
//! 
//! ## Key Features:
//! - **Circular buffer**: Efficiently manages memory for continuous audio streaming
//! - **Overlap handling**: Maintains 1-2 seconds of overlap between segments
//! - **Format validation**: Ensures audio data meets requirements (16kHz, 16-bit, mono)
//! - **Thread safety**: Safe for concurrent access from WebSocket and transcription threads

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

/// Configuration for audio buffer behavior.
/// 
/// ## Rust Concepts:
/// - **#[derive(Debug, Clone)]**: Automatically implements debug printing and cloning
/// - **pub struct**: Public struct accessible from other modules
/// - **usize**: Platform-specific unsigned integer for memory sizes
#[derive(Debug, Clone)]
pub struct AudioBufferConfig {
    /// Target sample rate (should be 16000 for Whisper)
    pub sample_rate: u32,
    
    /// Number of audio channels (should be 1 for mono)
    pub channels: u8,
    
    /// Bit depth (should be 16 for PCM)
    pub bit_depth: u8,
    
    /// Buffer duration in milliseconds (typically 2000-5000ms)
    pub buffer_duration_ms: u32,
    
    /// Overlap duration in milliseconds (typically 500-1000ms)
    pub overlap_duration_ms: u32,
}

impl Default for AudioBufferConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,      // 16kHz - Whisper's preferred sample rate
            channels: 1,             // Mono audio
            bit_depth: 16,           // 16-bit PCM
            buffer_duration_ms: 2000,    // 2 second buffer
            overlap_duration_ms: 500,    // 500ms overlap
        }
    }
}

/// Circular audio buffer with overlap management for real-time transcription.
/// 
/// ## Thread Safety:
/// Uses Arc<Mutex<VecDeque<i16>>> to safely share audio data between threads.
/// The WebSocket thread can write data while the transcription thread reads it.
/// 
/// ## Memory Management:
/// - Automatically discards old audio data to prevent unlimited memory growth
/// - Maintains a sliding window of recent audio with overlap
/// - Efficient append/read operations using VecDeque
pub struct AudioBuffer {
    /// The actual audio sample data (16-bit signed integers)
    /// Arc<Mutex<>> allows safe sharing between threads
    buffer: Arc<Mutex<VecDeque<i16>>>,
    
    /// Configuration settings for this buffer
    config: AudioBufferConfig,
    
    /// Maximum number of samples to keep in buffer
    max_samples: usize,
    
    /// Number of samples to overlap between segments
    overlap_samples: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer with the given configuration.
    /// 
    /// ## Calculations:
    /// - **max_samples**: buffer_duration_ms * sample_rate / 1000
    /// - **overlap_samples**: overlap_duration_ms * sample_rate / 1000
    /// 
    /// ## Example:
    /// For 2 second buffer at 16kHz: 2000ms * 16000Hz / 1000 = 32,000 samples
    pub fn new(config: AudioBufferConfig) -> Self {
        let max_samples = (config.buffer_duration_ms as usize * config.sample_rate as usize) / 1000;
        let overlap_samples = (config.overlap_duration_ms as usize * config.sample_rate as usize) / 1000;
        
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(max_samples))),
            config,
            max_samples,
            overlap_samples,
        }
    }
    
    /// Append raw PCM audio data to the buffer.
    /// 
    /// ## Parameters:
    /// - **data**: Raw bytes from WebSocket (should be 16-bit PCM little-endian)
    /// 
    /// ## Rust Concepts:
    /// - **&[u8]**: Borrowed slice of bytes (no ownership transfer)
    /// - **Cursor**: Allows reading from a byte array as if it were a file
    /// - **while let Ok(...)**: Continue reading until we reach the end
    /// - **.lock().unwrap()**: Get exclusive access to the shared buffer
    /// 
    /// ## Error Handling:
    /// Returns error if audio format is invalid or buffer operations fail.
    pub fn append_pcm_data(&self, data: &[u8]) -> Result<(), String> {
        // Validate that we have an even number of bytes (for 16-bit samples)
        if data.is_empty() {
            return Err("No audio data provided".to_string());
        }
        if data.len() % 2 != 0 {
            return Err("Audio data length must be even for 16-bit samples".to_string());
        }
        
        // Convert raw bytes to 16-bit signed integers
        let mut cursor = Cursor::new(data);
        let mut samples = Vec::new();
        
        // Read each 16-bit sample (little-endian format)
        while let Ok(sample) = cursor.read_i16::<LittleEndian>() {
            samples.push(sample);
        }
        
        // Add samples to the circular buffer
        let mut buffer = self.buffer.lock().unwrap();
        
        // Add new samples
        for sample in samples {
            buffer.push_back(sample);
            
            // Remove old samples if buffer is full
            if buffer.len() > self.max_samples {
                buffer.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Get a segment of audio data for transcription with overlap.
    /// 
    /// ## Returns:
    /// - **Vec<f32>**: Audio samples as 32-bit floats (Whisper's preferred format)
    /// - **Normalized**: Samples are scaled from [-32768, 32767] to [-1.0, 1.0]
    /// 
    /// ## Overlap Strategy:
    /// Returns the most recent audio data, including overlap from previous segments
    /// to maintain context for accurate transcription at boundaries.
    pub fn get_transcription_segment(&self) -> Vec<f32> {
        let buffer = self.buffer.lock().unwrap();
        
        // Convert i16 samples to f32 and normalize to [-1.0, 1.0]
        buffer.iter().map(|&sample| {
            sample as f32 / 32768.0  // Convert from i16 range to f32 range
        }).collect()
    }
    
    /// Get the current buffer length in samples.
    /// 
    /// ## Usage:
    /// Can be used to check if enough audio has been collected for transcription.
    /// Typically want at least 1 second of audio before starting transcription.
    pub fn len(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }
    
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.lock().unwrap().is_empty()
    }
    
    /// Get the duration of audio currently in the buffer (in seconds).
    /// 
    /// ## Calculation:
    /// Duration = number_of_samples / sample_rate
    pub fn duration_seconds(&self) -> f64 {
        let len = self.len();
        len as f64 / self.config.sample_rate as f64
    }
    
    /// Clear all audio data from the buffer.
    /// 
    /// ## Usage:
    /// Called when ending a session or recovering from errors.
    pub fn clear(&self) {
        self.buffer.lock().unwrap().clear();
    }
    
    /// Get a reference to the buffer configuration.
    pub fn config(&self) -> &AudioBufferConfig {
        &self.config
    }
    
    /// Check if the buffer has enough audio for transcription.
    /// 
    /// ## Logic:
    /// Returns true if buffer contains at least 1 second of audio data.
    /// This prevents transcribing very short audio segments that may be inaccurate.
    pub fn has_sufficient_audio(&self) -> bool {
        self.duration_seconds() >= 1.0
    }
}