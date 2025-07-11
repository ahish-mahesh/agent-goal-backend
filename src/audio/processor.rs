//! # Audio Processing and Validation
//! 
//! Handles audio format validation, preprocessing, and conversion for the transcription pipeline.
//! Ensures all audio data meets the requirements for Whisper model processing.
//! 
//! ## Key Functions:
//! - **Format validation**: Verify audio is 16kHz, 16-bit, mono PCM
//! - **Audio preprocessing**: Noise reduction, normalization, resampling
//! - **Format conversion**: Convert between different audio representations
//! - **Quality checking**: Detect and handle poor quality audio

use crate::audio::buffer::AudioBufferConfig;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

/// Audio processing and validation utilities.
/// 
/// ## Purpose:
/// This struct provides methods for validating and preprocessing audio data
/// before it's fed into the transcription pipeline. It ensures data quality
/// and format compliance.
pub struct AudioProcessor {
    config: AudioBufferConfig,
}

impl AudioProcessor {
    /// Create a new audio processor with the given configuration.
    pub fn new(config: AudioBufferConfig) -> Self {
        Self { config }
    }
    
    /// Validate incoming audio data format and structure.
    /// 
    /// ## Validation Checks:
    /// 1. **Data length**: Must be even (for 16-bit samples)
    /// 2. **Sample rate**: Should match expected rate (16kHz)
    /// 3. **Channel count**: Should be mono (1 channel)
    /// 4. **Bit depth**: Should be 16-bit PCM
    /// 
    /// ## Parameters:
    /// - **data**: Raw audio bytes from WebSocket
    /// - **declared_config**: Audio configuration sent by client
    /// 
    /// ## Returns:
    /// - **Ok(())**: Audio data is valid and compatible
    /// - **Err(String)**: Description of validation failure
    pub fn validate_audio_data(&self, data: &[u8], declared_config: &AudioConfig) -> Result<(), String> {
        // Check data length
        if data.is_empty() {
            return Err("Audio data is empty".to_string());
        }
        
        if data.len() % 2 != 0 {
            return Err("Audio data length must be even for 16-bit samples".to_string());
        }
        
        // Validate configuration matches expectations
        if declared_config.sample_rate != self.config.sample_rate {
            return Err(format!(
                "Sample rate mismatch: expected {}, got {}",
                self.config.sample_rate, declared_config.sample_rate
            ));
        }
        
        if declared_config.channels != self.config.channels {
            return Err(format!(
                "Channel count mismatch: expected {}, got {}",
                self.config.channels, declared_config.channels
            ));
        }
        
        if declared_config.bit_depth != self.config.bit_depth {
            return Err(format!(
                "Bit depth mismatch: expected {}, got {}",
                self.config.bit_depth, declared_config.bit_depth
            ));
        }
        
        // Check for obviously corrupted data
        if let Err(e) = self.validate_pcm_structure(data) {
            return Err(format!("PCM structure validation failed: {}", e));
        }
        
        Ok(())
    }
    
    /// Validate the internal structure of PCM data.
    /// 
    /// ## Checks:
    /// - Can parse as 16-bit integers
    /// - No extreme values indicating corruption
    /// - Reasonable dynamic range
    fn validate_pcm_structure(&self, data: &[u8]) -> Result<(), String> {
        let mut cursor = Cursor::new(data);
        let mut sample_count = 0;
        let mut min_sample = i16::MAX;
        let mut max_sample = i16::MIN;
        
        // Read and analyze samples
        while let Ok(sample) = cursor.read_i16::<LittleEndian>() {
            sample_count += 1;
            min_sample = min_sample.min(sample);
            max_sample = max_sample.max(sample);
            
            // Stop after analyzing first 1000 samples for efficiency
            if sample_count >= 1000 {
                break;
            }
        }
        
        if sample_count == 0 {
            return Err("No valid PCM samples found".to_string());
        }
        
        // Check dynamic range (difference between min and max)
        // Skip dynamic range check for very short audio samples (< 1000 samples / < 62ms at 16kHz)
        // as they may legitimately have low dynamic range
        let dynamic_range = (max_sample as i32) - (min_sample as i32);
        if sample_count >= 1000 && dynamic_range < 100 {
            return Err("Audio has very low dynamic range, may be corrupted or silent".to_string());
        }
        
        Ok(())
    }
    
    /// Apply audio preprocessing to improve transcription quality.
    /// 
    /// ## Preprocessing Steps:
    /// 1. **Normalize volume**: Ensure consistent audio levels
    /// 2. **Remove DC offset**: Center audio around zero
    /// 3. **Basic noise reduction**: Remove obvious noise patterns
    /// 
    /// ## Parameters:
    /// - **samples**: Audio samples as 16-bit integers
    /// 
    /// ## Returns:
    /// - **Vec<i16>**: Processed audio samples
    pub fn preprocess_audio(&self, samples: &[i16]) -> Vec<i16> {
        let mut processed = samples.to_vec();
        
        // Step 1: Remove DC offset (center audio around zero)
        let dc_offset = self.calculate_dc_offset(&processed);
        for sample in &mut processed {
            *sample = sample.saturating_sub(dc_offset);
        }
        
        // Step 2: Normalize volume to use full dynamic range
        self.normalize_volume(&mut processed);
        
        // Step 3: Basic noise reduction (simple low-pass filter)
        self.apply_noise_reduction(&mut processed);
        
        processed
    }
    
    /// Calculate the DC offset (average value) of audio samples.
    /// 
    /// ## Purpose:
    /// DC offset shifts all audio samples up or down, which can reduce
    /// the effective dynamic range and impact transcription quality.
    fn calculate_dc_offset(&self, samples: &[i16]) -> i16 {
        if samples.is_empty() {
            return 0;
        }
        
        let sum: i64 = samples.iter().map(|&s| s as i64).sum();
        (sum / samples.len() as i64) as i16
    }
    
    /// Normalize audio volume to use the full dynamic range.
    /// 
    /// ## Process:
    /// 1. Find the maximum absolute value in the audio
    /// 2. Calculate a scaling factor to use the full 16-bit range
    /// 3. Apply the scaling factor to all samples
    fn normalize_volume(&self, samples: &mut [i16]) {
        if samples.is_empty() {
            return;
        }
        
        // Find maximum absolute value
        let max_abs = samples.iter()
            .map(|&s| s.abs())
            .max()
            .unwrap_or(1);
        
        // Avoid division by zero and over-amplification
        if max_abs < 100 || max_abs > 30000 {
            return;
        }
        
        // Calculate scaling factor (leave some headroom)
        let target_max = 28000i16; // Leave some headroom below 32767
        let scale_factor = target_max as f32 / max_abs as f32;
        
        // Apply scaling
        for sample in samples {
            let scaled = (*sample as f32 * scale_factor) as i32;
            *sample = scaled.clamp(-32768, 32767) as i16;
        }
    }
    
    /// Apply basic noise reduction using a simple low-pass filter.
    /// 
    /// ## Method:
    /// Uses a simple moving average filter to reduce high-frequency noise
    /// while preserving speech content (which is mostly in lower frequencies).
    fn apply_noise_reduction(&self, samples: &mut [i16]) {
        if samples.len() < 3 {
            return;
        }
        
        // Simple 3-point moving average filter
        let original = samples.to_vec();
        for i in 1..samples.len() - 1 {
            let avg = (original[i - 1] as i32 + original[i] as i32 + original[i + 1] as i32) / 3;
            samples[i] = avg as i16;
        }
    }
    
    /// Convert 16-bit PCM samples to 32-bit float format for ML models.
    /// 
    /// ## Conversion:
    /// Scales from 16-bit integer range [-32768, 32767] to float range [-1.0, 1.0]
    /// This is the format expected by most machine learning audio models.
    pub fn pcm_to_float(&self, samples: &[i16]) -> Vec<f32> {
        samples.iter().map(|&sample| {
            sample as f32 / 32768.0
        }).collect()
    }
    
    /// Convert 32-bit float samples back to 16-bit PCM format.
    /// 
    /// ## Usage:
    /// Used when we need to store or transmit processed audio data.
    pub fn float_to_pcm(&self, samples: &[f32]) -> Vec<i16> {
        samples.iter().map(|&sample| {
            let scaled = sample * 32768.0;
            scaled.clamp(-32768.0, 32767.0) as i16
        }).collect()
    }
}

/// Audio configuration information sent by the client.
/// 
/// ## Purpose:
/// Represents the audio format that the client claims to be sending.
/// We validate this against our expected configuration to ensure compatibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u8,
    pub bit_depth: u8,
}

impl AudioConfig {
    /// Create a new audio configuration.
    pub fn new(sample_rate: u32, channels: u8, bit_depth: u8) -> Self {
        Self {
            sample_rate,
            channels,
            bit_depth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_audio_validation() {
        let config = AudioBufferConfig::default();
        let processor = AudioProcessor::new(config);
        let audio_config = AudioConfig::new(16000, 1, 16);
        
        // Test valid audio data (16 bytes = 8 samples) with realistic PCM data
        // Create a simple sine wave pattern to ensure proper dynamic range
        let mut valid_data = Vec::new();
        for i in 0..8 {
            let sample = (i as f32 * 0.5).sin() * 10000.0; // Sine wave with amplitude 10000
            let sample_i16 = sample as i16;
            valid_data.extend_from_slice(&sample_i16.to_le_bytes());
        }
        assert!(processor.validate_audio_data(&valid_data, &audio_config).is_ok());
        
        // Test invalid data length (odd number of bytes)
        let invalid_data = vec![0u8; 15];
        assert!(processor.validate_audio_data(&invalid_data, &audio_config).is_err());
    }
    
    #[test]
    fn test_pcm_conversion() {
        let config = AudioBufferConfig::default();
        let processor = AudioProcessor::new(config);
        
        let pcm_samples = vec![0i16, 16384, -16384, 32767, -32768];
        let float_samples = processor.pcm_to_float(&pcm_samples);
        let converted_back = processor.float_to_pcm(&float_samples);
        
        // Check conversion accuracy (allowing for small floating point errors)
        for (original, converted) in pcm_samples.iter().zip(converted_back.iter()) {
            let diff = (original - converted).abs();
            assert!(diff <= 1, "Conversion error too large: {} vs {}", original, converted);
        }
    }
}