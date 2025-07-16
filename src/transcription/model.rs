//! # Whisper Model Management
//! 
//! Handles loading, unloading, and management of Whisper models using Candle-rs.
//! Provides abstractions for different model sizes and configurations.
//! 
//! ## Model Loading Process:
//! 1. Download model files from HuggingFace if not cached locally
//! 2. Load model weights and tokenizer
//! 3. Initialize model on appropriate device (CPU/GPU)
//! 4. Validate model functionality with test input
//! 
//! ## Memory Management:
//! - Models are loaded on-demand to conserve memory
//! - Only one model loaded at a time (configurable)
//! - Automatic cleanup when switching models

use candle_core::{Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
// Removed unused import - now using ApiBuilder
use tokenizers::Tokenizer;
use anyhow::{Result, anyhow};
use std::error::Error;

/// Available Whisper model sizes with their characteristics.
/// 
/// ## Trade-offs:
/// - **Size vs Accuracy**: Larger models are more accurate but slower
/// - **Memory vs Speed**: More memory usage for better performance
/// - **Language vs Domain**: Some models better for specific languages/domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl ModelSize {
    /// Get the HuggingFace model repository name.
    pub fn repo_name(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "openai/whisper-tiny",
            ModelSize::Base => "openai/whisper-base", 
            ModelSize::Small => "openai/whisper-small",
            ModelSize::Medium => "openai/whisper-medium",
            ModelSize::Large => "openai/whisper-large-v2",
        }
    }
    
    /// Get the approximate model size in MB.
    pub fn size_mb(&self) -> u32 {
        match self {
            ModelSize::Tiny => 39,
            ModelSize::Base => 74,
            ModelSize::Small => 244,
            ModelSize::Medium => 769,
            ModelSize::Large => 1550,
        }
    }
    
    /// Get the expected performance characteristics.
    pub fn performance(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "fast",
            ModelSize::Base => "fast",
            ModelSize::Small => "balanced",
            ModelSize::Medium => "accurate",
            ModelSize::Large => "accurate",
        }
    }
    
    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "Fastest, basic accuracy",
            ModelSize::Base => "Fast, good for testing",
            ModelSize::Small => "Balanced speed and accuracy",
            ModelSize::Medium => "Good accuracy, handles technical vocabulary",
            ModelSize::Large => "Best accuracy, slower processing",
        }
    }
}

impl std::str::FromStr for ModelSize {
    type Err = anyhow::Error;
    
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(ModelSize::Tiny),
            "base" => Ok(ModelSize::Base),
            "small" => Ok(ModelSize::Small),
            "medium" => Ok(ModelSize::Medium),
            "large" => Ok(ModelSize::Large),
            _ => Err(anyhow!("Unknown model size: {}", s)),
        }
    }
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ModelSize::Tiny => "tiny",
            ModelSize::Base => "base",
            ModelSize::Small => "small", 
            ModelSize::Medium => "medium",
            ModelSize::Large => "large",
        };
        write!(f, "{}", name)
    }
}

/// Information about a Whisper model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub size: ModelSize,
    pub name: String,
    pub description: String,
    pub size_mb: u32,
    pub performance: String,
    pub loaded: bool,
}

impl ModelInfo {
    pub fn new(size: ModelSize) -> Self {
        Self {
            name: size.to_string(),
            description: size.description().to_string(),
            size_mb: size.size_mb(),
            performance: size.performance().to_string(),
            loaded: false,
            size,
        }
    }
}

/// A loaded Whisper model ready for transcription.
/// 
/// ## Thread Safety:
/// Wrapped in Arc<RwLock<>> to allow safe sharing between threads.
/// Multiple transcription requests can read the model simultaneously.
pub struct WhisperModel {
    /// The actual Candle model
    model: m::model::Whisper,
    
    /// Model configuration
    config: Config,
    
    /// Device where model is loaded (CPU/GPU)
    device: Device,
    
    /// Model size/type information
    info: ModelInfo,
    
    /// Tokenizer for text processing
    tokenizer: Tokenizer,
    
    /// Mel filters for audio preprocessing
    mel_filters: Vec<f32>,
}

impl WhisperModel {
    /// Load a Whisper model from HuggingFace.
    /// 
    /// ## Loading Process:
    /// 1. Create HuggingFace API client
    /// 2. Download model files (cached locally)
    /// 3. Load tokenizer and configuration
    /// 4. Initialize model weights
    /// 5. Validate model works with test input
    /// 
    /// ## Parameters:
    /// - **size**: Which model size to load
    /// - **device**: Device to load model on (CPU/GPU)
    /// 
    /// ## Returns:
    /// - **Ok(WhisperModel)**: Model loaded successfully
    /// - **Err(anyhow::Error)**: Loading failed with error details
    pub async fn load(size: ModelSize, device: Device) -> Result<Self> {
        tracing::info!("Loading Whisper {} model...", size);
        let start_time = std::time::Instant::now();
        
        // Initialize HuggingFace API with detailed logging
        tracing::info!("Initializing HuggingFace API...");
        
        // Log all relevant environment variables
        tracing::debug!("Environment variables:");
        tracing::debug!("  HF_ENDPOINT: {:?}", std::env::var("HF_ENDPOINT"));
        tracing::debug!("  HF_HOME: {:?}", std::env::var("HF_HOME"));
        tracing::debug!("  HF_TOKEN: {:?}", std::env::var("HF_TOKEN").map(|_| "***SET***"));
        tracing::debug!("  HF_HUB_CACHE: {:?}", std::env::var("HF_HUB_CACHE"));
        tracing::debug!("  HF_HUB_OFFLINE: {:?}", std::env::var("HF_HUB_OFFLINE"));
        tracing::debug!("  HTTPS_PROXY: {:?}", std::env::var("HTTPS_PROXY"));
        tracing::debug!("  HTTP_PROXY: {:?}", std::env::var("HTTP_PROXY"));
        
        // Try different API initialization approaches for hf-hub 0.4.3
        let api = {
            use hf_hub::api::tokio::{Api, ApiBuilder};
            
            // Try ApiBuilder first
            tracing::debug!("Attempting ApiBuilder initialization...");
            let builder_result = {
                let mut builder = ApiBuilder::new();
                
                // Configure with environment variables
                if let Ok(token) = std::env::var("HF_TOKEN") {
                    tracing::debug!("Setting HF token (length: {})", token.len());
                    builder = builder.with_token(Some(token));
                } else {
                    tracing::debug!("No HF_TOKEN set, proceeding without authentication");
                    builder = builder.with_token(None);
                }
                
                // Configure cache directory if specified
                if let Ok(cache_dir) = std::env::var("HF_HUB_CACHE") {
                    tracing::debug!("Setting custom cache dir: {}", cache_dir);
                    builder = builder.with_cache_dir(cache_dir.into());
                } else if let Ok(hf_home) = std::env::var("HF_HOME") {
                    let cache_path = std::path::PathBuf::from(hf_home).join("hub");
                    tracing::debug!("Setting cache dir from HF_HOME: {:?}", cache_path);
                    builder = builder.with_cache_dir(cache_path);
                } else {
                    tracing::debug!("Using default cache directory");
                }
                
                builder = builder.with_progress(false);
                
                tracing::debug!("Building API client with ApiBuilder...");
                builder.build()
            };
            
            match builder_result {
                Ok(api) => {
                    tracing::info!("HuggingFace API client created successfully with ApiBuilder");
                    api
                }
                Err(e) => {
                    tracing::error!("ApiBuilder failed: {}, trying fallback to Api::new()", e);
                    
                    // Fallback to basic Api::new()
                    tracing::debug!("Attempting fallback with Api::new()...");
                    Api::new().map_err(|e2| {
                        anyhow!("Both ApiBuilder and Api::new() failed. ApiBuilder error: {}. Api::new() error: {}. This suggests a fundamental issue with hf-hub configuration or network connectivity.", e, e2)
                    })?
                }
            }
        };

        tracing::info!("Creating model repository handle for: {}", size.repo_name());
        let repo = api.model(size.repo_name().to_string());
        
        // Test basic repository access
        tracing::debug!("Repository created, testing basic functionality...");
        tracing::debug!("Repository URL for file access: {:?}", repo.url("")); // Test URL generation
        
        // Download required files with detailed error handling
        tracing::info!("Starting download of model files from {}", size.repo_name());
        
        // Try to download config.json with detailed logging
        tracing::debug!("Attempting to download config.json...");
        tracing::debug!("Repository URL for config.json: {:?}", repo.url("config.json"));
        
        let config_filename = match repo.get("config.json").await {
            Ok(path) => {
                tracing::info!("Successfully downloaded config.json to: {:?}", path);
                path
            }
            Err(e) => {
                tracing::error!("Failed to download config.json: {}", e);
                tracing::error!("Error type: {:?}", e);
                tracing::debug!("Full error chain:");
                let mut source = e.source();
                let mut level = 0;
                while let Some(err) = source {
                    tracing::debug!("  Level {}: {}", level, err);
                    source = err.source();
                    level += 1;
                }
                return Err(anyhow!("Failed to download config.json from {}: {}", size.repo_name(), e));
            }
        };
        let tokenizer_filename = repo.get("tokenizer.json").await
            .map_err(|e| anyhow!("Failed to download tokenizer.json from {}: {}", size.repo_name(), e))?;
        let model_filename = match repo.get("model.safetensors").await {
            Ok(path) => path,
            Err(_) => repo.get("pytorch_model.bin").await
                .map_err(|e| anyhow!("Failed to download model weights from {}: {}", size.repo_name(), e))?,
        };
        
        // Load configuration
        let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        tracing::debug!("Model config: {:?}", config);
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Load mel filters based on model configuration
        let mel_filters = Self::load_mel_filters(&config)?;
        
        // Load model weights
        tracing::debug!("Loading model weights...");
        let vb = if model_filename.to_string_lossy().ends_with(".safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_filename], m::DTYPE, &device)?
            }
        } else {
            // For now, only support safetensors format
            return Err(anyhow!("Only safetensors format is supported in this implementation"));
        };
        
        // Initialize model
        let model = m::model::Whisper::load(&vb, config.clone())?;
        
        let load_time = start_time.elapsed();
        tracing::info!("Whisper {} model loaded in {:.2}s", size, load_time.as_secs_f64());
        
        // Create model info
        let mut info = ModelInfo::new(size);
        info.loaded = true;
        
        let mut whisper_model = Self {
            model,
            config,
            device,
            info,
            tokenizer,
            mel_filters,
        };
        
        // Validate the model
        // tracing::info!("Skipping validation for now, assuming model is valid");
        whisper_model.validate().await?;
        
        Ok(whisper_model)
    }
    
    /// Load mel filters based on model configuration.
    fn load_mel_filters(config: &Config) -> Result<Vec<f32>> {
        // Generate mel filters programmatically for now
        // This is a simplified implementation - in production, you'd load pre-computed filters
        let n_fft = 400; // Standard for 16kHz Whisper
        let n_mels = config.num_mel_bins as usize;
        
        // Create a basic mel filter bank
        let mel_filters = Self::create_mel_filter_bank(n_fft, n_mels);
        Ok(mel_filters)
    }
    
    /// Create a basic mel filter bank.
    fn create_mel_filter_bank(n_fft: usize, n_mels: usize) -> Vec<f32> {
        // Simplified mel filter bank creation
        // In a full implementation, this would use proper mel scale conversion
        let mut filters = vec![0.0f32; n_fft * n_mels];
        
        for i in 0..n_mels {
            for j in 0..n_fft {
                // Simple triangular filters (placeholder)
                let center = (i + 1) * n_fft / (n_mels + 1);
                let width = n_fft / (n_mels + 1);
                
                if j >= center.saturating_sub(width) && j <= center + width {
                    let distance = (j as i32 - center as i32).abs() as f32;
                    filters[i * n_fft + j] = (1.0 - distance / width as f32).max(0.0);
                }
            }
        }
        
        filters
    }
    
    /// Convert PCM audio data to mel spectrogram.
    fn pcm_to_mel(&self, pcm_data: &[f32]) -> Result<Tensor> {
        // Simplified mel spectrogram conversion
        // In a full implementation, this would include proper STFT and mel filtering
        
        // Pad or truncate to 30 seconds (480,000 samples at 16kHz)
        let target_len = 30 * 16000;
        let mut padded_audio = vec![0.0f32; target_len];
        let copy_len = pcm_data.len().min(target_len);
        padded_audio[..copy_len].copy_from_slice(&pcm_data[..copy_len]);
        
        // Create a simplified mel spectrogram
        // This is a placeholder - real implementation would use FFT and mel filters
        let n_mels = self.config.num_mel_bins as usize;
        let n_frames = 3000; // Standard Whisper frame count for 30s
        
        let mut mel_data = vec![0.0f32; n_mels * n_frames];
        
        // Simple energy-based features (placeholder)
        let frame_size = padded_audio.len() / n_frames;
        for frame in 0..n_frames {
            let start = frame * frame_size;
            let end = (start + frame_size).min(padded_audio.len());
            
            for mel_bin in 0..n_mels {
                // Compute energy in this frame for this mel bin
                let mut energy = 0.0f32;
                for i in start..end {
                    energy += padded_audio[i].abs();
                }
                
                // Apply log-mel scaling
                mel_data[mel_bin * n_frames + frame] = (energy / frame_size as f32).ln().max(-11.5129); // -80 dB floor
            }
        }
        
        Ok(Tensor::from_vec(mel_data, (n_mels, n_frames), &self.device)?)
    }
    
    /// Transcribe audio data to text.
    /// 
    /// ## Parameters:
    /// - **audio_data**: Audio samples as 32-bit floats, 16kHz sample rate
    /// - **language**: Optional language hint ("en", "es", etc.)
    /// 
    /// ## Returns:
    /// - **Ok(String)**: Transcribed text
    /// - **Err(anyhow::Error)**: Transcription failed
    /// 
    /// ## Audio Requirements:
    /// - Sample rate: 16kHz
    /// - Format: 32-bit float, range [-1.0, 1.0]
    /// - Channels: Mono (single channel)
    /// - Length: At least 1 second, max ~30 seconds for best results
    pub async fn transcribe(&mut self, audio_data: &[f32], language: Option<&str>) -> Result<String> {
        let start_time = std::time::Instant::now();
        
        // Validate audio data
        if audio_data.is_empty() {
            return Err(anyhow!("Audio data is empty"));
        }
        
        if audio_data.len() < 16000 {
            tracing::warn!("Audio data is shorter than 1 second, transcription may be inaccurate");
        }
        
        // Convert PCM to mel spectrogram
        let mel = self.pcm_to_mel(audio_data)?;
        let mel = mel.unsqueeze(0)?; // Add batch dimension
        
        // Run encoder
        let encoder_output = self.model.encoder.forward(&mel, false)?;
        
        // Prepare decoder inputs
        let mut tokens = vec![self.get_sot_token()];
        
        // Add language token if specified
        if let Some(lang) = language {
            if let Some(lang_token) = self.get_language_token(lang) {
                tokens.push(lang_token);
            }
        }
        
        // Add transcribe task token
        tokens.push(self.get_transcribe_token());
        
        // Initialize output tokens
        let mut output_tokens = Vec::new();
        
        // Decode iteratively with temperature fallback
        const MAX_TOKENS: usize = 200;
        const TEMPERATURES: &[f32] = &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        
        for &temperature in TEMPERATURES {
            tokens.truncate(3); // Keep only SOT, language, and task tokens
            output_tokens.clear();
            
            let mut decode_success = true;
            
            for _i in 0..MAX_TOKENS {
                let token_tensor = Tensor::new(&tokens[..], &self.device)?
                    .unsqueeze(0)?; // Add batch dimension
                
                let logits = self.model.decoder.forward(&token_tensor, &encoder_output, false)?;
                
                // Get the last token's logits
                let last_logits = logits.i((.., tokens.len() - 1, ..))?;
                
                // Apply temperature and sample
                let next_token = if temperature > 0.0 {
                    self.sample_token(&last_logits, temperature)?
                } else {
                    last_logits.argmax_keepdim(1)?.to_scalar::<u32>()?
                };
                
                // Check for end of text
                if next_token == self.get_eot_token() {
                    break;
                }
                
                // Check for repetition or invalid tokens
                if self.is_repetitive(&output_tokens, next_token) {
                    decode_success = false;
                    break;
                }
                
                tokens.push(next_token);
                output_tokens.push(next_token);
            }
            
            if decode_success && !output_tokens.is_empty() {
                break;
            }
        }
        
        // Decode tokens to text
        let text = self.decode_tokens(&output_tokens)?;
        
        let transcription_time = start_time.elapsed();
        tracing::debug!(
            "Transcribed {:.2}s of audio in {:.2}s: '{}'", 
            audio_data.len() as f64 / 16000.0,
            transcription_time.as_secs_f64(),
            text
        );
        
        Ok(text)
    }
    
    /// Get model information.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
    
    /// Get model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }
    
    /// Get the device where the model is loaded.
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Estimate memory usage of the loaded model in bytes.
    pub fn estimated_memory_usage(&self) -> usize {
        // Rough estimate based on model size
        // This is approximate - actual usage may vary
        self.info.size_mb as usize * 1024 * 1024
    }
    
    /// Validate the model works with test audio.
    /// 
    /// ## Purpose:
    /// Ensures the model loaded correctly and can process audio.
    /// Uses a short silence as test input.
    pub async fn validate(&mut self) -> Result<()> {
        tracing::debug!("Validating Whisper model...");
        
        // Create 1 second of silence as test audio
        let test_audio = vec![0.0f32; 16000];
        
        // Try to transcribe (should return empty or minimal text)
        let result = self.transcribe(&test_audio, Some("en")).await?;
        
        tracing::debug!("Model validation successful, test result: '{}'", result);
        Ok(())
    }
    
    /// Get start-of-transcription token.
    fn get_sot_token(&self) -> u32 {
        // Standard Whisper SOT token ID
        50258
    }
    
    /// Get end-of-transcription token.
    fn get_eot_token(&self) -> u32 {
        // Standard Whisper EOT token ID
        50257
    }
    
    /// Get transcription task token.
    fn get_transcribe_token(&self) -> u32 {
        // Standard Whisper transcribe task token ID
        50359
    }
    
    /// Get language token for specified language.
    fn get_language_token(&self, language: &str) -> Option<u32> {
        // Language token mapping (simplified - should be loaded from tokenizer)
        match language.to_lowercase().as_str() {
            "en" | "english" => Some(50259),
            "es" | "spanish" => Some(50262),
            "fr" | "french" => Some(50265),
            "de" | "german" => Some(50261),
            "it" | "italian" => Some(50274),
            "pt" | "portuguese" => Some(50267),
            "ru" | "russian" => Some(50263),
            "ja" | "japanese" => Some(50266),
            "ko" | "korean" => Some(50264),
            "zh" | "chinese" => Some(50260),
            _ => None,
        }
    }
    
    /// Sample token from logits with temperature.
    fn sample_token(&self, logits: &Tensor, temperature: f32) -> Result<u32> {
        let temp_tensor = Tensor::from_vec(vec![temperature], (1,), &self.device)?;
        let logits = logits.broadcast_div(&temp_tensor)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        
        // Simple greedy sampling for now - could implement more sophisticated sampling
        let token = probs.argmax_keepdim(1)?.to_scalar::<u32>()?;
        Ok(token)
    }
    
    /// Check if token sequence is repetitive.
    fn is_repetitive(&self, tokens: &[u32], new_token: u32) -> bool {
        if tokens.len() < 6 {
            return false;
        }
        
        // Check for immediate repetition
        if tokens.len() >= 3 && tokens[tokens.len() - 3..] == [new_token, new_token, new_token] {
            return true;
        }
        
        // Check for pattern repetition
        if tokens.len() >= 6 {
            let last_3 = &tokens[tokens.len() - 3..];
            let prev_3 = &tokens[tokens.len() - 6..tokens.len() - 3];
            if last_3 == prev_3 {
                return true;
            }
        }
        
        false
    }
    
    /// Decode tokens to text using the tokenizer.
    fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        // Convert tokens to text using the tokenizer
        let text = self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow!("Tokenizer decode error: {}", e))?;
            
        // Clean up the text
        let text = text.trim();
        
        // Remove common artifacts
        let cleaned_text = text
            .replace("<|startoftranscript|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|notimestamps|>", "");
        let final_text = cleaned_text.trim();
            
        Ok(final_text.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_size_parsing() {
        assert_eq!("medium".parse::<ModelSize>().unwrap(), ModelSize::Medium);
        assert_eq!("LARGE".parse::<ModelSize>().unwrap(), ModelSize::Large);
        assert!("invalid".parse::<ModelSize>().is_err());
    }
    
    #[test]
    fn test_model_info() {
        let info = ModelInfo::new(ModelSize::Medium);
        assert_eq!(info.name, "medium");
        assert_eq!(info.size_mb, 769);
        assert_eq!(info.performance, "accurate");
        assert!(!info.loaded);
    }
}