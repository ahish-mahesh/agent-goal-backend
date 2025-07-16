//! # Device Detection and Management
//! 
//! Handles automatic detection and selection of compute devices (CPU/GPU) for ML inference.
//! Provides fallback mechanisms and device availability checking.

use candle_core::Device;
use std::sync::OnceLock;
use tracing::{info, warn, debug};

/// Cached best available device to avoid repeated detection
static BEST_DEVICE: OnceLock<Device> = OnceLock::new();

/// Device preferences for model inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// Automatically select the best available device
    Auto,
    /// Force CPU usage
    Cpu,
    /// Force CUDA GPU usage (will fallback to CPU if not available)
    Cuda,
    /// Force Metal GPU usage (will fallback to CPU if not available)
    Metal,
}

impl std::str::FromStr for DevicePreference {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" | "automatic" => Ok(DevicePreference::Auto),
            "cpu" => Ok(DevicePreference::Cpu),
            "cuda" | "gpu" => Ok(DevicePreference::Cuda),
            "metal" => Ok(DevicePreference::Metal),
            _ => Err(format!("Unknown device preference: {}", s)),
        }
    }
}

impl Default for DevicePreference {
    fn default() -> Self {
        DevicePreference::Auto
    }
}

/// Device detection and selection utilities
pub struct DeviceManager;

impl DeviceManager {
    /// Get the best available device based on preference
    pub fn get_device(preference: DevicePreference) -> Device {
        match preference {
            DevicePreference::Auto => Self::get_best_device(),
            DevicePreference::Cpu => Device::Cpu,
            DevicePreference::Cuda => Self::get_cuda_device().unwrap_or(Device::Cpu),
            DevicePreference::Metal => Self::get_metal_device().unwrap_or(Device::Cpu),
        }
    }
    
    /// Get the best available device (cached)
    pub fn get_best_device() -> Device {
        BEST_DEVICE.get_or_init(|| {
            Self::detect_best_device()
        }).clone()
    }
    
    /// Detect the best available device
    fn detect_best_device() -> Device {
        info!("Detecting best available compute device...");
        
        // Try CUDA first (NVIDIA GPUs)
        if let Some(cuda_device) = Self::get_cuda_device() {
            info!("Selected CUDA GPU for ML inference");
            return cuda_device;
        }
        
        // Try Metal (Apple Silicon)
        if let Some(metal_device) = Self::get_metal_device() {
            info!("Selected Metal GPU for ML inference");
            return metal_device;
        }
        
        // Fallback to CPU
        info!("Using CPU for ML inference (no GPU acceleration available)");
        Device::Cpu
    }
    
    /// Try to get a CUDA device
    fn get_cuda_device() -> Option<Device> {
        match Device::new_cuda(0) {
            Ok(device) => {
                debug!("CUDA device 0 available");
                Some(device)
            }
            Err(e) => {
                debug!("CUDA not available: {}", e);
                None
            }
        }
    }
    
    /// Try to get a Metal device
    fn get_metal_device() -> Option<Device> {
        match Device::new_metal(0) {
            Ok(device) => {
                debug!("Metal device 0 available");
                Some(device)
            }
            Err(e) => {
                debug!("Metal not available: {}", e);
                None
            }
        }
    }
    
    /// Check if CUDA is available
    pub fn is_cuda_available() -> bool {
        Self::get_cuda_device().is_some()
    }
    
    /// Check if Metal is available
    pub fn is_metal_available() -> bool {
        Self::get_metal_device().is_some()
    }
    
    /// Check if any GPU is available
    pub fn is_gpu_available() -> bool {
        Self::is_cuda_available() || Self::is_metal_available()
    }
    
    /// Get device information for logging/debugging
    pub fn get_device_info(device: &Device) -> String {
        match device {
            Device::Cpu => "CPU".to_string(),
            Device::Cuda(_) => {
                // Try to get CUDA device properties
                format!("CUDA GPU")
            }
            Device::Metal(_) => {
                // Try to get Metal device properties
                format!("Metal GPU (Apple Silicon)")
            }
        }
    }
    
    /// Get system device summary
    pub fn get_device_summary() -> DeviceSummary {
        DeviceSummary {
            cuda_available: Self::is_cuda_available(),
            metal_available: Self::is_metal_available(),
            current_device: Self::get_device_info(&Self::get_best_device()),
            gpu_available: Self::is_gpu_available(),
        }
    }
    
    /// Force re-detection of devices (clears cache)
    pub fn refresh_device_cache() {
        // Unfortunately, OnceLock doesn't have a clear method
        // This would require using a different caching mechanism
        warn!("Device cache refresh requested, but cache is immutable. Restart application to re-detect devices.");
    }
}

/// Device availability summary
#[derive(Debug, Clone)]
pub struct DeviceSummary {
    pub cuda_available: bool,
    pub metal_available: bool,
    pub gpu_available: bool,
    pub current_device: String,
}

/// Create a device based on string preference with fallback
pub fn create_device_from_string(device_str: &str) -> Device {
    match device_str.parse::<DevicePreference>() {
        Ok(preference) => DeviceManager::get_device(preference),
        Err(_) => {
            warn!("Invalid device preference '{}', using auto", device_str);
            DeviceManager::get_best_device()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_preference_parsing() {
        assert_eq!("auto".parse::<DevicePreference>().unwrap(), DevicePreference::Auto);
        assert_eq!("cpu".parse::<DevicePreference>().unwrap(), DevicePreference::Cpu);
        assert_eq!("cuda".parse::<DevicePreference>().unwrap(), DevicePreference::Cuda);
        assert_eq!("metal".parse::<DevicePreference>().unwrap(), DevicePreference::Metal);
        assert!("invalid".parse::<DevicePreference>().is_err());
    }
    
    #[test]
    fn test_device_manager_cpu_fallback() {
        // Should always work
        let device = DeviceManager::get_device(DevicePreference::Cpu);
        matches!(device, Device::Cpu);
    }
    
    #[test]
    fn test_device_detection() {
        // This will actually test device detection on the current system
        let device = DeviceManager::get_best_device();
        let info = DeviceManager::get_device_info(&device);
        assert!(!info.is_empty());
    }
}