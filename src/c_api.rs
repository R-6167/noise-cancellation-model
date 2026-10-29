//! C ABI for embedding into C/C++ hosts (audio-studio, DAWs, VoIP, etc.).
//!
//! Enabled with feature `c-api`. Header: `include/noise_cancellation.h`.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

use crate::processing::{Denoiser, DenoiserConfig, EngineKind, Strength};
use crate::{Error, FRAME_SIZE, SAMPLE_RATE};

/// Opaque handle — heap-allocated Denoiser plus owned name string.
pub struct NcHandle {
    den: Denoiser,
    name: CString,
}

#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NcStatus {
    Ok = 0,
    ErrNullPointer = -1,
    ErrInvalidArg = -2,
    ErrEngine = -3,
    ErrFeatureDisabled = -4,
    ErrOutOfMemory = -5,
    ErrUnsupported = -6,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum NcEngine {
    Rnnoise = 0,
    DeepFilterNet = 1,
}

fn ranges_overlap(
    input: *const f32,
    output: *mut f32,
    samples: usize,
) -> Option<bool> {
    let bytes = samples.checked_mul(std::mem::size_of::<f32>())?;
    let input_start = input as usize;
    let output_start = output as usize;
    let input_end = input_start.checked_add(bytes)?;
    let output_end = output_start.checked_add(bytes)?;
    Some(input_start < output_end && output_start < input_end)
}

fn map_err(e: Error) -> NcStatus {
    match e {
        Error::InvalidConfig(_) => NcStatus::ErrInvalidArg,
        Error::UnsupportedFormat(_) => NcStatus::ErrUnsupported,
        Error::Engine(_) => NcStatus::ErrEngine,
        Error::FeatureDisabled(_) => NcStatus::ErrFeatureDisabled,
        Error::Io(_) => NcStatus::ErrEngine,
    }
}

fn build_config(
    engine: NcEngine,
    model_dir: Option<PathBuf>,
    strength: f32,
    sample_rate: u32,
) -> Result<DenoiserConfig, Error> {
    if !strength.is_finite() {
        return Err(Error::InvalidConfig("strength must be finite".into()));
    }
    let strength = Strength(strength.clamp(0.0, 1.0));
    let kind = match engine {
        NcEngine::Rnnoise => {
            #[cfg(feature = "rnnoise")]
            {
                EngineKind::Rnnoise
            }
            #[cfg(not(feature = "rnnoise"))]
            {
                return Err(Error::FeatureDisabled("rnnoise"));
            }
        }
        NcEngine::DeepFilterNet => {
            #[cfg(feature = "deepfilternet")]
            {
                let dir = model_dir.unwrap_or_else(crate::utils::default_model_dir);
                EngineKind::DeepFilterNet { model_dir: dir }
            }
            #[cfg(not(feature = "deepfilternet"))]
            {
                return Err(Error::FeatureDisabled("deepfilternet"));
            }
        }
    };
    Ok(DenoiserConfig {
        engine: kind,
        strength,
        wet_dry_mix: true,
        sample_rate: if sample_rate == 0 {
            SAMPLE_RATE
        } else {
            sample_rate
        },
    })
}

/// Create a denoiser at 48 kHz host rate.
#[no_mangle]
pub unsafe extern "C" fn nc_create(
    engine: i32,
    model_dir: *const c_char,
    strength: f32,
    out: *mut *mut NcHandle,
) -> i32 {
    unsafe { nc_create_with_rate(engine, model_dir, strength, SAMPLE_RATE, out) }
}

/// Create with explicit host sample rate (Hz).
#[no_mangle]
pub unsafe extern "C" fn nc_create_with_rate(
    engine: i32,
    model_dir: *const c_char,
    strength: f32,
    sample_rate: u32,
    out: *mut *mut NcHandle,
) -> i32 {
    if out.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    unsafe { ptr::write(out, ptr::null_mut()) };

    let eng = match engine {
        0 => NcEngine::Rnnoise,
        1 => NcEngine::DeepFilterNet,
        _ => return NcStatus::ErrInvalidArg as i32,
    };

    let dir = if model_dir.is_null() {
        None
    } else {
        let s = unsafe { CStr::from_ptr(model_dir) };
        match s.to_str() {
            Ok(s) if !s.is_empty() => Some(PathBuf::from(s)),
            Ok(_) => None,
            Err(_) => return NcStatus::ErrInvalidArg as i32,
        }
    };

    let config = match build_config(eng, dir, strength, sample_rate) {
        Ok(c) => c,
        Err(e) => return map_err(e) as i32,
    };

    let den = match Denoiser::new(config) {
        Ok(d) => d,
        Err(e) => return map_err(e) as i32,
    };

    let name = CString::new(den.engine_name()).unwrap_or_default();
    let handle = Box::new(NcHandle { den, name });
    unsafe { ptr::write(out, Box::into_raw(handle)) };
    NcStatus::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn nc_destroy(handle: *mut NcHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[no_mangle]
pub unsafe extern "C" fn nc_process(
    handle: *mut NcHandle,
    input: *const f32,
    output: *mut f32,
    num_samples: usize,
) -> i32 {
    if handle.is_null() || input.is_null() || output.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    if num_samples == 0 {
        return NcStatus::Ok as i32;
    }

    let overlaps = match ranges_overlap(input, output, num_samples) {
        Some(value) => value,
        None => return NcStatus::ErrInvalidArg as i32,
    };
    let h = unsafe { &mut *handle };
    if overlaps {
        // Copy before creating the mutable output slice so partial overlap is never aliased.
        let input_copy =
            unsafe { std::slice::from_raw_parts(input, num_samples) }.to_vec();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output, num_samples) };
        match h.den.process(&input_copy, out_slice) {
            Ok(()) => NcStatus::Ok as i32,
            Err(e) => map_err(e) as i32,
        }
    } else {
        let in_slice = unsafe { std::slice::from_raw_parts(input, num_samples) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output, num_samples) };
        match h.den.process(in_slice, out_slice) {
            Ok(()) => NcStatus::Ok as i32,
            Err(e) => map_err(e) as i32,
        }
    }
}

/// Drain delayed tail samples after the final nc_process call.
#[no_mangle]
pub unsafe extern "C" fn nc_flush(
    handle: *mut NcHandle,
    output: *mut f32,
    capacity: usize,
    out_written: *mut usize,
) -> i32 {
    if handle.is_null() || out_written.is_null() || (capacity > 0 && output.is_null()) {
        return NcStatus::ErrNullPointer as i32;
    }
    let h = unsafe { &mut *handle };
    let output_slice: &mut [f32] = if capacity == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(output, capacity) }
    };
    match h.den.flush(output_slice) {
        Ok(written) => {
            unsafe { ptr::write(out_written, written) };
            NcStatus::Ok as i32
        }
        Err(e) => map_err(e) as i32,
    }
}

#[no_mangle]
pub unsafe extern "C" fn nc_reset(handle: *mut NcHandle) -> i32 {
    if handle.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    unsafe { (*handle).den.reset() };
    NcStatus::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn nc_set_strength(handle: *mut NcHandle, strength: f32) -> i32 {
    if handle.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    if !strength.is_finite() {
        return NcStatus::ErrInvalidArg as i32;
    }
    unsafe { (*handle).den.set_strength(Strength(strength)) };
    NcStatus::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn nc_get_strength(handle: *const NcHandle, out_strength: *mut f32) -> i32 {
    if handle.is_null() || out_strength.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    let s = unsafe { (*handle).den.strength().0 };
    unsafe { ptr::write(out_strength, s) };
    NcStatus::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn nc_latency_samples(
    handle: *const NcHandle,
    out_latency: *mut usize,
) -> i32 {
    if handle.is_null() || out_latency.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    let lat = unsafe { (*handle).den.latency_samples() };
    unsafe { ptr::write(out_latency, lat) };
    NcStatus::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn nc_engine_name(
    handle: *const NcHandle,
    out_name: *mut *const c_char,
) -> i32 {
    if handle.is_null() || out_name.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    let h = unsafe { &*handle };
    unsafe { ptr::write(out_name, h.name.as_ptr()) };
    NcStatus::Ok as i32
}

#[no_mangle]
pub extern "C" fn nc_sample_rate() -> u32 {
    SAMPLE_RATE
}

#[no_mangle]
pub extern "C" fn nc_frame_size() -> usize {
    FRAME_SIZE
}

#[no_mangle]
pub extern "C" fn nc_status_string(status: i32) -> *const c_char {
    let bytes: &[u8] = match status {
        0 => b"ok\0",
        -1 => b"null pointer\0",
        -2 => b"invalid argument\0",
        -3 => b"engine error\0",
        -4 => b"feature disabled\0",
        -5 => b"out of memory\0",
        -6 => b"unsupported\0",
        _ => b"unknown error\0",
    };
    bytes.as_ptr() as *const c_char
}

// ─── Multi-channel ──────────────────────────────────────────────────

/// Opaque multi-channel handle.
pub struct NcMultiHandle {
    multi: crate::processing::MultiDenoiser,
    name: CString,
}

/// Create multi-channel denoiser (1..=16 channels) at 48 kHz.
#[no_mangle]
pub unsafe extern "C" fn nc_create_multi(
    engine: i32,
    model_dir: *const c_char,
    strength: f32,
    channels: u32,
    out: *mut *mut NcMultiHandle,
) -> i32 {
    unsafe { nc_create_multi_with_rate(engine, model_dir, strength, SAMPLE_RATE, channels, out) }
}

/// Create multi-channel denoiser with host sample rate.
#[no_mangle]
pub unsafe extern "C" fn nc_create_multi_with_rate(
    engine: i32,
    model_dir: *const c_char,
    strength: f32,
    sample_rate: u32,
    channels: u32,
    out: *mut *mut NcMultiHandle,
) -> i32 {
    if out.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    unsafe { ptr::write(out, ptr::null_mut()) };

    let eng = match engine {
        0 => NcEngine::Rnnoise,
        1 => NcEngine::DeepFilterNet,
        _ => return NcStatus::ErrInvalidArg as i32,
    };

    let dir = if model_dir.is_null() {
        None
    } else {
        let s = unsafe { CStr::from_ptr(model_dir) };
        match s.to_str() {
            Ok(s) if !s.is_empty() => Some(PathBuf::from(s)),
            Ok(_) => None,
            Err(_) => return NcStatus::ErrInvalidArg as i32,
        }
    };

    let config = match build_config(eng, dir, strength, sample_rate) {
        Ok(c) => c,
        Err(e) => return map_err(e) as i32,
    };

    let multi = match crate::processing::MultiDenoiser::new(channels as usize, config) {
        Ok(m) => m,
        Err(e) => return map_err(e) as i32,
    };

    let name = CString::new(multi.engine_name()).unwrap_or_default();
    let handle = Box::new(NcMultiHandle { multi, name });
    unsafe { ptr::write(out, Box::into_raw(handle)) };
    NcStatus::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn nc_destroy_multi(handle: *mut NcMultiHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Process interleaved multi-channel audio (frames * channels samples).
#[no_mangle]
pub unsafe extern "C" fn nc_process_interleaved(
    handle: *mut NcMultiHandle,
    input: *const f32,
    output: *mut f32,
    frames: usize,
) -> i32 {
    if handle.is_null() || input.is_null() || output.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    let h = unsafe { &mut *handle };
    let ch = h.multi.channels();
    let n = match frames.checked_mul(ch) {
        Some(value) => value,
        None => return NcStatus::ErrInvalidArg as i32,
    };
    if n == 0 {
        return NcStatus::Ok as i32;
    }
    let overlaps = match ranges_overlap(input, output, n) {
        Some(value) => value,
        None => return NcStatus::ErrInvalidArg as i32,
    };

    if overlaps {
        let input_copy = unsafe { std::slice::from_raw_parts(input, n) }.to_vec();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output, n) };
        match h.multi.process_interleaved(&input_copy, out_slice, frames) {
            Ok(()) => NcStatus::Ok as i32,
            Err(e) => map_err(e) as i32,
        }
    } else {
        let in_slice = unsafe { std::slice::from_raw_parts(input, n) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output, n) };
        match h.multi.process_interleaved(in_slice, out_slice, frames) {
            Ok(()) => NcStatus::Ok as i32,
            Err(e) => map_err(e) as i32,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn nc_multi_channels(handle: *const NcMultiHandle, out: *mut u32) -> i32 {
    if handle.is_null() || out.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    unsafe { ptr::write(out, (*handle).multi.channels() as u32) };
    NcStatus::Ok as i32
}

/// Drain delayed tail frames from an interleaved multi-channel stream.
#[no_mangle]
pub unsafe extern "C" fn nc_flush_interleaved(
    handle: *mut NcMultiHandle,
    output: *mut f32,
    capacity_frames: usize,
    out_frames: *mut usize,
) -> i32 {
    if handle.is_null() || out_frames.is_null() || (capacity_frames > 0 && output.is_null()) {
        return NcStatus::ErrNullPointer as i32;
    }
    let h = unsafe { &mut *handle };
    let samples = match capacity_frames.checked_mul(h.multi.channels()) {
        Some(n) => n,
        None => return NcStatus::ErrInvalidArg as i32,
    };
    let output_slice: &mut [f32] = if samples == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(output, samples) }
    };
    match h.multi.flush_interleaved(output_slice) {
        Ok(written) => {
            unsafe { ptr::write(out_frames, written) };
            NcStatus::Ok as i32
        }
        Err(e) => map_err(e) as i32,
    }
}

#[no_mangle]
pub unsafe extern "C" fn nc_multi_reset(handle: *mut NcMultiHandle) -> i32 {
    if handle.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    unsafe { (*handle).multi.reset() };
    NcStatus::Ok as i32
}

#[no_mangle]
pub unsafe extern "C" fn nc_multi_set_strength(handle: *mut NcMultiHandle, strength: f32) -> i32 {
    if handle.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    unsafe { (*handle).multi.set_strength(Strength(strength)) };
    NcStatus::Ok as i32
}


#[cfg(test)]
mod overlap_tests {
    use super::*;

    #[test]
    fn single_channel_partial_overlap_is_safe() {
        let mut handle: *mut NcHandle = ptr::null_mut();
        let status = unsafe {
            nc_create(NcEngine::Rnnoise as i32, ptr::null(), 0.85, &mut handle)
        };
        assert_eq!(status, NcStatus::Ok as i32);
        assert!(!handle.is_null());

        let mut buffer = [0.1f32; 129];
        let status = unsafe {
            nc_process(handle, buffer.as_ptr(), buffer.as_mut_ptr().add(1), 128)
        };
        assert_eq!(status, NcStatus::Ok as i32);
        assert!(buffer[1..].iter().all(|sample| sample.is_finite()));
        unsafe { nc_destroy(handle) };
    }

    #[test]
    fn interleaved_partial_overlap_is_safe() {
        let mut handle: *mut NcMultiHandle = ptr::null_mut();
        let status = unsafe {
            nc_create_multi(
                NcEngine::Rnnoise as i32,
                ptr::null(),
                0.85,
                2,
                &mut handle,
            )
        };
        assert_eq!(status, NcStatus::Ok as i32);
        assert!(!handle.is_null());

        let mut buffer = [0.1f32; 65];
        let status = unsafe {
            nc_process_interleaved(handle, buffer.as_ptr(), buffer.as_mut_ptr().add(1), 32)
        };
        assert_eq!(status, NcStatus::Ok as i32);
        assert!(buffer[1..].iter().all(|sample| sample.is_finite()));
        unsafe { nc_destroy_multi(handle) };
    }
}
