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
) -> Result<DenoiserConfig, Error> {
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
    })
}

/// Create a denoiser. See `noise_cancellation.h`.
#[no_mangle]
pub unsafe extern "C" fn nc_create(
    engine: i32,
    model_dir: *const c_char,
    strength: f32,
    out: *mut *mut NcHandle,
) -> i32 {
    if out.is_null() {
        return NcStatus::ErrNullPointer as i32;
    }
    ptr::write(out, ptr::null_mut());

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

    let config = match build_config(eng, dir, strength) {
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

/// Destroy a handle. Safe with NULL.
#[no_mangle]
pub unsafe extern "C" fn nc_destroy(handle: *mut NcHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Process mono f32 @ 48 kHz. input/output may alias.
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

    let h = unsafe { &mut *handle };
    let in_slice = unsafe { std::slice::from_raw_parts(input, num_samples) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(output, num_samples) };

    if std::ptr::eq(input, output as *const f32) {
        let mut tmp = in_slice.to_vec();
        match h.den.process(in_slice, &mut tmp) {
            Ok(()) => {
                out_slice.copy_from_slice(&tmp);
                NcStatus::Ok as i32
            }
            Err(e) => map_err(e) as i32,
        }
    } else {
        match h.den.process(in_slice, out_slice) {
            Ok(()) => NcStatus::Ok as i32,
            Err(e) => map_err(e) as i32,
        }
    }
}

/// Reset stream state.
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
