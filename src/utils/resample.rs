//! Lightweight streaming linear resampler.
//!
//! Converts arbitrary rates to/from the engine rate (48 kHz) with low CPU.
//! Suitable for real-time; quality is fine for speech enhancement side-conversion.
//! State is kept across `process` calls so block boundaries stay continuous.

/// Streaming linear interpolator between two sample rates.
#[derive(Debug, Clone)]
pub struct LinearResampler {
    from_rate: u32,
    to_rate: u32,
    /// Fractional read position in the *input* timeline (samples of from_rate).
    pos: f64,
    prev: f32,
    has_prev: bool,
}

impl LinearResampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        Self {
            from_rate: from_rate.max(1),
            to_rate: to_rate.max(1),
            pos: 0.0,
            prev: 0.0,
            has_prev: false,
        }
    }

    pub fn from_rate(&self) -> u32 {
        self.from_rate
    }

    pub fn to_rate(&self) -> u32 {
        self.to_rate
    }

    pub fn is_identity(&self) -> bool {
        self.from_rate == self.to_rate
    }

    pub fn reset(&mut self) {
        self.pos = 0.0;
        self.prev = 0.0;
        self.has_prev = false;
    }

    /// Expected output length for a given input length (may vary ±1 across blocks).
    pub fn output_len_hint(&self, input_len: usize) -> usize {
        if self.is_identity() {
            return input_len;
        }
        let ratio = self.to_rate as f64 / self.from_rate as f64;
        (input_len as f64 * ratio).round() as usize
    }

    /// Resample `input` into `output` (cleared/filled to exact needed length).
    /// Returns the number of output samples written.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> usize {
        if self.is_identity() {
            output.clear();
            output.extend_from_slice(input);
            return input.len();
        }

        if input.is_empty() {
            output.clear();
            return 0;
        }

        let step = self.from_rate as f64 / self.to_rate as f64;
        let ratio = self.to_rate as f64 / self.from_rate as f64;
        let out_len = (input.len() as f64 * ratio + 0.5) as usize;
        output.clear();
        output.reserve(out_len);

        let mut pos = self.pos;

        while output.len() < out_len {
            let idx = pos.floor() as i64;
            let frac = (pos - idx as f64) as f32;

            let (a, b) = if idx < 0 {
                if self.has_prev {
                    let next = if input.is_empty() {
                        self.prev
                    } else {
                        input[0]
                    };
                    (self.prev, next)
                } else {
                    let s = input[0];
                    (s, s)
                }
            } else if (idx as usize) + 1 < input.len() {
                (input[idx as usize], input[idx as usize + 1])
            } else if (idx as usize) < input.len() {
                let s = input[idx as usize];
                (s, s)
            } else {
                break;
            };

            output.push(a + (b - a) * frac);
            pos += step;
        }

        self.pos = pos - input.len() as f64;
        if let Some(&last) = input.last() {
            self.prev = last;
            self.has_prev = true;
        }

        output.len()
    }

    /// One-shot convenience for offline blocks.
    pub fn resample_vec(from_rate: u32, to_rate: u32, input: &[f32]) -> Vec<f32> {
        let mut r = Self::new(from_rate, to_rate);
        let mut out = Vec::new();
        r.process(input, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let mut r = LinearResampler::new(48000, 48000);
        let mut out = Vec::new();
        r.process(&[1.0, 2.0, 3.0], &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn upsample_length() {
        let mut r = LinearResampler::new(24000, 48000);
        let mut out = Vec::new();
        r.process(&[0.0; 100], &mut out);
        assert!((out.len() as i32 - 200).abs() <= 2);
    }

    #[test]
    fn downsample_length() {
        let mut r = LinearResampler::new(48000, 16000);
        let mut out = Vec::new();
        r.process(&[0.0; 480], &mut out);
        assert!((out.len() as i32 - 160).abs() <= 2);
    }
}
