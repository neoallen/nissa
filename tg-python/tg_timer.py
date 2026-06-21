#!/usr/bin/env python3
"""
tg-timer: Python port of 'tg' mechanical watch timing software.

Reads watch tick audio and computes daily rate deviation (s/d),
beat error (ms), and amplitude (°).

Algorithm (ported from C by Marcello Mamino / vacaboja):
  1. Audio → HPF 3kHz → rectify → LPF 3kHz → envelope
  2. Autocorrelation (FFT-based) → period estimation
  3. Phase-aligned waveform folding → tic/toc detection
  4. Beat error, amplitude, and rate computation

Usage:
    uv run tg-python/tg_timer.py <audio_file> [bph] [la] [--json]

Examples:
    uv run tg-python/tg_timer.py watch_tick.wav
    uv run tg-python/tg_timer.py input_example.m4a 21600 52
    uv run tg-python/tg_timer.py input_example.m4a --json
"""

import json
import wave
import subprocess
import tempfile
import os
import sys
import math
import numpy as np
from scipy import fft
from scipy.signal import lfilter

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100          # Processing sample rate (Hz)
FILTER_CUTOFF = 3000         # HPF/LPF cutoff (Hz)
DEFAULT_BPH = 21600          # Default beats per hour
DEFAULT_LA = 52.0            # Default lift angle (degrees)
CHUNK_SEC = 4.0              # Processing chunk size (seconds)
OVERLAP = 0.5                # Chunk overlap fraction
PRESET_BPHS = [12000, 14400, 17280, 18000, 19800, 21600,
               25200, 28800, 36000, 43200, 72000]


# ════════════════════════════════════════════════════════════════════════════
# IIR Biquad Filters (transposed direct-form II, matching C code)
# ════════════════════════════════════════════════════════════════════════════

def _make_hp(freq: float, sr: int) -> tuple:
    K = math.tan(math.pi * freq / sr)
    n = 1.0 / (1 + K * math.sqrt(2) + K * K)
    b = [1.0*n, -2.0*n, 1.0*n]
    a = [1.0, 2.0*(K*K-1)*n, (1 - K*math.sqrt(2) + K*K)*n]
    return b, a

def _make_lp(freq: float, sr: int) -> tuple:
    K = math.tan(math.pi * freq / sr)
    n = 1.0 / (1 + K * math.sqrt(2) + K * K)
    b = [K*K*n, 2.0*K*K*n, K*K*n]
    a = [1.0, 2.0*(K*K-1)*n, (1 - K*math.sqrt(2) + K*K)*n]
    return b, a


# ════════════════════════════════════════════════════════════════════════════
# Quickselect (descending partition, from C)
# ════════════════════════════════════════════════════════════════════════════

def _quickselect(arr: np.ndarray, k: int):
    """In-place: arr[0..k-1] >= arr[k] >= arr[k+1..] (descending)."""
    l, r = 0, len(arr) - 1
    while True:
        if l == r:
            return
        if r - l == 1:
            if arr[l] < arr[r]:
                arr[l], arr[r] = arr[r], arr[l]
            return
        m = (l + r) // 2
        if arr[l] < arr[r]:
            p = l if arr[m] <= arr[l] else (r if arr[m] >= arr[r] else m)
        else:
            p = r if arr[m] <= arr[r] else (l if arr[m] >= arr[l] else m)
        pv = arr[p]; arr[p] = arr[r]; p = l
        for i in range(l, r):
            if arr[i] > pv:
                arr[i], arr[p] = arr[p], arr[i]; p += 1
        arr[r], arr[p] = arr[p], pv
        if k == p:
            return
        (r, l) = (p - 1, l) if k < p else (r, p + 1)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _vmax(arr: np.ndarray, a: int, b: int) -> tuple:
    idx = a + int(np.argmax(arr[a:b]))
    return float(arr[idx]), idx


def _tmean(arr: np.ndarray) -> float:
    """Trimmed mean: average of lower 4 quintiles (exclude top 20%)."""
    n = len(arr)
    if n <= 5:
        return (float(np.sum(arr)) - float(np.max(arr))) / (n - 1)
    if n <= 10:
        s = np.sort(arr)[::-1]
        return float(np.mean(s[2:]))
    k = (n + 4) // 5
    cp = arr.copy().astype(np.float64)
    _quickselect(cp, k)
    return float(np.mean(cp[k:]))


def _bootstrap_ci(values: np.ndarray, n_resamples: int = 5000,
                  ci: float = 95.0) -> tuple:
    """Bootstrap confidence interval for the median. Returns (lo, hi)."""
    if len(values) < 3:
        return None, None
    rng = np.random.default_rng()
    medians = np.empty(n_resamples)
    for i in range(n_resamples):
        medians[i] = float(np.median(
            rng.choice(values, size=len(values), replace=True)))
    alpha = (100 - ci) / 2
    lo = float(np.percentile(medians, alpha))
    hi = float(np.percentile(medians, 100 - alpha))
    return lo, hi


# ════════════════════════════════════════════════════════════════════════════
# Peak Detector (from C — detects a single peak with valley constraints)
# ════════════════════════════════════════════════════════════════════════════

def _peak_detector(buff: np.ndarray, a: int, b: int) -> int:
    """Detect a single peak in buff[a:b). Returns index or -1."""
    n = b - a
    if n <= 0:
        return -1
    max_v, i_max = _vmax(buff, a, b)
    if max_v <= 0:
        return -1

    v = buff[a:b].copy()
    _quickselect(v, n // 2)
    med = float(v[n // 2])

    for i in range(a + 1, i_max):
        if buff[i] <= med:
            break
    else:
        return -1
    for i in range(i_max + 1, b):
        if buff[i] <= med:
            break
    else:
        return -1

    cnt = 0; down = 1
    for i in range(a + 1, b):
        if buff[i] > (max_v + med) / 2:
            cnt += down; down = 0
        if buff[i] < med:
            down = 1
    return -1 if cnt > 20 else i_max


# ════════════════════════════════════════════════════════════════════════════
# Noise Suppressor (from C — removes transients exceeding 2x median energy)
# ════════════════════════════════════════════════════════════════════════════

def _noise_suppressor(samples: np.ndarray, sr: int):
    n = len(samples)
    window = sr // 50
    sq = samples.astype(np.float64) ** 2
    b = np.empty(n - window + 1)
    r_av = np.sum(sq[:window])
    for i in range(n - window + 1):
        b[i] = r_av
        if i + window < n:
            r_av += sq[i + window] - sq[i]
    step = sr // 2
    a_vals = [np.max(b[i:i+step]) for i in range(0, n - window + 1, step)]
    a_arr = np.array(a_vals)
    _quickselect(a_arr, len(a_arr) // 2)
    k = float(a_arr[len(a_arr) // 2])
    for i in range(n):
        j = max(0, min(i - window // 2, n - window))
        if b[j] > 2 * k:
            samples[i] = 0.0


# ════════════════════════════════════════════════════════════════════════════
# Smooth (exponential running max, from C)
# ════════════════════════════════════════════════════════════════════════════

def _smooth(inp: np.ndarray, window: int) -> np.ndarray:
    n = len(inp)
    size = n - window
    k = 1.0 - 1.0 / window
    r_av = u = 0.0
    for i in range(window):
        u = u * k
        if inp[i] > u: u = inp[i]
        r_av += u
    out = np.empty(size)
    w = 0.0
    for i in range(size):
        out[i] = r_av
        u *= k; w *= k
        if inp[i + window] > u: u = inp[i + window]
        if inp[i] > w: w = inp[i]
        r_av += u - w
    return out


# ════════════════════════════════════════════════════════════════════════════
# Core Processing Functions (from C algo.c)
# ════════════════════════════════════════════════════════════════════════════

def prepare_data(samples: np.ndarray, sr: int, run_ns: bool = True):
    """
    HPF → (noise suppressor) → rectify → LPF → DC removal → window → FFT→|FFT|²→IFFT.
    Returns (autocorrelation, processed_samples).
    """
    n = len(samples)
    hp_b, hp_a = _make_hp(FILTER_CUTOFF, sr)
    s = lfilter(hp_b, hp_a, samples)
    if run_ns:
        _noise_suppressor(s, sr)
    s = np.abs(s)
    lp_b, lp_a = _make_lp(FILTER_CUTOFF, sr)
    s = lfilter(lp_b, lp_a, s)
    s = s - np.mean(s)
    win_len = sr // 10
    for i in range(win_len):
        k = (1 - math.cos(i * math.pi / win_len)) / 2
        s[i] *= k; s[n - 1 - i] *= k
    zp = np.zeros(2 * n, dtype=np.float64)
    zp[:n] = s
    f = fft.rfft(zp)
    ac = fft.irfft(f * np.conj(f), 2 * n).astype(np.float64)
    return ac, s


def _parabolic_peak(ac: np.ndarray, i: int) -> float:
    """Sub-sample peak position via parabolic interpolation."""
    if i <= 0 or i >= len(ac) - 1:
        return float(i)
    y0, y1, y2 = float(ac[i-1]), float(ac[i]), float(ac[i+1])
    if y1 <= 0:
        return float(i)
    denom = 2.0 * (y0 + y2 - 2.0 * y1)
    if abs(denom) < 1e-15:
        return float(i)
    return i + (y0 - y2) / denom


def _find_best_peak(ac: np.ndarray, a: int, b: int, expected: float) -> float | None:
    """Find the best peak near `expected` within [a, b).

    Enumerates all local maxima, scores them by distance from expected
    (primary) and amplitude (secondary tiebreaker). Uses parabolic
    interpolation for sub-sample precision.
    Returns refined peak position in samples, or None.
    """
    if a < 0 or b > len(ac):
        return None

    # Find all local maxima in range
    candidates = []
    for i in range(max(1, a + 1), min(len(ac) - 1, b - 1)):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0:
            candidates.append(i)

    if not candidates:
        return None

    # Heavily weight closeness to expected over amplitude
    max_amp = max(float(ac[i]) for i in candidates)
    best = None
    best_dist = float('inf')
    best_amp = 0.0

    for i in candidates:
        amp = float(ac[i])
        amp_ratio = amp / max_amp if max_amp > 0 else 0
        if amp_ratio < 0.05:
            continue
        dist = abs(i - expected)
        # Prefer closer peaks; tiebreak by amplitude
        if dist < best_dist - 0.5 or (abs(dist - best_dist) <= 0.5 and amp > best_amp):
            best_dist = dist
            best_amp = amp
            best = i

    if best is None:
        return None

    # Parabolic refinement
    refined = _parabolic_peak(ac, best)
    return refined


def compute_period(ac: np.ndarray, sr: int, bph: int = 0) -> float | None:
    """
    Estimate period (in samples) from autocorrelation.
    Returns None on failure.
    """
    n = len(ac)

    if bph:
        expected = 7200.0 * sr / bph
        # Tight search: ±0.7% of expected (not ±2% like before)
        # Prevents picking up shifted noise peaks
        margin = int(expected * 0.007)
        estimate = _find_best_peak(ac, int(expected - margin),
                                   int(expected + margin), expected)
        if estimate is None:
            # Fallback: wider search
            estimate = _find_best_peak(ac, int(expected - sr // 50),
                                       int(expected + sr // 50), expected)
            if estimate is None:
                return None
    else:
        # Auto-detect: find the strongest peak in plausible range
        _, first_est = _vmax(ac, sr // 12, sr)
        expected = float(first_est)
        margin = sr // 12
        estimate = _find_best_peak(ac,
                                   max(sr // 12, first_est - margin),
                                   first_est + margin, expected)
        if estimate is None:
            return None
        # Check for sub-harmonics
        f_est = int(round(estimate))
        for fct in range(2, 100):
            if f_est // fct <= sr // 12:
                break
            sub_expected = float(f_est) / fct
            a, b = f_est // fct - sr // 50, f_est // fct + sr // 50
            ne = _find_best_peak(ac, a, b, sub_expected)
            if ne is not None and ac[int(round(ne))] > 0.9 * ac[f_est]:
                estimate = ne
        # Check 3/2 harmonic rejection
        a, b = int(estimate * 1.5 - sr / 50), int(estimate * 1.5 + sr / 50)
        if a >= 0 and b < n and np.max(ac[a:b+1]) < 0.2 * ac[int(round(estimate))]:
            a, b = f_est * 2 - sr // 50, f_est * 2 + sr // 50
            ne = _find_best_peak(ac, a, b, float(f_est * 2))
            if ne is not None:
                estimate = ne

    # Return the initial parabolic-sub-sample estimate without refinement.
    # Multi-cycle refinement is skipped because it amplifies noise in
    # low-SNR recordings, pulling the period away from the true value.
    return estimate


def compute_phase(samples: np.ndarray, period: float) -> float:
    """Phase from folded waveform (Fourier fundamental). Returns phase in samples."""
    p = int(round(period))
    wf = np.zeros(p)
    cnt = np.zeros(p, dtype=int)
    for i in range(len(samples)):
        idx = int(round(i % period))
        if idx < p:
            wf[idx] += samples[i]; cnt[idx] += 1
    wf = np.divide(wf, cnt, out=np.zeros_like(wf), where=cnt > 0)
    x = y = 0.0
    for i in range(p):
        a = i * 2 * math.pi / p
        x += wf[i] * math.cos(a); y += wf[i] * math.sin(a)
    return period * (math.pi + math.atan2(y, x)) / (2 * math.pi)


def compute_waveform(samples: np.ndarray, period: float, phase: float,
                     wf_size: int) -> np.ndarray:
    """Build folded waveform using trimmed-mean folding. Returns waveform."""
    n = len(samples)
    wf = np.zeros(wf_size)
    for i in range(wf_size):
        k = (i + phase) % wf_size
        vals = []
        j = 0
        while True:
            idx = int(round(k + j * wf_size))
            if idx >= n:
                break
            vals.append(samples[idx]); j += 1
        if vals:
            wf[i] = _tmean(np.array(vals))
    step = max(1, wf_size // 100)
    dec = wf[::step].copy()
    _quickselect(dec, len(dec) // 2)
    wf -= dec[len(dec) // 2]
    return wf


def detect_beat_error(waveform: np.ndarray, period: float,
                       sr: int) -> dict:
    """
    Detect tic/toc and compute beat error from folded waveform.
    Returns {'tic', 'toc', 'be_samples', 'be_ms'} or None.
    """
    wf_size = len(waveform)
    half = int(round(period / 2))
    margin = sr // 100

    # Find the two strongest peaks separated by ~period/2
    peaks = []
    for i in range(1, wf_size - 1):
        if waveform[i] > waveform[i-1] and waveform[i] > waveform[i+1]:
            peaks.append((i, waveform[i]))
    if len(peaks) < 2:
        return None
    peaks.sort(key=lambda x: -x[1])
    top = peaks[:min(30, len(peaks))]

    # Find best pair near period/2 separation
    best = None; best_err = float('inf')
    for i, (p1, v1) in enumerate(top):
        for j, (p2, v2) in enumerate(top):
            if i >= j: continue
            gap = abs(p1 - p2)
            err = abs(gap - half)
            if err < best_err and gap > half * 0.5:
                best_err = err; best = (p1, p2, gap)

    if best is None or best_err > margin:
        return None
    p1, p2, gap = best
    be = half - gap  # beat error in samples
    return {'tic': p1, 'toc': p2, 'be_samples': abs(be),
            'be_ms': abs(be) * 1000 / sr}


def compute_rate(period: float, sr: int, bph: int) -> float:
    """Daily rate deviation in s/d. Positive = fast, negative = slow."""
    if bph == 0:
        return 0.0  # auto-detect mode: no target to compare against
    return (7200 / (bph * period / sr) - 1) * 24 * 3600


def compute_amplitude(waveform: np.ndarray, period: float, sr: int,
                      tic: int, toc: int, la: float) -> float | None:
    """Balance wheel amplitude in degrees. Returns None if unreliable."""
    wf_size = len(waveform)
    window = sr // 1000
    if wf_size < window * 2:
        return None
    wf_ext = np.concatenate([waveform, waveform[:window]])
    swf = _smooth(wf_ext, window)[:wf_size]

    glob_max = float(np.max(swf))
    if glob_max <= 0:
        return None

    # Max around tic and toc
    noise_max = 0.0
    for k in range(2):
        pos = tic if k else toc
        start = int(round((pos + period / 8) % wf_size))
        for _ in range(int(period / 8)):
            idx = start % wf_size
            if swf[idx] > noise_max: noise_max = swf[idx]
            start += 1

    threshold = max(0.01 * glob_max, 1.4 * noise_max)
    result = None

    while threshold < 0.3 * glob_max:
        pulses = []
        for k in range(2):
            pos = tic if k else toc
            start = int(round((pos + 7 * period / 8) % wf_size))
            i = 0; pulse = -1.0; max_p = 0.0
            while i < period / 8:
                if swf[start % wf_size] > threshold:
                    break
                i += 1; start += 1
            while i < period / 8:
                x = swf[start % wf_size]
                if x <= max_p:
                    break
                max_p = x; i += 1; start += 1; pulse += 1
            if i < period / 8 and pulse >= 0:
                pulses.append(pulse)
            else:
                break
        if len(pulses) == 2:
            a1 = 0.5 / math.sin(math.pi * pulses[0] / period)
            a2 = 0.5 / math.sin(math.pi * pulses[1] / period)
            amp1, amp2 = la * a1, la * a2
            if (100 < amp1 < 360 and 100 < amp2 < 360
                    and abs(amp1 - amp2) < 90):
                result = (a1 + a2) / 2
                break
        threshold *= 1.15
    return round(result * la, 1) if result else None


# ════════════════════════════════════════════════════════════════════════════
# Full Processing Pipeline
# ════════════════════════════════════════════════════════════════════════════

def process_chunk(samples: np.ndarray, sr: int, bph: int = DEFAULT_BPH,
                  la: float = DEFAULT_LA) -> dict | None:
    """Process one chunk of audio. Returns result dict or None."""
    n = len(samples)
    if n < sr:
        return None

    # Note: noise_suppressor is skipped (run_ns=False).
    # For low-SNR recordings (phone mic, room ambient),
    # the suppressor can incorrectly zero out tick events.
    try:
        ac, proc = prepare_data(samples, sr, run_ns=False)
    except Exception:
        return None

    period = compute_period(ac, sr, bph)
    if period is None or period >= sr / 2 or period < sr / 50:
        return None

    phase = compute_phase(proc, period / 2)
    wf_size = int(math.ceil(period))
    wf = compute_waveform(proc, period, phase, wf_size)

    be_info = detect_beat_error(wf, period, sr)

    amp = None
    if be_info:
        amp = compute_amplitude(wf, period, sr,
                                be_info['tic'], be_info['toc'], la)

    rate = compute_rate(period, sr, bph)
    guessed_bph = round(7200 / (period / sr))

    return {
        'period': period,
        'period_ms': period / sr * 1000,
        'bph_measured': guessed_bph,
        'rate_sd': rate,
        'beat_error_ms': be_info['be_ms'] if be_info else None,
        'amplitude_deg': amp,
    }


# ════════════════════════════════════════════════════════════════════════════
# Audio I/O
# ════════════════════════════════════════════════════════════════════════════

def read_audio(path: str) -> tuple:
    """Read audio at native sample rate, return (data, sample_rate).

    For stereo files, averages L+R channels. Never resamples.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            nf = wf.getnframes()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(nf)
        dtype = 'int16' if sw == 2 else 'int32'
        data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        data /= np.iinfo(np.int16).max if sw == 2 else np.iinfo(np.int32).max
        return data, sr
    else:
        # ffmpeg: detect native rate from file header, decode at that rate
        # First probe to get native sample rate
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', path],
            capture_output=True, text=True)
        try:
            info = json.loads(probe.stdout)
            native_sr = int(info['streams'][0]['sample_rate'])
        except (KeyError, IndexError, json.JSONDecodeError):
            native_sr = 48000  # fallback

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(['ffmpeg', '-y', '-i', path,
                           '-ac', '2', '-ar', str(native_sr),
                           '-f', 'wav', '-sample_fmt', 's16', tmp_path],
                          capture_output=True, check=True)
            return read_audio(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════════════
# Analysis Pipeline
# ════════════════════════════════════════════════════════════════════════════

def analyze_audio(samples: np.ndarray, sr: int, bph: int, la: float,
                  source_label: str = "") -> dict | None:
    """Run full analysis pipeline on raw audio samples.

    Slices audio into overlapping chunks, processes each, and aggregates
    with IQR outlier rejection. Returns result_obj dict or None.
    """
    n_total_samples = len(samples)
    dur = n_total_samples / sr
    chunk_n = int(sr * CHUNK_SEC)
    step = int(chunk_n * (1 - OVERLAP))
    results = []

    for start in range(0, n_total_samples - chunk_n + 1, step):
        chunk = samples[start:start + chunk_n]
        r = process_chunk(chunk, sr, bph, la)
        if r:
            results.append(r)

    if not results:
        return None

    n_total = len(results)
    periods = np.array([r['period_ms'] for r in results])
    rates = np.array([r['rate_sd'] for r in results])
    bes = np.array([r['beat_error_ms'] for r in results
                    if r['beat_error_ms'] is not None])
    amps = np.array([r['amplitude_deg'] for r in results
                     if r['amplitude_deg'] is not None])

    q1_p, q3_p = float(np.percentile(periods, 25)), float(np.percentile(periods, 75))
    iqr_p = q3_p - q1_p
    mask_p = (periods >= q1_p - 1.5 * iqr_p) & (periods <= q3_p + 1.5 * iqr_p)

    q1_r, q3_r = float(np.percentile(rates, 25)), float(np.percentile(rates, 75))
    iqr_r = q3_r - q1_r
    mask_r = (rates >= q1_r - 1.5 * iqr_r) & (rates <= q3_r + 1.5 * iqr_r)

    mask = mask_p & mask_r

    clean_periods = periods[mask]
    clean_rates = rates[mask]
    clean_bes = np.array([results[i]['beat_error_ms']
                          for i in range(len(results))
                          if mask[i] and results[i]['beat_error_ms'] is not None])
    n_accepted = int(np.sum(mask))

    med_period = float(np.median(clean_periods)) if len(clean_periods) else float(np.median(periods))
    mad = float(np.median(np.abs(periods - med_period)))
    avg_rate = float(np.median(clean_rates)) if len(clean_rates) else float(np.median(rates))
    rate_ci_lo, rate_ci_hi = _bootstrap_ci(clean_rates) if len(clean_rates) >= 3 else (None, None)
    avg_be = (float(np.median(clean_bes)) if len(clean_bes)
              else (float(np.median(bes)) if len(bes) else None))
    avg_amp = float(np.median(amps)) if len(amps) else 0
    bph_meas = round(7200 / (med_period / 1000))

    chunks_out = []
    for i, r in enumerate(results):
        chunks_out.append({
            'period_ms': round(r['period_ms'], 1),
            'rate_sd': round(r['rate_sd'], 1),
            'beat_error_ms': (round(r['beat_error_ms'], 1)
                              if r['beat_error_ms'] is not None else None),
            'amplitude_deg': r['amplitude_deg'],
            'accepted': bool(mask[i]),
        })

    result_obj = {
        'input': {
            'file': source_label or "audio_input",
            'duration_s': round(dur, 1),
            'sample_rate': sr,
            'bph_target': bph,
            'lift_angle': la,
        },
        'summary': {
            'chunks_total': n_total,
            'chunks_accepted': n_accepted,
            'period_ms': round(med_period, 1),
            'period_mad_ms': round(mad, 1),
            'bph_measured': bph_meas,
            'rate_sd': round(avg_rate, 1),
            'rate_ci_95': (
                [round(rate_ci_lo, 1), round(rate_ci_hi, 1)]
                if rate_ci_lo is not None else None
            ),
            'beat_error_ms': round(avg_be, 2) if avg_be is not None else None,
            'amplitude_deg': round(avg_amp, 0) if avg_amp > 0 else None,
        },
        'chunks': chunks_out,
    }
    return result_obj


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0 if sys.argv[1:2] == ['--help'] else 1)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    use_json = '--json' in sys.argv

    if not args:
        print(__doc__)
        sys.exit(1)

    path = args[0]
    bph = int(args[1]) if len(args) > 1 else DEFAULT_BPH
    la = float(args[2]) if len(args) > 2 else DEFAULT_LA

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    def log(msg):
        if use_json:
            sys.stderr.write(msg + '\n')
        else:
            print(msg)

    log(f"Reading: {path}")
    samples, sr = read_audio(path)
    if samples is None or len(samples) == 0:
        log("ERROR: Could not read audio file")
        sys.exit(1)

    dur = len(samples) / sr
    log(f"Duration: {dur:.1f}s @ {sr} Hz")
    log(f"BPH target: {bph}, Lift angle: {la}°")
    log("")

    result_obj = analyze_audio(samples, sr, bph, la, source_label=path)

    if result_obj is None:
        log("No valid measurements. Try:")
        log("  - Different BPH (e.g. 18000, 25200, 28800)")
        log("  - Check that the audio contains clear watch tick sounds")
        log("  - Use a contact microphone for better SNR")
        sys.exit(1)

    summ = result_obj['summary']
    chunks_out = result_obj['chunks']
    n_total = summ['chunks_total']
    n_accepted = summ['chunks_accepted']

    if use_json:
        json.dump(result_obj, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write('\n')
        return

    SEP = "=" * 55
    print(SEP)
    print("  TG — Watch Timing Results")
    print(SEP)
    print(f"  Chunks:            {n_accepted} accepted / {n_total} total")
    print(f"  Period:            {summ['period_ms']:.1f} ms  (±{summ['period_mad_ms']:.1f} MAD)")
    print(f"  BPH (target):      {bph}")
    print(f"  BPH (measured):    {summ['bph_measured']}")
    print(f"  Rate:              {summ['rate_sd']:+.1f} s/d"
          + (f"  (95% CI: {summ['rate_ci_95'][0]:+.1f}..{summ['rate_ci_95'][1]:+.1f})"
             if summ.get('rate_ci_95') else ""))
    if summ['beat_error_ms'] is not None:
        print(f"  Beat Error:        {summ['beat_error_ms']:.2f} ms")
    else:
        print(f"  Beat Error:        ---")
    if summ['amplitude_deg'] is not None:
        print(f"  Amplitude:         {summ['amplitude_deg']:.0f}°")
    else:
        print(f"  Amplitude:         ---")
    print()

    results = chunks_out  # chunks_out is already a list of dicts

    def chunk_str(val, fmt, rejected):
        s = f"{val:{fmt}}"
        return s + " *" if rejected else s

    if n_accepted < n_total:
        mask = [c['accepted'] for c in chunks_out]
        periods_str = ", ".join(chunk_str(r['period_ms'], ".1f", not mask[i])
                                for i, r in enumerate(chunks_out))
        rates_str = ", ".join(chunk_str(r['rate_sd'], "+.1f", not mask[i])
                              for i, r in enumerate(chunks_out))
        be_parts = []
        for i, r in enumerate(chunks_out):
            if r['beat_error_ms'] is not None:
                be_parts.append(f"{r['beat_error_ms']:.1f}" + (" *" if not mask[i] else ""))
            else:
                be_parts.append("---" + (" *" if not mask[i] else ""))
        bes_str = ", ".join(be_parts)
        print(f"  Chunk periods (ms): {periods_str}")
        print(f"  Chunk rates (s/d):  {rates_str}")
        print(f"  Chunk beat err(ms): {bes_str}")
        print("  (* = rejected by IQR filter)")
    else:
        periods_str = ", ".join(f"{r['period_ms']:.1f}" for r in chunks_out)
        rates_str = ", ".join(f"{r['rate_sd']:+.1f}" for r in chunks_out)
        bes_str = ", ".join(f"{r['beat_error_ms']:.1f}" if r['beat_error_ms'] is not None else "---"
                            for r in chunks_out)
        print(f"  Chunk periods (ms): {periods_str}")
        print(f"  Chunk rates (s/d):  {rates_str}")
        print(f"  Chunk beat err(ms): {bes_str}")
    print(SEP)


if __name__ == '__main__':
    main()
