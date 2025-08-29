from flask import Flask, request, jsonify, send_file
import os
import io
import numpy as np
import soundfile as sf
import librosa

app = Flask(__name__)

# ----------------------------- #
# Helpers                       #
# ----------------------------- #

def parse_bool(v, default=False):
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def parse_float(v, default):
    try:
        return float(v)
    except Exception:
        return default

def parse_int(v, default):
    try:
        return int(v)
    except Exception:
        return default

def normalize_to_dbfs(y, target_dbfs=-18.0, eps=1e-12):
    """
    Loudness-normalize to target dBFS by matching RMS.
    """
    rms = np.sqrt(np.mean(np.square(y)) + eps)
    ref_amp = 10.0 ** (target_dbfs / 20.0)
    if rms < eps:
        return y
    gain = ref_amp / rms
    y2 = y * gain
    # prevent accidental clipping
    mx = np.max(np.abs(y2)) + eps
    if mx > 1.0:
        y2 = y2 / mx
    return y2

def band_limits_to_bin(bass_hz, sr, n_fft):
    """
    Return bin index that corresponds to bass_hz cutoff.
    """
    if bass_hz <= 0:
        return 0
    return int(np.clip(np.round(bass_hz * n_fft / sr), 0, n_fft // 2))

def xf_crossfade(a, b, xf_ms, sr):
    """
    Crossfade tail of a into head of b with linear window.
    """
    xf = int(sr * (xf_ms / 1000.0))
    if xf <= 0:
        return np.concatenate([a, b], axis=0)
    n_a = len(a)
    n_b = len(b)
    xf = min(xf, n_a, n_b)
    if xf <= 0:
        return np.concatenate([a, b], axis=0)

    fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
    fade_in  = 1.0 - fade_out
    x_tail = a[-xf:] * fade_out
    y_head = b[:xf]  * fade_in
    mid = x_tail + y_head
    return np.concatenate([a[:-xf], mid, b[xf:]], axis=0)

def concatenate_with_crossfades(chunks, xf_ms, sr):
    """
    Concatenate chunks with crossfades between each pair.
    """
    if len(chunks) == 0:
        return np.zeros(1, dtype=np.float32)
    out = chunks[0]
    for c in chunks[1:]:
        out = xf_crossfade(out, c, xf_ms=xf_ms, sr=sr)
    return out

def gate_quiet(y, sr, gate_db=-38.0, frame_ms=30):
    """
    Downward expander style gating: zero out very quiet tails/heads.
    """
    frame = max(32, int(sr * frame_ms / 1000.0))
    hop = max(16, frame // 2)
    win = np.hanning(frame + 2)[1:-1]
    th = 10.0 ** (gate_db / 20.0)

    out = y.copy()
    for i in range(0, len(y) - frame, hop):
        seg = y[i:i+frame]
        rms = np.sqrt(np.mean(seg*seg) + 1e-12)
        if rms < th:
            out[i:i+frame] = 0.0 * win  # soft zero
    return out

def trim_silence_segments(y, sr, top_db=24, xf_ms=120):
    """
    Keep only non-silent parts and re-join with crossfades.
    """
    intervals = librosa.effects.split(y, top_db=float(top_db), frame_length=2048, hop_length=512)
    chunks = [y[s:e] for (s, e) in intervals]
    if len(chunks) == 0:
        return np.zeros(1, dtype=np.float32)
    return concatenate_with_crossfades(chunks, xf_ms=xf_ms, sr=sr)

def spectral_hold_sustain(y, sr, sustain_alpha=0.7, bass_hz=180.0):
    """
    Sustain/morph-like processing, but keep bass from the original
    to avoid 'thin' sound; apply exponential temporal hold on magnitude.
    sustain_alpha in [0..1]: closer to 1 = stronger sustaining (flatter troughs).
    """
    # STFT
    n_fft = 2048
    hop = 512
    win = "hann"
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=win)
    mag, phase = np.abs(S), np.angle(S)

    # temporal exponential hold on magnitude (per-bin EMA of the max-like envelope)
    # H[t,f] = max(H[t-1,f]*alpha, mag[t,f])
    H = np.zeros_like(mag)
    alpha = float(np.clip(sustain_alpha, 0.0, 0.999))
    for t in range(mag.shape[1]):
        if t == 0:
            H[:, t] = mag[:, t]
        else:
            H[:, t] = np.maximum(H[:, t-1] * alpha, mag[:, t])

    # Preserve low-frequency region from the original to keep body/bass.
    cutbin = band_limits_to_bin(bass_hz, sr=sr, n_fft=n_fft)
    if cutbin > 0:
        H_low  = mag[:cutbin, :]
        H[:cutbin, :] = 0.75 * H[:cutbin, :] + 0.25 * H_low  # gentle blend

    # Reconstruct
    S_new = H * np.exp(1j * phase)
    y_out = librosa.istft(S_new, hop_length=hop, window=win, length=len(y))
    return y_out.astype(np.float32)

# ----------------------------- #
# Processing modes              #
# ----------------------------- #

def process_blend(y, sr, params):
    top_db       = parse_float(params.get("top_db"), 24.0)
    xf_ms        = parse_float(params.get("xf_ms"), 120.0)
    inter_fade   = parse_float(params.get("inter_fade_ms"), 0.0)
    target_sec   = parse_float(params.get("target_sec"), 120.0)

    y1 = trim_silence_segments(y, sr, top_db=top_db, xf_ms=xf_ms)

    # optional inter-segment smoothing
    if inter_fade > 0:
        y1 = xf_crossfade(y1[:], y1[:], xf_ms=inter_fade, sr=sr)[:len(y1)]

    # fit/loop to target duration
    target_len = int(sr * target_sec)
    if len(y1) < target_len:
        reps = int(np.ceil(target_len / max(1, len(y1))))
        y1 = np.tile(y1, reps)
    y1 = y1[:target_len]
    return y1

def process_morph(y, sr, params):
    """
    Earlier 'morph' (spectral sustain + light compression).
    """
    morph_alpha  = parse_float(params.get("morph_alpha"), 0.7)
    target_sec   = parse_float(params.get("target_sec"), 120.0)
    comp_thresh  = parse_float(params.get("comp_thresh_db"), -24.0)
    comp_ratio   = parse_float(params.get("comp_ratio"), 3.0)

    y2 = spectral_hold_sustain(y, sr, sustain_alpha=morph_alpha, bass_hz=140.0)

    # simple soft-knee-ish compression
    thr = 10 ** (comp_thresh / 20.0)
    x = y2.copy()
    above = np.abs(x) > thr
    x[above] = np.sign(x[above]) * (thr + (np.abs(x[above]) - thr) / max(1e-6, comp_ratio))
    y2 = x

    target_len = int(sr * target_sec)
    if len(y2) < target_len:
        reps = int(np.ceil(target_len / max(1, len(y2))))
        y2 = np.tile(y2, reps)
    y2 = y2[:target_len]
    return y2

def process_sustain(y, sr, params):
    """
    New 'sustain' mode: stronger temporal hold + bass preservation
    + gating of very quiet tails + crossfades around joins.
    """
    sustain_alpha = parse_float(params.get("sustain_alpha"), 0.85)   # stronger by default
    bass_hz       = parse_float(params.get("bass_hz"), 180.0)
    gate_db       = parse_float(params.get("gate_db"), -38.0)
    target_sec    = parse_float(params.get("target_sec"), 120.0)
    xf_ms         = parse_float(params.get("xf_ms"), 500.0)          # generous crossfade
    inter_fade    = parse_float(params.get("inter_fade_ms"), 300.0)

    # Gate quiet tails first (skip wispy breaks)
    y_gated = gate_quiet(y, sr, gate_db=gate_db, frame_ms=30)

    # Spectral sustain with bass preservation
    y_sus = spectral_hold_sustain(y_gated, sr, sustain_alpha=sustain_alpha, bass_hz=bass_hz)

    # Re-join with itself (gentle 'glue' around joins) if requested
    if inter_fade > 0:
        y_sus = xf_crossfade(y_sus, y_sus, xf_ms=inter_fade, sr=sr)[:len(y_sus)]

    # Fit to target with long crossfades to hide joins
    target_len = int(sr * target_sec)
    if len(y_sus) < target_len:
        # tile with crossfades instead of blunt concatenation
        parts = []
        remain = target_len
        cur = y_sus
        while remain > 0:
            need = min(remain, len(cur))
            parts.append(cur[:need])
            remain -= need
            if remain > 0:
                cur = xf_crossfade(cur, y_sus, xf_ms=xf_ms, sr=sr)
        y_out = concatenate_with_crossfades(parts, xf_ms=xf_ms, sr=sr)
    else:
        y_out = y_sus[:target_len]

    return y_out

# ----------------------------- #
# Flask routes                  #
# ----------------------------- #

@app.route("/")
def home():
    return "SnoreEase processor is running"

@app.route("/process", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    # --------- memory-friendly defaults ---------
    # Downsample to 22050 mono to keep memory small on Free
    target_sr = 22050
    # Absolute hard cap on processed duration (seconds)
    hard_cap_sec = 210.0  # ~3.5 minutes

    # Load
    try:
        y, sr = librosa.load(io.BytesIO(f.read()), sr=target_sr, mono=True)
    except Exception as e:
        return jsonify({"error": f"Could not read audio: {e}"}), 400

    # Soft length cap to avoid Render memory restarts
    if len(y) > int(hard_cap_sec * target_sr):
        y = y[: int(hard_cap_sec * target_sr)]

    # Params
    mode          = (request.form.get("mode") or "sustain").strip().lower()
    normalize     = parse_bool(request.form.get("normalize"), True)
    target_dbfs   = parse_float(request.form.get("target_dbfs"), -18.0)

    # Bound target_sec so requests can’t push us over the hard cap
    user_target_sec = parse_float(request.form.get("target_sec"), 120.0)
    target_sec = min(user_target_sec, hard_cap_sec)

    # Put back the bounded target_sec for processing functions
    form = request.form.to_dict(flat=True)
    form["target_sec"] = str(target_sec)

    # Process by mode
    if mode == "blend":
        y_out = process_blend(y, sr, form)
    elif mode == "morph":
        y_out = process_morph(y, sr, form)
    else:
        # default to sustain if unknown
        y_out = process_sustain(y, sr, form)

    if normalize:
        y_out = normalize_to_dbfs(y_out, target_dbfs=target_dbfs)

    # Write to a BytesIO (avoid disk writes and keep it fast)
    buf = io.BytesIO()
    sf.write(buf, y_out, samplerate=sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=True,
        download_name="processed_snore.wav",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
