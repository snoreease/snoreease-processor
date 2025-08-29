from flask import Flask, request, jsonify, send_file
import io
import os
import numpy as np
import soundfile as sf
import librosa

app = Flask(__name__)

# ---- Safe caps for a free container ----
# You can raise these later if you upgrade your instance.
CAP_SECONDS = int(os.getenv("CAP_SECONDS", "240"))   # max output duration (seconds)
MAX_UPLOAD_SECONDS = int(os.getenv("MAX_UPLOAD_SECONDS", "180"))  # max input length

# ------------- small utils -------------
def to_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32, copy=False)
    # avoid NaNs
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y

def dbfs(y: np.ndarray) -> float:
    peak = 1.0  # full-scale
    rms = np.sqrt(np.mean(np.square(y)) + 1e-12)
    return 20.0 * np.log10(rms / peak + 1e-12)

def normalize_to_dbfs(y: np.ndarray, target_dbfs: float = -18.0) -> np.ndarray:
    cur = dbfs(y)
    gain = 10 ** ((target_dbfs - cur) / 20.0)
    out = y * gain
    # protect against clipping
    mx = np.max(np.abs(out))
    if mx > 1.0:
        out = out / mx
    return out

def lin_fade(n: int):
    # linear crossfade windows that sum to 1
    w_in  = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float32)
    w_out = 1.0 - w_in
    return w_out, w_in

def xfade_append(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Append b to a with n-sample crossfade."""
    if n <= 0 or len(a) == 0:
        return np.concatenate([a, b])
    n = min(n, len(a), len(b))
    w_out, w_in = lin_fade(n)
    head = a[:-n]
    tail = a[-n:] * w_out + b[:n] * w_in
    rest = b[n:]
    return np.concatenate([head, tail, rest])

def concat_non_silent(y: np.ndarray, sr: int, top_db: float = 24.0,
                      inter_fade_ms: float = 200.0) -> np.ndarray:
    """Keep only non-silent chunks and join them with small crossfades."""
    if len(y) == 0:
        return y
    intervals = librosa.effects.split(y, top_db=float(top_db))
    if len(intervals) == 0:
        return y  # nothing detected, fall back
    nfade = int(sr * (inter_fade_ms / 1000.0))
    out = np.array([], dtype=np.float32)
    for i, (s, e) in enumerate(intervals):
        seg = y[s:e]
        if i == 0:
            out = seg.astype(np.float32)
        else:
            out = xfade_append(out, seg.astype(np.float32), nfade)
    return out

def loop_to_length(y: np.ndarray, sr: int, target_sec: float, xf_ms: float = 300.0) -> np.ndarray:
    """Loop y to target length using crossfades at the seams."""
    if len(y) == 0:
        return y
    target_n = int(sr * float(target_sec))
    if target_n <= len(y):
        return y[:target_n]

    xf = int(sr * (xf_ms / 1000.0))
    out = y.copy()
    while len(out) < target_n:
        out = xfade_append(out, y, xf)
        # small guard to avoid runaway memory
        if len(out) > target_n + sr * 5:
            break
    return out[:target_n]

# ----------------- MODES -----------------
def mode_blend(y: np.ndarray, sr: int,
               target_sec: float,
               xf_ms: float,
               inter_fade_ms: float,
               top_db: float) -> np.ndarray:
    # Remove very quiet tails and join significant parts
    y_ns = concat_non_silent(y, sr, top_db=top_db, inter_fade_ms=inter_fade_ms)
    out = loop_to_length(y_ns, sr, target_sec=target_sec, xf_ms=xf_ms)
    return out

def mode_morph(y: np.ndarray, sr: int,
               target_sec: float,
               xf_ms: float,
               inter_fade_ms: float,
               top_db: float,
               alpha: float = 0.7) -> np.ndarray:
    """
    Simple spectral smoothing across time (light) to soften troughs,
    then loop. (Kept for completeness; stronger option is 'sustain' below.)
    """
    # trim quiet tails first
    y_ns = concat_non_silent(y, sr, top_db=top_db, inter_fade_ms=inter_fade_ms)

    n_fft = 2048
    hop = 512
    S = librosa.stft(y_ns, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann")
    M = np.abs(S).astype(np.float32)
    P = np.angle(S)

    # temporal EMA on magnitudes
    M_s = M.copy()
    for t in range(1, M.shape[1]):
        M_s[:, t] = alpha * M[:, t] + (1.0 - alpha) * M_s[:, t - 1]

    S_new = M_s * np.exp(1j * P)
    y_new = librosa.istft(S_new, hop_length=hop, win_length=n_fft, window="hann", length=len(y_ns))
    out = loop_to_length(to_mono_float32(y_new), sr, target_sec=target_sec, xf_ms=xf_ms)
    return out

def mode_sustain(y: np.ndarray, sr: int,
                 target_sec: float,
                 xf_ms: float,
                 inter_fade_ms: float,
                 top_db: float,
                 sustain_alpha: float = 0.8,
                 bass_hz: float = 150.0,
                 gate_db: float = -35.0) -> np.ndarray:
    """
    Spectral Sustain:
    - Hold/refresh bass magnitudes (<= bass_hz) from loud frames.
    - During quiet frames, keep the last strong bass instead of letting it drop.
    - Then loop with long crossfades.
    """
    # 1) Cut quiet tails first so we don't sustain actual silence
    y_ns = concat_non_silent(y, sr, top_db=top_db, inter_fade_ms=inter_fade_ms)
    if len(y_ns) == 0:
        y_ns = y

    # 2) STFT
    n_fft = 2048
    hop = 512
    S = librosa.stft(y_ns, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann")
    M = np.abs(S).astype(np.float32)
    P = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bass_idx = np.where(freqs <= float(bass_hz))[0]

    # loudness per frame (dBFS-like)
    frame_rms = np.sqrt(np.mean(np.square(librosa.istft(S, hop_length=hop, win_length=n_fft,
                                                        window="hann", length=len(y_ns))) + 1e-12))
    # faster: approximate per-frame gate using spectral energy
    frame_energy = np.sqrt(np.mean(M**2, axis=0) + 1e-12)
    frame_db = 20.0 * np.log10(frame_energy + 1e-12)

    held = None
    for t in range(M.shape[1]):
        if frame_db[t] > float(gate_db):
            # refresh the held bass magnitudes
            held = M[bass_idx, t].copy()
        else:
            if held is not None:
                # mix held bass with current bass
                M[bass_idx, t] = sustain_alpha * held + (1.0 - sustain_alpha) * M[bass_idx, t]

    S_new = M * np.exp(1j * P)
    y_new = librosa.istft(S_new, hop_length=hop, win_length=n_fft, window="hann", length=len(y_ns))
    out = loop_to_length(to_mono_float32(y_new), sr, target_sec=target_sec, xf_ms=xf_ms)
    return out

# ------------- Flask routes -------------
@app.route("/")
def home():
    return "SnoreEase processor is running"

@app.route("/healthz")
def healthz():
    return "ok"

@app.route("/process", methods=["POST"])
def process_audio():
    # ---- Parse and guard ----
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # general params (with gentle defaults)
    mode = request.form.get("mode", "sustain").strip().lower()

    try:
        target_sec = float(request.form.get("target_sec", "120"))
        xf_ms = float(request.form.get("xf_ms", "400"))
        inter_fade_ms = float(request.form.get("inter_fade_ms", "300"))
        top_db = float(request.form.get("top_db", "24"))
    except Exception:
        return jsonify({"error": "Bad numeric parameters"}), 400

    normalize = str(request.form.get("normalize", "true")).lower() == "true"
    target_dbfs = float(request.form.get("target_dbfs", "-18"))

    # sustain-specific (safe defaults)
    sustain_alpha = float(request.form.get("sustain_alpha", "0.8"))
    bass_hz = float(request.form.get("bass_hz", "150"))
    gate_db = float(request.form.get("gate_db", "-35"))

    # caps
    if target_sec > CAP_SECONDS:
        return jsonify({"error": f"target_sec too large for this service (cap {CAP_SECONDS}s)."}), 400

    # ---- Load audio ----
    up = request.files["file"]
    # load without resampling to preserve bass; mono to save memory
    try:
        y, sr = librosa.load(up, sr=None, mono=True)  # librosa can read file-like objects
    except Exception as e:
        return jsonify({"error": f"Unable to read audio: {e}"}), 400

    y = to_mono_float32(y)
    # input length cap for safety
    if len(y) > (MAX_UPLOAD_SECONDS * (sr or 44100)):
        y = y[:MAX_UPLOAD_SECONDS * sr]

    # ---- Process by mode ----
    try:
        if mode == "blend":
            y_out = mode_blend(y, sr, target_sec, xf_ms, inter_fade_ms, top_db)
        elif mode == "morph":
            y_out = mode_morph(y, sr, target_sec, xf_ms, inter_fade_ms, top_db, alpha=0.7)
        elif mode == "sustain":
            y_out = mode_sustain(
                y, sr, target_sec, xf_ms, inter_fade_ms, top_db,
                sustain_alpha=sustain_alpha, bass_hz=bass_hz, gate_db=gate_db
            )
        else:
            # fallback: just loop trimmed content
            y_ns = concat_non_silent(y, sr, top_db=top_db, inter_fade_ms=inter_fade_ms)
            y_out = loop_to_length(y_ns, sr, target_sec, xf_ms)
    except MemoryError:
        return jsonify({"error": "Memory limit reached during processing. Try shorter target_sec."}), 502
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    # ---- Normalize & render ----
    if normalize:
        y_out = normalize_to_dbfs(y_out, target_dbfs=target_dbfs)

    buf = io.BytesIO()
    try:
        sf.write(buf, y_out, sr, format="WAV", subtype="PCM_16")
    except Exception as e:
        return jsonify({"error": f"Failed to write WAV: {e}"}), 500
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", as_attachment=True, download_name="output.wav")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
