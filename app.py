from flask import Flask, request, jsonify, send_file
import os
import io
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

app = Flask(__name__)

# ---------------------------
# Helpers
# ---------------------------

def db_to_gain(db):
    return 10.0 ** (db / 20.0)

def gain_to_db(g):
    g = np.maximum(1e-12, np.abs(g))
    return 20.0 * np.log10(g)

def normalise_to_dbfs(y, target_dbfs=-18.0):
    peak = np.max(np.abs(y)) + 1e-12
    current_dbfs = gain_to_db(peak)
    diff = target_dbfs - current_dbfs
    return y * db_to_gain(diff)

def detect_intervals(y, sr, top_db=24):
    # Keep “non-silent” parts; high top_db = more lenient (keeps more tails)
    intervals = librosa.effects.split(y, top_db=top_db)
    return intervals

def concat_intervals(y, intervals):
    if len(intervals) == 0:
        return np.zeros(1, dtype=np.float32)
    parts = [y[s:e] for s, e in intervals]
    out = np.concatenate(parts)
    return out.astype(np.float32, copy=False)

def build_coloured_noise_like(y, sr, secs=None):
    """
    Create coloured noise with the average magnitude spectrum of y.
    """
    if secs is None:
        N = y.size
    else:
        N = int(secs * sr)
    # average spectrum
    n_fft = 1024
    hop = n_fft // 4
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))  # (freq, frames)
    mean_mag = np.mean(S, axis=1) + 1e-12
    # white noise -> STFT -> apply mean_mag -> iSTFT
    w = np.random.randn(N).astype(np.float32)
    W = librosa.stft(w, n_fft=n_fft, hop_length=hop)
    phase = np.exp(1j * np.angle(W))
    # normalise each frame then apply target spectrum
    W_mag = np.abs(W) + 1e-12
    W_col = phase * (W / W_mag)  # unit-mag with original phase
    # scale rows to match mean_mag
    scale = (mean_mag / np.mean(W_mag, axis=1, keepdims=False)).reshape(-1, 1)
    W_col = np.abs(W_col) * scale * phase
    z = librosa.istft(W_col, hop_length=hop, length=N)
    return z.astype(np.float32, copy=False)

def loop_to_seconds(y, sr, target_sec=120, xf_ms=300, inter_fade_ms=300):
    """
    Tile y to target_sec with crossfades to hide seams.
    """
    if y.size == 0:
        return y
    xf = int(xf_ms * sr / 1000)
    inter = int(inter_fade_ms * sr / 1000)
    target_len = int(target_sec * sr)
    chunks = []
    cur = 0
    first = True
    while cur < target_len:
        take = min(y.size, target_len - cur)
        seg = y[:take].copy()
        if not first and xf > 0 and len(chunks) > 0:
            # crossfade with previous end
            prev = chunks[-1]
            L = min(xf, seg.size, prev.size)
            win = np.linspace(0, 1, L, dtype=np.float32)
            prev[-L:] = prev[-L:] * (1 - win) + seg[:L] * win
            seg = seg[L:]
            chunks[-1] = prev
        chunks.append(seg)
        # small inter-fade dip between repeats (optional)
        if inter > 0 and (cur + take) < target_len:
            pad = np.zeros(min(inter, target_len - (cur + take)), dtype=np.float32)
            chunks.append(pad)
        cur += take + (0 if inter <= 0 else min(inter, target_len - (cur + take)))
        first = False
    out = np.concatenate(chunks) if len(chunks) else y
    return out

def apply_reverb(y, sr, reverb_ms=200.0, decay=0.35, mix=0.2):
    """
    Very light exponential IR reverb. mix in [0..1]
    """
    reverb_ms = max(0.0, float(reverb_ms))
    mix = float(np.clip(mix, 0.0, 1.0))
    if reverb_ms <= 1 or mix <= 0:
        return y
    ir_len = int(sr * reverb_ms / 1000.0)
    t = np.arange(ir_len, dtype=np.float32)
    ir = (decay ** (t / (ir_len + 1.0))).astype(np.float32)
    ir[0] = 1.0  # direct
    wet = fftconvolve(y, ir, mode="full")[: y.size]
    return (1 - mix) * y + mix * wet

# ---------------------------
# Core processors
# ---------------------------

def process_sustain(y, sr,
                    sustain_alpha=0.9,   # more floor (0..1)  higher = keep more original peaks
                    gate_db=-50.0,       # relax gate threshold (less negative keeps more)
                    floor_db=-40.0,      # level of floor vs full scale
                    reverb_ms=0.0,       # optional reverb
                    reverb_decay=0.35,
                    reverb_mix=0.0):
    """
    Fill troughs using a coloured floor and optional reverb/echo.
    """
    # 1) Keep “content” segments; gate_db affects how much tail we keep
    intervals = detect_intervals(y, sr, top_db=abs(gate_db))
    core = concat_intervals(y, intervals)

    # 2) Build coloured floor (snore-shaped) for the same duration
    floor = build_coloured_noise_like(core, sr, secs=len(core)/sr)

    # 3) Scale floor to a chosen dBFS
    floor = floor * db_to_gain(floor_db)

    # 4) Mix: sustain_alpha keeps core, (1 - alpha) adds floor
    out = sustain_alpha * core + (1.0 - sustain_alpha) * floor

    # 5) Optional reverb/echo to fill micro-gaps
    if reverb_ms > 0.0 and reverb_mix > 0.0:
        out = apply_reverb(out, sr, reverb_ms=reverb_ms, decay=reverb_decay, mix=reverb_mix)

    return out.astype(np.float32, copy=False)

# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def home():
    return "SnoreEase processor is running"

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Read bytes -> librosa
    raw = f.read()
    y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)

    # Common params (with defaults)
    target_sec     = float(request.form.get('target_sec', 120))
    xf_ms          = float(request.form.get('xf_ms', 300))
    inter_fade_ms  = float(request.form.get('inter_fade_ms', 300))
    normalize      = request.form.get('normalize', 'true').lower() == 'true'
    target_dbfs    = float(request.form.get('target_dbfs', -18))
    top_db         = float(request.form.get('top_db', 24))  # used only in “concat only” baseline
    mode           = request.form.get('mode', 'sustain').lower()

    # Sustain-specific knobs
    sustain_alpha  = float(request.form.get('sustain_alpha', 0.95))
    gate_db        = float(request.form.get('gate_db', -50.0))        # less negative = fewer troughs
    floor_db       = float(request.form.get('floor_db', -40.0))       # raise if you still hear dips

    # Reverb / echo
    reverb_ms      = float(request.form.get('reverb_ms', 180.0))      # 0 to disable
    reverb_decay   = float(request.form.get('reverb_decay', 0.35))
    reverb_mix     = float(request.form.get('reverb_mix', 0.20))      # 0..1

    # --- Step A: create a continuous snore stream based on mode ---
    if mode == 'sustain':
        stream = process_sustain(y, sr,
                                 sustain_alpha=sustain_alpha,
                                 gate_db=gate_db,
                                 floor_db=floor_db,
                                 reverb_ms=reverb_ms,
                                 reverb_decay=reverb_decay,
                                 reverb_mix=reverb_mix)
    else:
        # Baseline: just keep non-silent parts
        intervals = detect_intervals(y, sr, top_db=top_db)
        stream = concat_intervals(y, intervals)

    # --- Step B: Loop it nicely to the target length ---
    stream = loop_to_seconds(stream, sr,
                             target_sec=target_sec,
                             xf_ms=xf_ms,
                             inter_fade_ms=inter_fade_ms)

    # --- Step C: Normalise if requested ---
    if normalize:
        stream = normalise_to_dbfs(stream, target_dbfs=target_dbfs)

    # Write to BytesIO to avoid disk churn
    buf = io.BytesIO()
    sf.write(buf, stream, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav',
                     as_attachment=True, download_name='processed.wav')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
