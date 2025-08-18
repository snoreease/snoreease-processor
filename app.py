from flask import Flask, request, jsonify, send_file
import librosa
import soundfile as sf
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# Helpers
# -----------------------------
def parse_bool(s, default=False):
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "on")

def parse_float(s, default):
    try:
        return float(s)
    except Exception:
        return default

def rms_linear(x):
    return np.sqrt(np.mean(np.square(x))) + 1e-12  # avoid div-by-zero

def normalize_to_target_rms(y, target_dbfs=-20.0):
    """
    Normalize a signal 'y' so its RMS ~= target_dbfs (e.g. -20 dBFS).
    0 dBFS means full-scale (|y|=1). -20 dBFS ~ 0.1 RMS.
    """
    target_lin = 10.0 ** (target_dbfs / 20.0)
    cur_rms = rms_linear(y)
    y_out = y * (target_lin / cur_rms)
    # hard clip to avoid intersample peaks after gains
    return np.clip(y_out, -1.0, 1.0)

def linear_fade_in(n):
    return np.linspace(0.0, 1.0, num=n, endpoint=True)

def linear_fade_out(n):
    return np.linspace(1.0, 0.0, num=n, endpoint=True)

def crossfade_concat(segments, sr, xf_ms=50):
    """
    Concatenate segments with linear crossfades.
    xf_ms: crossfade length in milliseconds.
    """
    if len(segments) == 0:
        return np.array([], dtype=np.float32)
    if len(segments) == 1:
        return segments[0].astype(np.float32)

    xf_samps = max(1, int(sr * (xf_ms / 1000.0)))
    out = segments[0].astype(np.float32)

    for idx in range(1, len(segments)):
        a = out
        b = segments[idx].astype(np.float32)

        if len(a) < xf_samps or len(b) < xf_samps:
            # If a segment is very short, just butt-join
            out = np.concatenate([a, b])
            continue

        # overlap last xf_samps of a with first xf_samps of b
        a_head = a[:-xf_samps]
        a_tail = a[-xf_samps:]
        b_head = b[:xf_samps]
        b_tail = b[xf_samps:]

        fadeA = linear_fade_out(xf_samps).astype(np.float32)
        fadeB = linear_fade_in(xf_samps).astype(np.float32)

        cross = (a_tail * fadeA) + (b_head * fadeB)
        out = np.concatenate([a_head, cross, b_tail])

    return out

def detect_non_silent_segments(y, top_db=20):
    """
    Return list of time-domain segments for non-silent portions.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return [y]
    return [y[s:e] for (s, e) in intervals]

def build_loop(y, sr, target_sec=30, xf_ms=50, top_db=20, normalize=True, target_dbfs=-20.0):
    """
    Your original 'loop' idea:
      - Grab non-silent pieces
      - (optional) normalize each piece to same loudness
      - crossfade join them
      - tile to fill target length, then trim
    """
    segs = detect_non_silent_segments(y, top_db=top_db)
    if normalize:
        segs = [normalize_to_target_rms(s, target_dbfs) for s in segs]

    joined = crossfade_concat(segs, sr, xf_ms=xf_ms)

    # tile to reach target length
    target_samples = int(sr * target_sec)
    if len(joined) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    reps = max(1, int(np.ceil(target_samples / len(joined))))
    looped = np.tile(joined, reps).astype(np.float32)
    looped = looped[:target_samples]

    return looped

def build_blend(y, sr, target_sec=30, xf_ms=50, top_db=20, normalize=True, target_dbfs=-20.0):
    """
    'Strongest snore blend':
      - take non-silent snore intervals
      - normalize all intervals to the same RMS to remove dips
      - sort by loudness (RMS) so the “strong” character dominates
      - crossfade join them
      - tile + trim to target_sec
    This makes a steady, full sound without obvious level dips.
    """
    segs = detect_non_silent_segments(y, top_db=top_db)

    if len(segs) == 0:
        target_samples = int(sr * target_sec)
        return np.zeros(target_samples, dtype=np.float32)

    # Normalize every interval so they sit at the same loudness
    if normalize:
        segs = [normalize_to_target_rms(s, target_dbfs) for s in segs]

    # Sort by their RMS (descending): densest/strongest intervals first
    segs = sorted(segs, key=lambda s: rms_linear(s), reverse=True)

    # Crossfade-join once, then tile to target length
    base = crossfade_concat(segs, sr, xf_ms=xf_ms)

    target_samples = int(sr * target_sec)
    if len(base) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    reps = max(1, int(np.ceil(target_samples / len(base))))
    strong = np.tile(base, reps).astype(np.float32)
    strong = strong[:target_samples]

    # Safety normalize the final output a little (prevent minor overs)
    peak = np.max(np.abs(strong)) + 1e-12
    if peak > 1.0:
        strong = strong / peak

    return strong

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return "SnoreEase processor is running"

@app.route('/process', methods=['POST'])
def process_audio():
    # Validate upload
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    input_path = "input.wav"
    output_path = "output.wav"
    f.save(input_path)

    # Parameters (all optional)
    mode = request.form.get("mode", "loop").strip().lower()  # "loop" or "blend"
    target_sec = parse_float(request.form.get("target_sec"), 30.0)
    xf_ms = parse_float(request.form.get("xf_ms"), 50.0)            # crossfade ms
    top_db = parse_float(request.form.get("top_db"), 20.0)           # silence threshold
    normalize = parse_bool(request.form.get("normalize"), True)      # level each piece
    target_dbfs = parse_float(request.form.get("target_dbfs"), -20.0)  # RMS target

    # Load mono at native sampling rate
    y, sr = librosa.load(input_path, sr=None, mono=True)

    if mode == "blend":
        y_out = build_blend(
            y, sr,
            target_sec=target_sec,
            xf_ms=xf_ms,
            top_db=top_db,
            normalize=normalize,
            target_dbfs=target_dbfs
        )
    else:
        # default = loop
        y_out = build_loop(
            y, sr,
            target_sec=target_sec,
            xf_ms=xf_ms,
            top_db=top_db,
            normalize=normalize,
            target_dbfs=target_dbfs
        )

    # Write WAV
    sf.write(output_path, y_out, sr)

    # Return the file as an attachment
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
