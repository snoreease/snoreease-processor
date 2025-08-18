from flask import Flask, request, jsonify, send_file
import librosa
import soundfile as sf
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "SnoreEase processor is running"

# ---------- helpers ----------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))

def normalize_to_target(x: np.ndarray, target_rms: float) -> np.ndarray:
    cur = rms(x)
    if cur <= 0:
        return x
    g = target_rms / cur
    y = x * g
    # avoid clipping
    y = np.clip(y, -1.0, 1.0)
    return y

def crossfade_concat(segments, fade_samples: int) -> np.ndarray:
    """Concatenate segments with linear crossfades."""
    if not segments:
        return np.array([], dtype=np.float32)
    out = segments[0].astype(np.float32).copy()
    for seg in segments[1:]:
        seg = seg.astype(np.float32)
        if fade_samples > 0 and len(out) > 0 and len(seg) > 0:
            n = min(fade_samples, len(out), len(seg))
            if n > 0:
                w = np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[-n:] = out[-n:] * (1.0 - w) + seg[:n] * w
                out = np.concatenate([out, seg[n:]])
            else:
                out = np.concatenate([out, seg])
        else:
            out = np.concatenate([out, seg])
    return out

def loop_to_length(y: np.ndarray, sr: int, target_sec: float, crossfade_ms: int) -> np.ndarray:
    """Repeat y with crossfades to reach target_sec; trims to exact length."""
    if target_sec is None or target_sec <= 0:
        return y
    target_len = int(target_sec * sr)
    if len(y) >= target_len:
        return y[:target_len]

    fade = int(sr * (crossfade_ms / 1000.0))
    pieces = [y]
    # how many repeats (subtract a fade to keep size growth reasonable)
    unit = max(1, len(y) - fade)
    repeats = int(np.ceil((target_len - len(y)) / unit)) + 1
    for _ in range(repeats - 1):
        pieces.append(y)

    out = pieces[0].astype(np.float32).copy()
    for seg in pieces[1:]:
        n = min(fade, len(out), len(seg))
        if n > 0:
            w = np.linspace(0.0, 1.0, n, dtype=np.float32)
            out[-n:] = out[-n:] * (1.0 - w) + seg[:n] * w
            out = np.concatenate([out, seg[n:].astype(np.float32)])
        else:
            out = np.concatenate([out, seg.astype(np.float32)])
    return out[:target_len]

# ---------- main processing ----------

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Optional tweakable params (sent as form fields)
    # Silence removal
    top_db = int(request.form.get('top_db', 20))            # higher => more aggressive silence removal
    frame_length = int(request.form.get('frame_length', 2048))
    hop_length = int(request.form.get('hop_length', 512))

    # Normalization / crossfades
    normalize = request.form.get('normalize', '1') not in ['0', 'false', 'False']
    inter_fade_ms = int(request.form.get('inter_fade_ms', 30))   # crossfade between bursts (ms)

    # Loop controls
    target_sec = request.form.get('target_sec', None)
    target_sec = float(target_sec) if target_sec is not None else None
    loop_fade_ms = int(request.form.get('loop_fade_ms', 50))     # crossfade between loop copies (ms)

    # Save input to disk
    input_path = "input.wav"
    output_path = "output.wav"
    file.save(input_path)

    # Load mono (keep native sample rate)
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # Find non-silent intervals
    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    if intervals.size == 0:
        # no detected audio, return original file
        sf.write(output_path, y, sr)
        return send_file(output_path, as_attachment=True)

    # Collect bursts and (optionally) normalize each to a common RMS
    bursts = []
    if normalize:
        rms_vals = []
        for start, end in intervals:
            seg = y[start:end]
            if len(seg) > 0:
                rms_vals.append(rms(seg))
        target = float(np.median(rms_vals)) if len(rms_vals) > 0 else rms(y)

        for start, end in intervals:
            seg = y[start:end]
            seg = normalize_to_target(seg, target)
            bursts.append(seg)
    else:
        for start, end in intervals:
            bursts.append(y[start:end])

    # Crossfade between bursts to reduce dips between snore intervals
    inter_fade = int(sr * (inter_fade_ms / 1000.0))
    y_processed = crossfade_concat(bursts, inter_fade)

    # Build a long loop if requested
    y_looped = loop_to_length(y_processed, sr, target_sec, loop_fade_ms)

    # Save & return
    sf.write(output_path, y_looped if target_sec else y_processed, sr)
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
