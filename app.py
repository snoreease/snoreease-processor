from flask import Flask, request, jsonify, send_file
import os
import io
import numpy as np
import soundfile as sf
import librosa

app = Flask(__name__)

@app.route('/')
def home():
    return "SnoreEase processor is running"

def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sqrt(np.mean(x**2) + 1e-12)

def dbfs(x):
    return 20.0 * np.log10(rms(x) + 1e-12)

def apply_fade_in_out(x, fade_samps):
    n = len(x)
    if fade_samps <= 0 or fade_samps*2 >= n:
        return x
    w_in  = np.linspace(0.0, 1.0, fade_samps, dtype=np.float32)
    w_out = np.linspace(1.0, 0.0, fade_samps, dtype=np.float32)
    y = x.copy()
    y[:fade_samps]  *= w_in
    y[-fade_samps:] *= w_out
    return y

def crossfade_join(a, b, fade_samps):
    """Gain-match on overlap RMS, then linear crossfade."""
    if fade_samps <= 0:
        return np.concatenate([a, b])

    # choose overlap tails/heads (clamp if segments short)
    fa = min(fade_samps, len(a)//2, len(a))
    fb = min(fade_samps, len(b)//2, len(b))
    f  = min(fa, fb)
    if f <= 0:
        return np.concatenate([a, b])

    a_tail = a[-f:]
    b_head = b[:f]

    # match RMS of b_head to a_tail
    ra = rms(a_tail)
    rb = rms(b_head)
    gain = 1.0 if rb == 0 else (ra / rb)
    b_head = b_head * gain
    b_adj  = b.copy()
    b_adj[:f] = b_head

    # crossfade window
    w = np.linspace(0.0, 1.0, f, dtype=np.float32)
    mix = a_tail * (1.0 - w) + b_adj[:f] * w

    return np.concatenate([a[:-f], mix, b_adj[f:]])

def build_from_intervals(y, sr, top_db, target_sec, inter_fade_ms, wrap_fade_ms):
    # 1) find loud (non-silent) regions
    intervals = librosa.effects.split(y, top_db=top_db)
    segments = []
    for start, end in intervals:
        seg = y[start:end].astype(np.float32, copy=True)
        segments.append(seg)

    if not segments:
        # if nothing detected, just return silence of target length
        return np.zeros(int(target_sec * sr), dtype=np.float32)

    # 2) “inter-fade” each join between detected segments
    inter_fade = int(sr * (inter_fade_ms / 1000.0))
    seq = segments[0]
    seq = apply_fade_in_out(seq, min(inter_fade//2, len(seq)//2))

    for seg in segments[1:]:
        seg = apply_fade_in_out(seg, min(inter_fade//2, len(seg)//2))
        seq = crossfade_join(seq, seg, inter_fade)

    # 3) Tile up to the target length with a wrap-around crossfade
    wrap_fade = int(sr * (wrap_fade_ms / 1000.0))
    out = np.zeros(0, dtype=np.float32)
    while len(out) < int(target_sec * sr):
        if len(out) == 0:
            out = seq.copy()
        else:
            out = crossfade_join(out, seq, wrap_fade)

    # 4) Trim to exact target length
    need = int(target_sec * sr)
    if len(out) > need:
        out = out[:need]
    return out

def normalize_and_limit(x, target_dbfs=-18.0):
    # normalize to target RMS
    cur = dbfs(x)
    gain = 10 ** ((target_dbfs - cur) / 20.0)
    y = x * gain

    # simple soft limiter to tame peaks (tanh waveshaper)
    y = np.tanh(y * 1.2).astype(np.float32)
    # final safety clip
    y = np.clip(y, -0.999, 0.999)
    return y

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    input_path = "input.wav"
    f.save(input_path)

    # Read controls from form (with defaults)
    try:
        top_db        = float(request.form.get('top_db', '22'))   # trim tails harder with bigger number
        target_dbfs   = float(request.form.get('target_dbfs', '-18'))
        normalize     = request.form.get('normalize', 'true').lower() == 'true'
        target_sec    = float(request.form.get('target_sec', '30'))   # final loop length
        inter_fade_ms = float(request.form.get('inter_fade_ms', '250'))  # between intervals
        xf_ms         = float(request.form.get('xf_ms', '180'))         # wrap-around crossfade
    except Exception:
        return jsonify({"error": "Bad parameters"}), 400

    # Load audio (mono, keep native sr)
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # Build continuous stream
    y_out = build_from_intervals(
        y=y, sr=sr,
        top_db=top_db,
        target_sec=target_sec,
        inter_fade_ms=inter_fade_ms,
        wrap_fade_ms=xf_ms,
    )

    if normalize:
        y_out = normalize_and_limit(y_out, target_dbfs=target_dbfs)

    # Write to bytes and send
    buf = io.BytesIO()
    sf.write(buf, y_out, sr, subtype='PCM_16', format='WAV')
    buf.seek(0)
    return send_file(
        buf,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='processed.wav'
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
