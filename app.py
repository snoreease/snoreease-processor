from flask import Flask, request, jsonify, send_file
import os
import numpy as np
import librosa
import soundfile as sf

app = Flask(__name__)

# ---- safety limits ----
MAX_TARGET_SEC = int(os.environ.get("MAX_TARGET_SEC", 300))  # cap at 5 min by default
DEFAULT_SR = int(os.environ.get("AUDIO_SR", 22050))          # downsample for speed/memory

# --------------------------
# helpers
# --------------------------
def dbfs(x):
    rms = np.sqrt(np.mean(np.maximum(x.astype(np.float64)**2, 1e-12)))
    return 20 * np.log10(rms + 1e-12)

def apply_gain_dbfs(x, target_db):
    cur = dbfs(x)
    gain_db = target_db - cur
    gain = 10 ** (gain_db / 20.0)
    y = x * gain
    mx = np.max(np.abs(y)) + 1e-12
    if mx > 1.0:
        y = y / mx * 0.999
    return y

def morph_spectrum(y, sr, alpha=0.6, n_fft=1024, hop_length=256):
    """Blend each frame’s spectrum toward the global median spectrum."""
    if alpha <= 0:
        return y
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    global_mag = np.median(mag, axis=1, keepdims=True)
    mag_blend = (1.0 - alpha) * mag + alpha * global_mag
    S_out = mag_blend * np.exp(1j * phase)
    y_out = librosa.istft(S_out, hop_length=hop_length)
    return y_out.astype(np.float32)

def level_compander(y, sr, thresh_db=-24.0, ratio=3.0,
                    attack_ms=15.0, release_ms=200.0):
    """Simple compressor to reduce dips."""
    if ratio <= 1.0:
        return y
    win = max(1, int(0.02 * sr))
    y2 = y.astype(np.float64) ** 2
    rms = np.sqrt(np.convolve(y2, np.ones(win) / win, mode="same") + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    over = rms_db - thresh_db
    gain_db = np.where(over > 0.0, -over * (1.0 - 1.0 / ratio), 0.0)

    atk = np.exp(-1.0 / max(1, int((attack_ms/1000.0)*sr)))
    rel = np.exp(-1.0 / max(1, int((release_ms/1000.0)*sr)))
    smoothed = np.zeros_like(gain_db)
    g = 0.0
    for i in range(len(gain_db)):
        target = gain_db[i]
        g = atk*g + (1-atk)*target if target < g else rel*g + (1-rel)*target
        smoothed[i] = g
    linear = 10.0 ** (smoothed / 20.0)
    out = y * linear.astype(y.dtype, copy=False)
    m = np.max(np.abs(out)) + 1e-9
    if m > 0.999:
        out = out / m * 0.999
    return out.astype(np.float32)

# --------------------------
# routes
# --------------------------
@app.route('/')
def home():
    return "SnoreEase processor is running"

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # ---- options from request ----
    mode            = request.form.get('mode', 'blend')
    target_sec_req  = float(request.form.get('target_sec', 120))
    target_sec      = min(target_sec_req, MAX_TARGET_SEC)
    xf_ms           = int(request.form.get('xf_ms', 120))
    inter_fade_ms   = int(request.form.get('inter_fade_ms', 200))
    top_db          = float(request.form.get('top_db', 21))
    normalize       = request.form.get('normalize', 'true').lower() == 'true'
    target_dbfs     = float(request.form.get('target_dbfs', -18))

    # morph + compressor settings
    morph_alpha     = float(request.form.get('morph_alpha', 0.6))
    n_fft           = int(request.form.get('n_fft', 1024))
    hop_length      = int(request.form.get('hop_length', 256))
    comp_thresh_db  = float(request.form.get('comp_thresh_db', -24))
    comp_ratio      = float(request.form.get('comp_ratio', 3.0))
    comp_attack_ms  = float(request.form.get('comp_attack_ms', 15))
    comp_release_ms = float(request.form.get('comp_release_ms', 200))

    # ---- save + load ----
    f = request.files['file']
    input_path  = "input.wav"
    output_path = "output.wav"
    f.save(input_path)

    try:
        y, sr = librosa.load(input_path, sr=DEFAULT_SR, mono=True)
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) == 0:
            return jsonify({"error": "No audible content found."}), 400
        parts = [y[s:e] for (s, e) in intervals]
        base  = np.concatenate(parts)

        # normalization
        if normalize:
            base = apply_gain_dbfs(base, target_dbfs)

        # optional spectral morph + compression
        if mode == "morph":
            base = morph_spectrum(base, sr, alpha=morph_alpha,
                                  n_fft=n_fft, hop_length=hop_length)
            base = level_compander(base, sr, thresh_db=comp_thresh_db,
                                   ratio=comp_ratio,
                                   attack_ms=comp_attack_ms,
                                   release_ms=comp_release_ms)

        # stream loop to disk
        target_samples = int(target_sec * sr)
        xf_samps       = int(xf_ms * sr / 1000)
        inter_samps    = int(inter_fade_ms * sr / 1000)

        with sf.SoundFile(output_path, mode='w', samplerate=sr, channels=1, subtype='PCM_16') as out:
            written = 0
            tail = np.zeros(0, dtype=np.float32)

            def write_one_pass():
                nonlocal written, tail
                x = base.copy()

                # small fades inside one pass
                if inter_samps > 0 and len(x) > 2*inter_samps:
                    fade = np.linspace(0, 1, inter_samps, dtype=np.float32)
                    x[:inter_samps]  *= fade
                    x[-inter_samps:] *= fade[::-1]

                if xf_samps > 0 and tail.size > 0:
                    n = min(xf_samps, tail.size, x.size)
                    fade_up   = np.linspace(0, 1, n, dtype=np.float32)
                    fade_down = 1.0 - fade_up
                    head = x[:n]*fade_up + tail[-n:]*fade_down
                    out.write(head)
                    written += n
                    x = x[n:]

                if x.size > 0:
                    out.write(x)
                    written += x.size

                if xf_samps > 0:
                    tail = x[-xf_samps:].copy() if x.size >= xf_samps else x.copy()
                else:
                    tail = np.zeros(0, dtype=np.float32)

            while written < target_samples:
                write_one_pass()

            if written > target_samples:
                with sf.SoundFile(output_path, 'r') as tmp:
                    audio = tmp.read(dtype='float32')
                audio = audio[:target_samples]
                sf.write(output_path, audio, sr, subtype='PCM_16')

        return send_file(output_path, as_attachment=True)

    except MemoryError:
        return jsonify({"error": "Out of memory"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
