from flask import Flask, request, jsonify, send_file
import os
import numpy as np
import librosa
import soundfile as sf

app = Flask(__name__)

# ---- safety limits (tweakable) ----
MAX_TARGET_SEC = int(os.environ.get("MAX_TARGET_SEC", 300))   # hard cap, e.g., 5 minutes
DEFAULT_SR      = int(os.environ.get("AUDIO_SR", 22050))      # lower SR = lower RAM/CPU

def dbfs(x):
    rms = np.sqrt(np.mean(np.maximum(x.astype(np.float64)**2, 1e-12)))
    return 20 * np.log10(rms + 1e-12)

def apply_gain_dbfs(x, target_db):
    cur = dbfs(x)
    gain_db = target_db - cur
    gain = 10 ** (gain_db / 20.0)
    y = x * gain
    # avoid clipping
    mx = np.max(np.abs(y)) + 1e-12
    if mx > 1.0:
        y = y / mx * 0.999
    return y

@app.route('/')
def home():
    return "SnoreEase processor is running"

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # ---- read request options (with safe defaults) ----
    mode            = request.form.get('mode', 'blend')
    target_sec_req  = float(request.form.get('target_sec', 120))     # user ask
    target_sec      = min(target_sec_req, MAX_TARGET_SEC)            # cap length
    xf_ms           = int(request.form.get('xf_ms', 120))            # crossfade ms at joins
    inter_fade_ms   = int(request.form.get('inter_fade_ms', 200))    # extra smoothing
    top_db          = float(request.form.get('top_db', 21))          # silence threshold
    normalize       = request.form.get('normalize', 'true').lower() == 'true'
    target_dbfs     = float(request.form.get('target_dbfs', -18))

    # ---- save upload to disk ----
    f = request.files['file']
    input_path  = "input.wav"
    output_path = "output.wav"
    f.save(input_path)

    try:
        # ---- load mono at reduced SR ----
        y, sr = librosa.load(input_path, sr=DEFAULT_SR, mono=True)

        # ---- keep only non-silent parts (doesn't duplicate to huge loops) ----
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) == 0:
            return jsonify({"error": "No audible content found."}), 400

        # stitch non-silent parts ONCE (short – typically a few MB, OK)
        parts = [y[s:e] for (s, e) in intervals]
        base  = np.concatenate(parts)

        # normalize base if requested
        if normalize:
            base = apply_gain_dbfs(base, target_dbfs)

        # ---- stream the loop to disk (no giant arrays) ----
        target_samples = int(target_sec * sr)
        xf_samps       = int(xf_ms * sr / 1000)
        inter_samps    = int(inter_fade_ms * sr / 1000)

        with sf.SoundFile(output_path, mode='w', samplerate=sr, channels=1, subtype='PCM_16') as out:
            written = 0
            tail = np.zeros(0, dtype=np.float32)

            # a helper to write one “pass” of the base with crossfades
            def write_one_pass():
                nonlocal written, tail
                x = base

                # optional tiny pre/post smoothing inside the pass
                if inter_samps > 0 and len(x) > 2 * inter_samps:
                    fade = np.linspace(0, 1, inter_samps, dtype=np.float32)
                    x[:inter_samps]  *= fade
                    x[-inter_samps:] *= fade[::-1]

                # crossfade with previous tail (overlap-add)
                if xf_samps > 0 and tail.size > 0:
                    n = min(xf_samps, tail.size, x.size)
                    fade_up   = np.linspace(0, 1, n, dtype=np.float32)
                    fade_down = 1.0 - fade_up
                    head = x[:n] * fade_up + tail[-n:] * fade_down
                    out.write(head)
                    written += n
                    x = x[n:]

                # write the rest
                if x.size > 0:
                    out.write(x)
                    written += x.size

                # keep tail for next crossfade
                if xf_samps > 0:
                    tail = x[-xf_samps:].copy() if x.size >= xf_samps else x.copy()
                else:
                    tail = np.zeros(0, dtype=np.float32)

            # loop until we reach target samples
            while written < target_samples:
                write_one_pass()

            # trim any extra samples beyond target
            if written > target_samples:
                # soundfile has no “truncate”, so re-open and rewrite trimmed head
                # (rare: only a small overrun)
                with sf.SoundFile(output_path, 'r') as tmp:
                    audio = tmp.read(dtype='float32')
                audio = audio[:target_samples]
                sf.write(output_path, audio, sr, subtype='PCM_16')

        # send the file back
        return send_file(output_path, as_attachment=True)

    except MemoryError:
        return jsonify({"error": "Out of memory; try a shorter target length or lower sample rate."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
