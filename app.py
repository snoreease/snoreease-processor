from flask import Flask, request, jsonify
import librosa
import soundfile as sf
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "SnoreEase processor is running"

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    input_path = "input.wav"
    output_path = "output.wav"
    file.save(input_path)

    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    # Detect non-silent parts
    non_silent_intervals = librosa.effects.split(y, top_db=20)

    # Concatenate only non-silent parts
    y_processed = np.concatenate([y[start:end] for start, end in non_silent_intervals])

    # Save processed audio
    sf.write(output_path, y_processed, sr)

    return jsonify({"message": "Processing complete", "output_file": output_path})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
