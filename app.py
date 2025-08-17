from flask import Flask, request, jsonify, send_file
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

    # 🔁 Make it loop smoothly (repeat the snore several times)
    num_loops = 10  # you can adjust this number for longer loops
    y_looped = np.tile(y_processed, num_loops)

    # Save processed audio
    sf.write(output_path, y_looped, sr)

    # Return processed file
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
