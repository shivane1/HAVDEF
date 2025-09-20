import argparse      # For handling command-line arguments (like --file, --record)
import os            # For file system operations
import sys           # For system-level functions like exiting
import numpy as np   # For numerical operations (arrays, normalization)
import soundfile as sf   # For reading/writing audio files
import librosa       # For audio processing (loading, resampling, spectrograms)
import sounddevice as sd  # For recording audio from microphone

SAMPLE_RATE = 20000   # Target audio sample rate (20 kHz)
# Pre-trained Hugging Face model for audio deepfake detection
HF_MODEL_ID = "Kaustubh911/havdef-audio-detector"  

def info(msg): 
    print(f"[HAVDEF] {msg}")   # Logging helper to prefix messages with [HAVDEF]

def record_microphone(seconds: int, out_path: str, samplerate=SAMPLE_RATE):
    try:
        import sounddevice as sd   # Import sounddevice inside function
    except Exception as e:
        # If sounddevice not available, raise error (use --file instead)
        raise RuntimeError("sounddevice not available. Install it or use --file.") from e
    
    info(f"Recording {seconds}s from mic at {samplerate} Hz …")
    # Record audio: duration = seconds × sample rate
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()   # Wait until recording is finished
    sf.write(out_path, audio, samplerate)   # Save audio to file
    info(f"Saved raw recording -> {out_path}")
    return out_path   # Return saved file path

def preprocess_wav(in_wav: str, out_wav: str, target_sr=SAMPLE_RATE):
    # Load audio with original sample rate
    y, sr = librosa.load(in_wav, sr=None, mono=True)
    # Resample if input sr ≠ target sr
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # Normalize waveform between -1 and 1
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    # Save preprocessed audio
    sf.write(out_wav, y, target_sr)
    return out_wav   # Return new preprocessed file path

def trim_with_vad(in_wav: str, out_wav: str, sr=SAMPLE_RATE):
    """
    Uses Silero VAD (Voice Activity Detection) to keep only speech.
    Falls back to untrimmed audio if VAD fails.
    """
    try:
        import torch
        # Load Silero VAD model from torch hub
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        wav = read_audio(in_wav, sampling_rate=sr)   # Read audio waveform
        ts = get_speech_timestamps(wav, model, sampling_rate=sr)  # Detect speech segments
        if not ts:   # If no speech detected
            info("VAD found no speech; using the original audio.")
            return preprocess_wav(in_wav, out_wav, sr)
        # Collect only detected speech chunks
        speech = collect_chunks(ts, wav)
        sf.write(out_wav, speech, sr)   # Save trimmed speech
        return out_wav
    except Exception as e:
        # If VAD fails, fall back to just preprocessing
        info(f"VAD unavailable ({e}); using untrimmed audio.")
        return preprocess_wav(in_wav, out_wav, sr)

def load_detector(model_id: str = HF_MODEL_ID):
    from transformers import pipeline   # Import HF pipeline
    info(f"Loading detector: {model_id}")
    # Load pre-trained HF audio-classification pipeline
    clf = pipeline("audio-classification", model=model_id, device=-1, top_k=None)
    return clf   # Return classifier

def decide_from_labels(results, fake_threshold=0.5, real_threshold=0.5):
    """
    Decide final verdict based on labels + thresholds.
    - If fake score >= fake_threshold -> DEEPFAKE
    - Else if real score >= real_threshold -> REAL
    - Else fallback to comparing top scores.
    """
    score_fake, score_real = 0.0, 0.0
    top_fake, top_real = 0.0, 0.0

    # Loop through classifier results
    for r in results:
        lab = r["label"].lower()
        sc = float(r["score"])
        # If label matches fake-related words
        if any(k in lab for k in ["fake", "spoof", "synth", "ai", "generated"]):
            top_fake = max(top_fake, sc)
        # If label matches real-related words
        if any(k in lab for k in ["real", "bona", "bonafide", "human"]):
            top_real = max(top_real, sc)

    # Hard rules
    if top_fake >= fake_threshold:
        return "DEEPFAKE", top_fake
    if top_real >= real_threshold and top_fake < fake_threshold:
        return "REAL", top_real

    # Soft comparison if thresholds not met
    if top_fake > 0.0 or top_real > 0.0:
        if top_fake >= top_real:
            return "DEEPFAKE", top_fake
        else:
            return "REAL", top_real

    # Fallback: take top label if nothing matched
    top = max(results, key=lambda x: x["score"])
    if any(k in top["label"].lower() for k in ["fake", "spoof", "synth", "ai", "generated"]):
        return "DEEPFAKE", float(top["score"])
    return "REAL", float(top["score"])

def run_detection(path: str, model_id: str, fake_threshold: float = 0.5):
    # Step 1: Preprocess audio
    pre = preprocess_wav(path, "_preprocessed.wav", SAMPLE_RATE)
    # Step 2: Trim silence using VAD
    trimmed = trim_with_vad(pre, "_speech.wav", SAMPLE_RATE)
    # Step 3: Load detector model
    clf = load_detector(model_id)
    # Step 4: Run classifier
    results = clf(trimmed)
    # Step 5: Decide based on labels
    verdict, conf = decide_from_labels(results, fake_threshold=fake_threshold)
    # Log verdict
    info(f"Verdict: {verdict}  |  confidence ~ {conf:.3f}  |  fake_threshold={fake_threshold}")
    # Print top 5 labels
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:5]:
        print(f"  - {r['label']}: {r['score']:.3f}")
    return verdict, conf

def main():
    # Setup argument parser for CLI
    ap = argparse.ArgumentParser(description="HAVDEF demo: mic record + deepfake detection")
    g = ap.add_mutually_exclusive_group(required=True)  # Either record or file, not both
    g.add_argument("--record", type=int, help="Record N seconds from mic, then analyze.")
    g.add_argument("--file", type=str, help="Analyze an existing audio file (.wav/.mp3/.flac).")
    ap.add_argument("--model", type=str, default=HF_MODEL_ID, help="Hugging Face model id to use.")
    args = ap.parse_args()   # Parse command-line args

    if args.record is not None:
        raw = "_mic.wav"   # Save recorded file name
        record_microphone(args.record, raw, SAMPLE_RATE)   # Record audio
        run_detection(raw, args.model)   # Run detection
    else:
        # If file mode chosen but file not found
        if not os.path.exists(args.file):
            print("Input file not found. In Colab, upload a file and pass its path.")
            sys.exit(1)
        run_detection(args.file, args.model)   # Run detection on existing file

if __name__ == "__main__":
    main()   # Program entry point
