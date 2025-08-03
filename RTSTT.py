from numba import none
import sounddevice as sd
import numpy as np
import queue
import time
from faster_whisper import WhisperModel
from threading import Event

q = queue.Queue()
VOLUME_THRESHOLD_DB = -30  # dB
SILENCE_TIMEOUT = 1  # seconds
device_id = 12
sample_rate = 16000

mic_paused_event = Event()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    
    # Skip processing if microphone is paused
    if mic_paused_event.is_set():
        return

    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    volume_db = 20 * np.log10(volume_norm + 1e-6)

    if volume_db > VOLUME_THRESHOLD_DB:
        q.put(indata.copy())

def is_loud_enough(data):
    volume_norm = np.linalg.norm(data) / np.sqrt(len(data))
    volume_db = 20 * np.log10(volume_norm + 1e-6)
    return volume_db > VOLUME_THRESHOLD_DB

def start_transcription(callback, device_id=device_id, sample_rate=sample_rate):
    model = WhisperModel("distil-large-v3", device="cuda")

    chunk_duration = 1  # seconds
    chunk_samples = chunk_duration * sample_rate

    buffer = np.zeros(0, dtype=np.float32)
    last_voice_time = time.time()

    print(f"Using input device {device_id}... Press Ctrl+C to stop.")

    with sd.InputStream(samplerate=sample_rate, device=device_id, channels=1,
                        dtype='float32', callback=audio_callback):
        while True:
            # Skip processing if microphone is paused
            if mic_paused_event.is_set():
                time.sleep(0.1)
                continue

            try:
                data = q.get(timeout=0.1)
                if is_loud_enough(data):
                    buffer = np.concatenate((buffer, data.flatten()))
                    last_voice_time = time.time()
                elif time.time() - last_voice_time > SILENCE_TIMEOUT and len(buffer) > 0:
                    segments, _ = model.transcribe(buffer, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments])
                    if transcript.strip():  # Only send non-empty transcripts
                        print(f"[TRANSCRIPT] {transcript}")
                        callback(transcript)
                    buffer = np.zeros(0, dtype=np.float32)

            except queue.Empty:
                if buffer.size > 0 and len(buffer) >= chunk_samples:
                    segments, _ = model.transcribe(buffer, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments])
                    if transcript.strip():  # Only send non-empty transcripts
                        print(f"[TRANSCRIPT] {transcript}")
                        callback(transcript)
                    buffer = np.zeros(0, dtype=np.float32)
