# # # WORKING

import io
import wave
import queue
import threading
import resampy
import soundfile as sf
import sounddevice as sd
from piper import PiperVoice

voice = PiperVoice.load(
    model_path="voices/glados_piper_medium.onnx",
    config_path="voices/glados_piper_medium.onnx.json"
)
# sd.default.device = (12, 12) 
speech_queue = queue.Queue()
tts_done = threading.Event()

def _speak(text):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize(text, wav_file)

    buffer.seek(0)
    data, samplerate = sf.read(buffer, dtype='float32')

    # Resample
    target_samplerate = 48000
    if samplerate != target_samplerate:
        data = resampy.resample(data, samplerate, target_samplerate)
        samplerate = target_samplerate

    sd.play(data, samplerate) 
    sd.wait()

def tts_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        _speak(text)
    tts_done.set()  # Let main thread know we're done

def stream_speech():
    """Starts TTS queue processing"""
    tts_done.clear()
    threading.Thread(target=tts_worker, daemon=True).start()

def say_chunk(text):
    """Adds text to the speech queue"""
    speech_queue.put(text)

def finish_speech():
    """Closes the queue and waits for TTS to finish"""
    speech_queue.put(None)
    tts_done.wait()
